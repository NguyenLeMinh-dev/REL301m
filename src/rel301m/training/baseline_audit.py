"""RNG-neutral probes of fixed training/validation subsets and optimizer schedules."""
from copy import deepcopy
import json
import numpy as np
import torch

from rel301m.imitation.demo_finetune import DemonstrationReplay
from rel301m.utils.logger import parameter_sha256


COUNTER_FIELDS = ['critic_optimizer_updates', 'actor_optimizer_updates_0', 'actor_optimizer_updates_1',
                  'actor_optimizer_steps_total', 'alpha_optimizer_updates']
DRIFT_FIELDS = [f'{split}_{metric}_{i}' for split in ('train', 'validation')
                for metric in ('expert_mse', 'bc_drift_mse') for i in range(2)]
AUDIT_METRICS = ['q_loss', 'q1_mean', 'q2_mean', 'target_q_mean', 'td_abs_p50', 'td_abs_p95', 'td_abs_p99',
                 *[key for i in range(2) for key in (f'actor_{i}_loss', f'actor_{i}_grad_norm',
                   f'alpha_{i}', f'entropy_{i}', f'action_std_{i}', f'action_saturation_{i}')]]


class BCReferencePolicy:
    def __init__(self, actors):
        self.actors = actors

    @torch.no_grad()
    def act(self, observations, deterministic=False):
        return tuple(actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=actor.action_scale.device),
                                  deterministic)[0].cpu().numpy() for actor, obs in zip(self.actors, observations))


def validate_audit(config):
    audit = config.get('audit')
    if not audit:
        return
    if not isinstance(audit, dict) or not config.get('bc_checkpoint'):
        raise ValueError('Baseline audit requires a mapping and a BC checkpoint')
    if config['total_steps'] > 5000:
        raise ValueError('Baseline audit is capped at 5k; long-pilot gates remain unresolved')
    required = {'dataset', 'interval_updates', 'subset_size', 'subset_seed', 'early_eval_steps',
                'early_eval_episodes', 'stochastic_eval_episodes'}
    if set(audit) != required:
        raise ValueError('Baseline audit keys mismatch')
    for key in ('interval_updates', 'subset_size', 'early_eval_episodes', 'stochastic_eval_episodes'):
        if type(audit[key]) is not int or audit[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(audit['subset_seed']) is not int or audit['subset_seed'] < 0:
        raise ValueError('Invalid subset seed')
    if not isinstance(audit['early_eval_steps'], list) or any(type(n) is not int or n <= 0 for n in audit['early_eval_steps']):
        raise ValueError('Invalid early evaluation steps')
    if max(audit['early_eval_episodes'], audit['stochastic_eval_episodes']) > config['final_eval_episodes']:
        raise ValueError('Diagnostic episodes must fit the paired random-reference prefix')


def actor_sha(model):
    return [parameter_sha256(actor) for actor in model.actors]


class BaselineAudit:
    def __init__(self, model, reference_actors, demos, settings, run_dir):
        self.reference = deepcopy(reference_actors)
        self.reference.requires_grad_(False)
        self.settings, self.run_dir = settings, run_dir
        self.freeze_sha = actor_sha(model)
        self.freeze_alpha = model.log_alpha.detach().clone()
        self.first_actor_update = False
        rng = np.random.default_rng(settings['subset_seed'])
        self.probes, indices = {}, {}
        for split in ('train', 'validation'):
            size = len(demos.replay) if split == 'train' else len(demos.validation[0][0])
            chosen = rng.choice(size, min(size, settings['subset_size']), replace=False)
            indices[split] = chosen.tolist()
            tensors = []
            for i in range(2):
                if split == 'train':
                    obs, actions = (demos.replay.arrays[f'{key}{i}'][chosen] for key in ('o', 'a'))
                    tensors.append(tuple(torch.as_tensor(value, device=model.device) for value in (obs, actions)))
                else:
                    tensors.append(tuple(value[chosen] for value in demos.validation[i]))
            self.probes[split] = tensors
        self.metadata = dict(demonstrations=demos.metadata, subset_transition_indices=indices,
                             subset_seed=settings['subset_seed'], reference='loaded BC mean before any std override',
                             metrics_timing='loss/Q/TD pre-update; drift post-update', rng_neutral=True,
                             actor_sha_before_freeze=self.freeze_sha)
        (run_dir / 'audit_subset.json').write_text(json.dumps(self.metadata, indent=2)+'\n')
        self.event('freeze_start', model, 0)

    def event(self, name, model, step):
        row = dict(event=name, environment_steps=step, **model.optimizer_counts(),
                   actor_sha256=actor_sha(model), alpha=model.alpha.detach().cpu().tolist())
        with (self.run_dir / 'optimizer_events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def assert_frozen(self, model, step):
        if actor_sha(model) != self.freeze_sha or not torch.equal(model.log_alpha.detach(), self.freeze_alpha):
            raise RuntimeError('Actor/alpha changed during critic-only freeze')
        self.event('freeze_end', model, step)

    def before_update(self, model, step, update_actor):
        if update_actor and not self.first_actor_update:
            self.assert_frozen(model, step)
            self.first_actor_update = True
            self.event('first_actor_update_pending', model, step)

    @torch.no_grad()
    def drift(self, model):
        result = {}
        for split, probes in self.probes.items():
            for i, (obs, expert) in enumerate(probes):
                actor, reference = model.actors[i], self.reference[i]
                mean = actor(obs)[0]
                prediction = actor.action_bias + actor.action_scale * mean.tanh()
                old_mean = reference(obs)[0]
                old = reference.action_bias + reference.action_scale * old_mean.tanh()
                result[f'{split}_expert_mse_{i}'] = (prediction - expert).square().mean().item()
                result[f'{split}_bc_drift_mse_{i}'] = (prediction - old).square().mean().item()
        return result

    def row(self, model, step, metrics):
        result = dict(environment_steps=step, updates=model.updates, **model.optimizer_counts(), **self.drift(model))
        result.update({key: metrics.get(key) for key in AUDIT_METRICS})
        return result
