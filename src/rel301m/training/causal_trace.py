"""Optional observer installed only by the causal-audit CLI, never by normal train."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from rel301m.algorithms.masac import MASAC
from rel301m.algorithms.replay_buffer import ReplayBuffer
from rel301m.imitation.demo_finetune import DemonstrationReplay
from rel301m.imitation.demonstrations import load_dataset
from .causal_tools import (cpu_tree, tree_hash, rng_state, peek_indices, composition,
                           component_gradient, cosine, flatten, adam_expected_delta, policy_probe)


MILESTONES = {1, 2, 5, 10, 20, 50, 100}
REPLAY_STEPS = {0, 100, 500, 1000, 1100, 2000}


class CausalTrace:
    def __init__(self, dataset, checkpoint, output, subset_size=256):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.manifest, self.episodes = load_dataset(dataset)
        bc = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.split = bc['metadata']['episode_split']
        self.labels = [('demo', episode, row) for episode in self.split['train']
                       for row in range(len(self.episodes[episode]['r']))]
        self.subset_size = subset_size
        self.main, self.demos, self.model = None, None, None
        self.in_demos, self.in_prefill = False, False
        self.sample_log, self.records, self.replay_records = [], [], []
        self.actor_context, self.q_outputs, self.mixed_sources = {}, {}, None
        self.snapshot_handles = []
        self.probes, self.reference = None, None
        self.executed_actor_updates = 0

    def replay_inventory(self):
        sources = self.main._causal_sources[:len(self.main)]
        return dict(main_demo_count=sum(source[0]=='demo' for source in sources),
                    main_online_count=sum(source[0]=='online' for source in sources))

    def _initialize_model(self, model):
        self.model = model
        self.reference = deepcopy(model.actors)
        self.reference.requires_grad_(False)
        rng = np.random.default_rng(20261002)
        indices = {}
        self.probes = {}
        for split in ['train', 'validation']:
            self.probes[split] = []
            lengths = [len(self.episodes[episode]['r']) for episode in self.split[split]]
            chosen = rng.choice(sum(lengths), min(self.subset_size, sum(lengths)), replace=False)
            indices[split] = chosen.tolist()
            for i in range(2):
                arrays = [np.concatenate([self.episodes[e][f'{key}{i}'] for e in self.split[split]])[chosen]
                          for key in ['o', 'a']]
                self.probes[split].append(tuple(torch.as_tensor(a, device=model.device) for a in arrays))
        (self.output/'probe_indices.json').write_text(json.dumps(indices, indent=2)+'\n')
        def hook(name):
            def observe(module, inputs, result):
                self.q_outputs[name] = (inputs, result)
            return observe
        self.snapshot_handles = [model.q1.register_forward_hook(hook('q1')),
                                 model.q2.register_forward_hook(hook('q2'))]
        for i, optimizer in enumerate(model.actor_optimizers):
            original_step = optimizer.step
            def observed_step(*args, _i=i, _original=original_step, **kwargs):
                return self._actor_step(_i, _original, args, kwargs)
            optimizer.step = observed_step

    def _actor_loss(self, original, model, index, batch, **kwargs):
        loss, log_probability = original(model, index, batch, **kwargs)
        if not self.active:
            return loss, log_probability
        actor = model.actors[index]
        if any(p.grad is not None for p in actor.parameters()):
            raise AssertionError('Actor gradient not clean before backward')
        q1, q2 = self.q_outputs['q1'][1], self.q_outputs['q2'][1]
        q_term = -torch.minimum(q1, q2).mean()
        entropy = (model.alpha[index].detach()*log_probability).mean()
        mean, _ = actor(self.demo_batch[f'o{index}'])
        prediction = actor.action_bias + actor.action_scale*mean.tanh()
        bc = model.config['fine_tune']['lambda_bc']*F.mse_loss(prediction, self.demo_batch[f'a{index}'])
        vectors, metrics = {}, {}
        for name, term in [('Q', q_term), ('entropy', entropy), ('BC', bc), ('total', loss+bc)]:
            vectors[name], metrics[name] = component_gradient(term, actor)
        pairwise = {f'{a}_vs_{b}': cosine(vectors[a], vectors[b]) for a,b in
                    [('Q','entropy'),('Q','BC'),('entropy','BC'),('total','BC')]}
        alignment = {}
        for name in ['Q','entropy']:
            c = pairwise[f'{name}_vs_BC']
            alignment[name] = 'undefined' if c is None else ('aligned_with_BC_descent' if c>.1 else
                              'opposed_to_BC_descent' if c<-.1 else 'nearly_orthogonal_to_BC_descent')
        self.actor_context[index] = dict(vectors=vectors, gradients=metrics,
            gradient_cosines=pairwise, gradient_alignment=alignment,
            critic_parameter_sha256=tree_hash(dict(q1=model.q1.state_dict(), q2=model.q2.state_dict())),
            alpha=float(model.alpha[index]), batch_sha256=tree_hash(batch), demo_batch_sha256=tree_hash(self.demo_batch))
        return loss, log_probability

    def _actor_step(self, index, original, args, kwargs):
        if not self.active:
            return original(*args, **kwargs)
        actor, optimizer = self.model.actors[index], self.model.actor_optimizers[index]
        names, parameters = zip(*actor.named_parameters())
        before = [p.detach().clone() for p in parameters]
        optimizer_before = cpu_tree(optimizer.state_dict())
        gradients = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]
        expected = [adam_expected_delta(p.detach(), gradient, optimizer.state.get(p, {}), optimizer.param_groups[0])
                    for p, gradient in zip(parameters, gradients)]
        actual_gradient = flatten(gradients)
        context = self.actor_context.pop(index)
        combined_error = float((actual_gradient-context['vectors']['total']).norm())
        teammate_clean = all(p.grad is None for p in self.model.actors[1-index].parameters())
        critic_clean = all(p.grad is None for critic in [self.model.q1,self.model.q2] for p in critic.parameters())
        result = original(*args, **kwargs)
        delta = [p.detach()-old for p,old in zip(parameters,before)]
        flat_delta = flatten(delta)
        adam_error = max(float((a-b).abs().max()) for a,b in zip(delta,expected))
        if adam_error > 2e-6:
            raise AssertionError(f'Adam displacement mismatch {adam_error}')
        row = dict(actor_update=self.actor_update, agent=index, environment_steps=self.environment_step,
            actor_sha_before=tree_hash(dict(zip(names,before))), actor_sha_after=tree_hash(dict(actor.named_parameters())),
            optimizer_sha_before=tree_hash(optimizer_before), optimizer_sha_after=tree_hash(optimizer.state_dict()),
            optimizer_lr=optimizer.param_groups[0]['lr'], optimizer_betas=optimizer.param_groups[0]['betas'],
            optimizer_state_empty_before=not optimizer_before['state'],
            combined_gradient_error_l2=combined_error, actual_gradient_l2=float(actual_gradient.norm()),
            adam_delta_l2=float(flat_delta.norm()), adam_expected_delta_max_error=adam_error,
            delta_per_parameter_l2={name:float(d.norm()) for name,d in zip(names,delta)},
            delta_cosine_vs_component_descent={name:cosine(flat_delta,-v) for name,v in context['vectors'].items()},
            teammate_grad_clean=teammate_clean, critic_grad_clean=critic_clean,
            **{key:value for key,value in context.items() if key!='vectors'},
            policy_after=policy_probe(self.model,self.reference,self.probes))
        local = self.output/f'actor_{self.actor_update:03d}_{index}.pt'
        torch.save(dict(parameters_before=cpu_tree(dict(zip(names,before))),
                        parameters_after=cpu_tree(dict(actor.named_parameters())),
                        optimizer_before=optimizer_before, optimizer_after=cpu_tree(optimizer.state_dict()),
                        actual_gradients=cpu_tree(dict(zip(names,gradients))),
                        component_gradients=context['vectors']),local)
        self.records.append(row)
        with (self.output/'actor_updates.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        return result

    def _update(self, original, model, batch, **kwargs):
        self.environment_step = self.main._causal_online_count if self.main is not None else 0
        enabled = kwargs.get('update_actor', True)
        self.active = enabled and model.actor_optimizer_updates[0] < 100
        self.actor_update = model.actor_optimizer_updates[0]+1
        self.demo_batch = kwargs.get('demo_batch')
        if self.environment_step in REPLAY_STEPS and not any(r['environment_steps']==self.environment_step for r in self.replay_records):
            sources = self.mixed_sources if self.environment_step else self.sample_log[-1]['sources']
            direct = self.mixed_direct if self.environment_step else len(sources)
            row = dict(environment_steps=self.environment_step, **self.replay_inventory(),
                       configured_demo_ratio=.75-.25*min(self.environment_step/5000,1),
                       offline_pretraining=self.environment_step==0, **composition(sources,direct),
                       critic_batch_sha256=tree_hash(batch), replay_sample_indices=deepcopy(self.sample_log))
            self.replay_records.append(row)
        if self.active:
            if self.model is None:
                self._initialize_model(model)
                model.save(self.output/'before_first_update.pt', dict(step=self.environment_step-1, checkpoint_timing='before Q and first actor update'))
            artifact = dict(batch=cpu_tree(batch), demo_batch=cpu_tree(self.demo_batch), rng_before=rng_state(),
                            environment_steps=self.environment_step, sample_indices=deepcopy(self.sample_log),
                            critic_batch_sha256=tree_hash(batch), demo_batch_sha256=tree_hash(self.demo_batch))
            torch.save(artifact,self.output/f'update_{self.actor_update:03d}.pt')
        result = original(model,batch,**kwargs)
        if self.active:
            self.executed_actor_updates += 1
            if self.actor_update in MILESTONES:
                model.save(self.output/f'after_{self.actor_update:03d}.pt',dict(step=self.environment_step))
            if self.actor_update in {1,2,5,10,20,50,100}:
                print(f'CAUSAL_TRACE actor_update={self.actor_update} env_step={self.environment_step}',flush=True)
        self.q_outputs.clear()
        self.sample_log.clear()
        self.active = False
        return result

    def __enter__(self):
        self.active = False
        self.stack = ExitStack()
        original_init, original_add, original_sample = ReplayBuffer.__init__, ReplayBuffer.add, ReplayBuffer.sample
        original_demos_init, original_prefill = DemonstrationReplay.__init__, DemonstrationReplay.prefill
        original_mixed = DemonstrationReplay.sample_mixed
        def init(replay,*args,**kwargs):
            original_init(replay,*args,**kwargs)
            replay._causal_sources = [None]*replay.capacity
            replay._causal_online_count = 0
            replay._causal_role = 'demo' if self.in_demos else 'main'
            if not self.in_demos:
                self.main = replay
        def add(replay,*args,**kwargs):
            position = replay.position
            if replay._causal_role=='demo' or self.in_prefill:
                label = self.labels[position]
            else:
                n = replay._causal_online_count
                label = ('online',n//self.manifest['env_config']['horizon'],n%self.manifest['env_config']['horizon'])
                replay._causal_online_count += 1
            original_add(replay,*args,**kwargs)
            replay._causal_sources[position] = label
        def sample(replay,count,device='cpu'):
            indices = peek_indices(replay,count)
            result = original_sample(replay,count,device)
            self.sample_log.append(dict(role=replay._causal_role, indices=indices.tolist(),
                sources=[replay._causal_sources[i] for i in indices], batch_sha256=tree_hash(result)))
            return result
        def demos_init(demos,*args,**kwargs):
            self.in_demos = True
            try:
                original_demos_init(demos,*args,**kwargs)
                self.demos = demos
            finally:
                self.in_demos = False
        def prefill(demos,replay):
            self.in_prefill = True
            try:
                return original_prefill(demos,replay)
            finally:
                self.in_prefill = False
        def mixed(demos,replay,size,ratio,device):
            start = len(self.sample_log)
            result = original_mixed(demos,replay,size,ratio,device)
            logs = self.sample_log[start:]
            self.mixed_sources = [source for entry in logs for source in entry['sources']]
            self.mixed_direct = int(round(size*ratio))
            return result
        original_loss,original_update = MASAC.actor_loss,MASAC.update
        self.stack.enter_context(patch.object(ReplayBuffer,'__init__',init))
        self.stack.enter_context(patch.object(ReplayBuffer,'add',add))
        self.stack.enter_context(patch.object(ReplayBuffer,'sample',sample))
        self.stack.enter_context(patch.object(DemonstrationReplay,'__init__',demos_init))
        self.stack.enter_context(patch.object(DemonstrationReplay,'prefill',prefill))
        self.stack.enter_context(patch.object(DemonstrationReplay,'sample_mixed',mixed))
        self.stack.enter_context(patch.object(MASAC,'actor_loss',lambda model,index,batch,**kwargs:self._actor_loss(original_loss,model,index,batch,**kwargs)))
        self.stack.enter_context(patch.object(MASAC,'update',lambda model,batch,**kwargs:self._update(original_update,model,batch,**kwargs)))
        return self

    def __exit__(self,*args):
        self.stack.__exit__(*args)
        for handle in self.snapshot_handles:
            handle.remove()
        (self.output/'replay_composition.json').write_text(json.dumps(self.replay_records,indent=2)+'\n')
