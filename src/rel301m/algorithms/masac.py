"""Shared-reward SAC with two independent actors and twin centralized critics."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .networks import GaussianActor, CentralizedQ
from rel301m.utils.diagnostics import require_finite, td_error_metrics, action_metrics


class MASAC(nn.Module):
    def __init__(self, obs_dims, state_dim, action_specs, config, device='cpu'):
        super().__init__()
        if len(obs_dims) != 2 or len(action_specs) != 2:
            raise ValueError("MASAC requires exactly two agents")
        self.config = deepcopy(config)
        self.obs_dims, self.state_dim = tuple(obs_dims), int(state_dim)
        self.action_specs = [(np.asarray(low).copy(), np.asarray(high).copy()) for low, high in action_specs]
        self.action_dims = tuple(len(low) for low, _ in action_specs)
        hidden = tuple(config['hidden_dims'])
        self.actors = nn.ModuleList(GaussianActor(size, *bounds, hidden) for size, bounds in zip(obs_dims, action_specs))
        self.q1 = CentralizedQ(state_dim, sum(self.action_dims), hidden)
        self.q2 = CentralizedQ(state_dim, sum(self.action_dims), hidden)
        self.target_q1, self.target_q2 = deepcopy(self.q1), deepcopy(self.q2)
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.full((2,), np.log(config.get('initial_alpha', 0.2)), dtype=torch.float32))
        self.to(device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=config['actor_lr']) for actor in self.actors]
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=config['critic_lr'])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['alpha_lr'])
        self.target_entropy = torch.tensor([-size for size in self.action_dims], device=device, dtype=torch.float32)
        self.updates = 0
        self.critic_optimizer_updates = 0
        self.actor_optimizer_updates = [0, 0]
        self.alpha_optimizer_updates = 0

    @torch.no_grad()
    def initialize_log_std(self, value):
        if not np.isfinite(value) or not -20 <= value <= 2:
            raise ValueError("Initial log_std must be finite and within [-20, 2]")
        for actor in self.actors:
            actor.net[-1].weight[actor.action_dim:].zero_()
            actor.net[-1].bias[actor.action_dim:].fill_(value)

    def configure_log_std(self):
        settings = self.config.get("fine_tune", {})
        fixed = settings.get("actor_init_log_std") if self.updates < settings.get("freeze_log_std_updates", 0) else None
        for actor in self.actors:
            actor.fixed_log_std = fixed

    @property
    def device(self):
        return self.log_alpha.device

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, observations, deterministic=False):
        if len(observations) != 2:
            raise ValueError("Exactly two actor observations are required")
        self.configure_log_std()
        # Deliberately accept only each actor's own observation vector.
        return tuple(actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=self.device), deterministic)[0].cpu().numpy()
                     for actor, obs in zip(self.actors, observations))

    @torch.no_grad()
    def critic_target(self, batch):
        samples = [actor.sample(batch[f'next_o{i}']) for i, actor in enumerate(self.actors)]
        action = torch.cat([sample[0] for sample in samples], dim=-1)
        entropy = sum(self.alpha[i] * sample[1] for i, sample in enumerate(samples))
        next_q = torch.minimum(self.target_q1(batch['next_s'], action), self.target_q2(batch['next_s'], action))
        terminal = batch['done']
        if self.config.get('bootstrap_time_limits', True):
            terminal = terminal * (1 - batch['timeout'])
        return batch['r'] + self.config['gamma'] * (1 - terminal) * (next_q - entropy)

    def actor_loss(self, index, batch, *, diagnostics=None):
        own_action, own_log_prob = self.actors[index].sample(batch[f'o{index}'])
        other = 1 - index
        # The teammate is a fixed policy in this agent's update, not a gradient path.
        with torch.no_grad():
            other_action, _ = self.actors[other].sample(batch[f'o{other}'])
        actions = [None, None]
        actions[index], actions[other] = own_action, other_action
        joint = torch.cat(actions, dim=-1)
        q = torch.minimum(self.q1(batch['s'], joint), self.q2(batch['s'], joint))
        require_finite('actor Q/log probability', q, own_log_prob)
        if diagnostics is not None:
            diagnostics.update({f'{key}_{index}': value for key, value in
                                action_metrics(self.actors[index], batch[f'o{index}'], own_action).items()})
        return (self.alpha[index].detach() * own_log_prob - q).mean(), own_log_prob

    @torch.no_grad()
    def soft_update(self):
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.lerp_(parameter, self.config['tau'])

    def update(self, batch, *, collect_diagnostics=True, demo_batch=None, update_actor=True):
        self.configure_log_std()
        require_finite('replay batch', *batch.values())
        coefficient = self.config.get('fine_tune', {}).get('lambda_bc', 0.0)
        if update_actor and coefficient > 0 and demo_batch is None:
            raise ValueError('BC auxiliary loss requires a demonstration batch')
        if demo_batch is not None:
            require_finite('demo batch', *demo_batch.values())
        if any(p.grad is not None for net in (self.target_q1, self.target_q2) for p in net.parameters()):
            raise RuntimeError('Target networks must not receive gradients')
        target = self.critic_target(batch)
        joint = torch.cat((batch['a0'], batch['a1']), dim=-1)
        q1, q2 = self.q1(batch['s'], joint), self.q2(batch['s'], joint)
        require_finite('critic Q/target', q1, q2, target)
        diagnostics = td_error_metrics(q1, q2, target) if collect_diagnostics else {}
        loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        loss_q.backward()
        q_grad = self._gradient_norm(list(self.q1.parameters()) + list(self.q2.parameters()))
        require_finite('critic loss/gradient', loss_q.item(), q_grad)
        self.q_optimizer.step()
        self.critic_optimizer_updates += 1
        self.q_optimizer.zero_grad(set_to_none=True)
        if update_actor:
            actor_losses, log_probs, actor_grads = [], [], []
            bc_losses = []
            self.q1.requires_grad_(False)
            self.q2.requires_grad_(False)
            try:
                for i, optimizer in enumerate(self.actor_optimizers):
                    loss, log_prob = self.actor_loss(i, batch, diagnostics=diagnostics if collect_diagnostics else None)
                    bc_loss = torch.zeros((), device=self.device)
                    if coefficient > 0:
                        mean, _ = self.actors[i](demo_batch[f'o{i}'])
                        predicted = self.actors[i].action_bias + self.actors[i].action_scale * mean.tanh()
                        bc_loss = F.mse_loss(predicted, demo_batch[f'a{i}'])
                        loss = loss + coefficient * bc_loss
                    bc_losses.append(bc_loss.detach())
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_grads.append(self._gradient_norm(self.actors[i].parameters()))
                    require_finite('actor loss/gradient', loss.item(), actor_grads[-1])
                    optimizer.step()
                    self.actor_optimizer_updates[i] += 1
                    optimizer.zero_grad(set_to_none=True)
                    actor_losses.append(loss.detach())
                    log_probs.append(log_prob.detach())
            finally:
                self.q1.requires_grad_(True)
                self.q2.requires_grad_(True)
            alpha_loss = sum(-(self.log_alpha[i] * (log_probs[i] + self.target_entropy[i])).mean() for i in range(2))
            if self.config.get('fine_tune', {}).get('alpha_mode', 'auto') == 'auto':
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha_optimizer_updates += 1
            else:
                alpha_loss = torch.zeros((), device=self.device)
        else:
            actor_losses = [torch.zeros((), device=self.device)] * 2
            actor_grads = [0.0, 0.0]
            with torch.no_grad():
                samples = [actor.sample(batch[f'o{i}']) for i, actor in enumerate(self.actors)]
                log_probs = [sample[1] for sample in samples]
                if collect_diagnostics:
                    for i, sample in enumerate(samples):
                        diagnostics.update({f'{key}_{i}': value for key, value in
                            action_metrics(self.actors[i], batch[f'o{i}'], sample[0]).items()})
            bc_losses = [torch.zeros((), device=self.device)] * 2
            alpha_loss = torch.zeros((), device=self.device)
        self.soft_update()
        require_finite('model parameters', *self.parameters())
        self.updates += 1
        metrics = {
            'q_loss': loss_q.item(), 'q1_mean': q1.mean().item(), 'q2_mean': q2.mean().item(),
            'target_mean': target.mean().item(), 'target_q_mean': target.mean().item(),
            'alpha_loss': alpha_loss.item(), 'q_grad_norm': q_grad, **diagnostics,
        }
        for i in range(2):
            metrics.update({f'actor_{i}_loss': actor_losses[i].item(), f'alpha_{i}': self.alpha[i].item(),
                            f'actor_{i}_grad_norm': actor_grads[i],
                            f'log_pi_mean_{i}': log_probs[i].mean().item(),
                            f'entropy_{i}': -log_probs[i].mean().item()})
        if 'fine_tune' in self.config:
            metrics.update(actor_updated=int(update_actor), bc_loss_0=bc_losses[0].item(), bc_loss_1=bc_losses[1].item())
        metrics.update(self.optimizer_counts())
        require_finite('MASAC metrics', *metrics.values())
        if min(metrics['alpha_0'], metrics['alpha_1']) <= 0:
            raise FloatingPointError('Alpha underflowed to zero')
        return metrics

    def optimizer_counts(self):
        return dict(critic_optimizer_updates=self.critic_optimizer_updates,
                    actor_optimizer_updates_0=self.actor_optimizer_updates[0],
                    actor_optimizer_updates_1=self.actor_optimizer_updates[1],
                    actor_optimizer_steps_total=sum(self.actor_optimizer_updates),
                    alpha_optimizer_updates=self.alpha_optimizer_updates)

    @staticmethod
    def _gradient_norm(parameters):
        squared = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
        return torch.stack(squared).sum().sqrt().item() if squared else 0.0

    def parameter_counts(self):
        return {name: sum(p.numel() for p in module.parameters()) for name, module in
                [('actor_0', self.actors[0]), ('actor_1', self.actors[1]),
                 ('q1', self.q1), ('q2', self.q2), ('target_q1', self.target_q1), ('target_q2', self.target_q2)]}

    def save(self, path, metadata=None):
        from rel301m.utils.checkpoint import save_checkpoint
        save_checkpoint(self, path, metadata)

    @classmethod
    def load(cls, path, device='cpu', load_optimizers=True):
        from rel301m.utils.checkpoint import load_checkpoint
        return load_checkpoint(path, device, load_optimizers)
