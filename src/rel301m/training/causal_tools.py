"""Read-only measurements for the first actor updates; no learner settings changed."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import torch


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return deepcopy(value)


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(type(item).__name__).encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True).encode())
    visit(value)
    return digest.hexdigest()


def rng_state():
    return dict(cpu=torch.get_rng_state().clone(),
                cuda=[state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def peek_indices(replay, count):
    generator = np.random.default_rng(0)
    generator.bit_generator.state = deepcopy(replay.rng.bit_generator.state)
    return generator.integers(len(replay), size=count)


def composition(sources, direct_count=0):
    canonical = [tuple(source) for source in sources]
    demos = sum(source[0] == 'demo' for source in canonical)
    distribution = {}
    for kind, episode, _ in canonical:
        key = f'{kind}:{episode}'
        distribution[key] = distribution.get(key, 0) + 1
    return dict(batch_size=len(canonical), direct_demo_samples=direct_count,
                effective_demo_samples=demos, effective_online_samples=len(canonical)-demos,
                effective_demo_ratio=demos / len(canonical) if canonical else None,
                duplicate_rate=1-len(set(canonical))/len(canonical) if canonical else None,
                unique_episode_distribution=distribution)


def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return float(torch.dot(a.double(), b.double()) / denominator) if denominator else None


def flatten(tensors):
    return torch.cat([tensor.detach().reshape(-1).double().cpu() for tensor in tensors])


def component_gradient(loss, actor):
    parameters = list(actor.named_parameters())
    gradients = torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=True, allow_unused=True)
    gradients = [torch.zeros_like(p) if g is None else g for (_, p), g in zip(parameters, gradients)]
    layers = {name: float(gradient.norm()) for (name, _), gradient in zip(parameters, gradients)}
    last = f'net.{len(actor.net)-1}.'
    special = {}
    for label, prefix, rows in [('first_hidden', 'net.0.', None), ('final_mean', last, slice(0, actor.action_dim)),
                                 ('final_std', last, slice(actor.action_dim, None))]:
        selected = [gradient if rows is None else gradient[rows]
                    for (name, _), gradient in zip(parameters, gradients) if name.startswith(prefix)]
        special[label] = float(flatten(selected).norm())
    vector = flatten(gradients)
    return vector, dict(loss=float(loss.detach()), gradient_l2=float(vector.norm()), per_parameter_l2=layers,
                        **{f'{name}_gradient_l2': value for name, value in special.items()})


def adam_expected_delta(parameter, gradient, state, group):
    """Standard non-AMSGrad Adam, including previously accumulated moments."""
    if group.get('amsgrad') or group.get('weight_decay', 0) or group.get('maximize', False):
        raise ValueError('This diagnostic requires the baseline ordinary Adam settings')
    beta1, beta2 = group['betas']
    step = int(state.get('step', 0)) + 1
    m = state.get('exp_avg', torch.zeros_like(parameter)).to(parameter)
    v = state.get('exp_avg_sq', torch.zeros_like(parameter)).to(parameter)
    m = beta1*m + (1-beta1)*gradient
    v = beta2*v + (1-beta2)*gradient.square()
    delta = -group['lr'] * (m/(1-beta1**step)) / ((v/(1-beta2**step)).sqrt()+group['eps'])
    # Compare effective displacement after float32 addition/rounding, not ideal real arithmetic.
    return (parameter + delta) - parameter


@torch.no_grad()
def bounded_mean(actor, observations):
    return actor.action_bias + actor.action_scale * actor(observations)[0].tanh()


@torch.no_grad()
def policy_probe(model, reference, probes):
    results = {}
    for split, entries in probes.items():
        for i, (observations, expert) in enumerate(entries):
            actor = model.actors[i]
            actions = bounded_mean(actor, observations)
            bc = bounded_mean(reference[i], observations)
            _, log_std = actor(observations)
            normalized = (actions-actor.action_bias)/actor.action_scale
            results[f'{split}_bc_drift_{i}'] = float((actions-bc).square().mean())
            results[f'{split}_expert_mse_{i}'] = float((actions-expert).square().mean())
            if split == 'train':
                results[f'action_mean_{i}'] = actions.mean(0).cpu().tolist()
                results[f'action_across_states_std_{i}'] = actions.std(0).cpu().tolist()
                results[f'gaussian_std_mean_{i}'] = float(log_std.exp().mean())
                results[f'log_std_min_{i}'] = float(log_std.min())
                results[f'log_std_max_{i}'] = float(log_std.max())
                results[f'deterministic_saturation_{i}'] = float((normalized.abs()>.95).float().mean())
    return results


def discounted_returns(rewards, gamma):
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    result = np.empty_like(rewards)
    total = 0.0
    for index in range(len(rewards)-1, -1, -1):
        total = rewards[index] + gamma*total
        result[index] = total
    return result
