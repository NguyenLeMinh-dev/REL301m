"""CSV/TensorBoard outputs and exact software/config/source snapshots."""

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import subprocess
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from rel301m.envs.robosuite_factory import PROJECT_ROOT


def parameter_sha256(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}\n'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialization_metadata(model):
    modules = dict(model=model, actor_0=model.actors[0], actor_1=model.actors[1],
                   q1=model.q1, q2=model.q2, target_q1=model.target_q1, target_q2=model.target_q2)
    optimizers = dict(actor_0=model.actor_optimizers[0], actor_1=model.actor_optimizers[1],
                      critic=model.q_optimizer, alpha=model.alpha_optimizer)
    settings = {name: {key: optimizer.defaults[key] for key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad')}
                for name, optimizer in optimizers.items()}
    return dict(initialization='torch.nn.Linear default reset_parameters',
                initial_parameter_sha256={name: parameter_sha256(module) for name, module in modules.items()},
                optimizer_class='torch.optim.Adam', optimizer_settings=settings)


def run_metadata(run_dir, config, env_config, model, replay, device, *, extra_metadata=None):
    def git(*args):
        result = subprocess.run(['git', *args], cwd=PROJECT_ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    sha = git('rev-parse', '--verify', 'HEAD') or 'uncommitted (no HEAD yet)'
    status = git('status', '--short')
    (run_dir / 'git_status.txt').write_text((status or '')+'\n', encoding='utf-8')
    freeze = subprocess.run([sys.executable, '-I', '-m', 'pip', 'freeze'], text=True, capture_output=True, check=True)
    (run_dir / 'requirements.freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    (run_dir / 'pip_freeze_stderr.txt').write_text(freeze.stderr, encoding='utf-8')
    hashes = {}
    for folder in ('src/rel301m', 'configs', 'artifacts'):
        for path in sorted((PROJECT_ROOT / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.json'):
                hashes[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes['pyproject.toml'] = hashlib.sha256((PROJECT_ROOT / 'pyproject.toml').read_bytes()).hexdigest()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True)
    metadata = dict(config=config, env_config=env_config, git_sha=sha, git_status=status,
                    created_at=datetime.now(timezone.utc).isoformat(), seed=config['seed'],
                    python=sys.version, torch=str(torch.__version__),
                    packages={name: version(name) for name in ('robosuite', 'mujoco', 'numpy', 'PyYAML', 'torch', 'tensorboard', 'pytest')}, cuda_runtime=torch.version.cuda,
                    cuda_available=torch.cuda.is_available(), device=device,
                    gpu=gpu.stdout.strip() if gpu.returncode == 0 else None,
                    observation_dims=list(model.obs_dims), critic_state_dim=model.state_dim,
                    action_dims=list(model.action_dims), target_entropy=model.target_entropy.cpu().tolist(),
                    parameter_counts=model.parameter_counts(), replay_allocated_bytes=replay.allocated_bytes,
                    source_sha256=hashes, predictor_enabled=False,
                    communication_cost='N/A',
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    evaluation_protocol=dict(seed=config['eval_seed'],
                        periodic_episodes=config['eval_episodes'], final_episodes=config['final_eval_episodes'],
                        initialization_sequence='fixed_rng_state', policy='deterministic',
                    primary_metric='ever_success'), **initialization_metadata(model))
    metadata.update(extra_metadata or {})
    (run_dir / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (run_dir / 'env.yaml').write_text(yaml.safe_dump(env_config, sort_keys=False), encoding='utf-8')
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return metadata


class RunLogger:
    def __init__(self, run_dir, fields):
        self.tensorboard = SummaryWriter(str(run_dir / 'tensorboard'))
        self.streams = {}
        self.writers = {}
        for name, columns in fields.items():
            stream = (run_dir / f'{name}.csv').open('w', newline='', encoding='utf-8')
            self.streams[name] = stream
            self.writers[name] = csv.DictWriter(stream, columns)
            self.writers[name].writeheader()

    def flush(self):
        for stream in self.streams.values():
            stream.flush()
        self.tensorboard.flush()

    def close(self):
        for stream in self.streams.values():
            stream.close()
        self.tensorboard.close()
