"""Collect verified successful scripted TwoArmLift demonstrations without new dependencies."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from .demonstrations import file_sha256, validate_trajectory


def collect(output, episodes=40, max_attempts=100, seed=40000, config_path=None, video=True):
    from rel301m.envs.robosuite_factory import DEFAULT_CONFIG, make_two_arm_lift, load_env_config, PROJECT_ROOT
    from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
    from rel301m.envs.contract import inspect_actions
    from rel301m.evaluation.metrics import episode_record
    from .scripted_teacher import ScriptedLiftTeacher

    if episodes < 3 or max_attempts < episodes:
        raise ValueError('Need at least 3 episodes and max_attempts >= episodes')
    config_path = config_path or DEFAULT_CONFIG
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = MultiAgentWrapper(make_two_arm_lift(config_path, seed=seed))
    bounds = [env.action_specs[a] for a in env.agent_ids]
    manifest = dict(format_version=1, status='collecting', created_at=datetime.now(timezone.utc).isoformat(),
                    seed=seed, env_config=load_env_config(config_path),
                    observation_dims=[env.observation_dims[a]['actor_obs'] for a in env.agent_ids],
                    critic_state_dim=env.critic_state_dim, action_dims=[len(low) for low, _ in bounds],
                    action_specs=[(low.tolist(), high.tolist()) for low, high in bounds],
                    action_layout=inspect_actions(env._env), actor_observations='raw_phase2_float32',
                    teacher='privileged_scripted_reach_grasp_lift',
                    acceptance='native_final_success AND two-handle grasp during native success',
                    requested_episodes=episodes, successful_episodes=0, attempts=[], episodes=[],
                    contract_sha256={p.name: file_sha256(p) for p in (PROJECT_ROOT / 'artifacts').glob('*.json')},
                    teacher_source_sha256=file_sha256(Path(__file__).with_name('scripted_teacher.py')))
    manifest['packages'] = {name: version(name) for name in ('robosuite', 'mujoco', 'numpy', 'scipy')}
    git = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=PROJECT_ROOT, capture_output=True, text=True)
    manifest['git_sha'] = git.stdout.strip() if git.returncode == 0 else 'uncommitted (no HEAD yet)'
    freeze = subprocess.run([sys.executable, '-I', '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (output / 'requirements.freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    def save_manifest():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    save_manifest()
    try:
        for attempt in range(max_attempts):
            observations = env.reset()
            teacher = ScriptedLiftTeacher(env)
            base = env._env
            initial = env.check_success()
            initial_hash = hashlib.sha256(env.critic_state.tobytes()).hexdigest()
            ever, first, total = initial, 0 if initial else None, 0.
            traces = {key: [] for key in ('o0', 'o1', 's', 'a0', 'a1', 'r', 'next_o0', 'next_o1', 'next_s', 'done', 'timeout')}
            success_trace, grasp_trace, phase_trace = [initial], [teacher.grasps()], []
            qpos, qvel = [base.sim.data.qpos.copy()], [base.sim.data.qvel.copy()]
            video_path = None
            with ExitStack() as stack:
                if video and not manifest['episodes']:
                    import mujoco
                    from rel301m.evaluation.visualize import VideoEncoder, make_camera, video_frame
                    renderer = stack.enter_context(mujoco.Renderer(base.sim.model._model, height=480, width=640))
                    camera, option = make_camera(base, 'overview'), mujoco.MjvOption()
                    option.geomgroup[0], option.geomgroup[1] = 0, 1
                    video_path = output / f'preview_attempt_{attempt:03d}.mp4'
                    encoder = VideoEncoder(video_path, 640, 576, env.control_freq)
                    stack.callback(encoder.close)
                for step in range(1, env.horizon + 1):
                    actors = [observations[a]['actor_obs'] for a in env.agent_ids]
                    state = env.critic_state
                    actions = teacher.act()
                    phase_trace.append(teacher.phases.index(teacher.phase))
                    next_obs, reward, done, _ = env.step(*actions)
                    if bool(done) != (step == env.horizon):
                        raise RuntimeError('Demonstration termination differs from native horizon')
                    values = dict(o0=actors[0], o1=actors[1], s=state, a0=actions[0], a1=actions[1], r=[reward],
                                  next_o0=next_obs['agent_0']['actor_obs'], next_o1=next_obs['agent_1']['actor_obs'],
                                  next_s=env.critic_state, done=[done], timeout=[done])
                    for key, value in values.items():
                        traces[key].append(np.asarray(value, dtype=np.float32).copy())
                    total += float(reward)
                    success = env.check_success()
                    ever |= success
                    if success and first is None:
                        first = step
                    success_trace.append(success)
                    grasp_trace.append(teacher.grasps())
                    qpos.append(base.sim.data.qpos.copy())
                    qvel.append(base.sim.data.qvel.copy())
                    observations = next_obs
                    if video_path:
                        encoder.write(video_frame(renderer, base, camera, option, step, reward, total, ever, success,
                                                  first, f'scripted / {teacher.phase}'))
            arrays = {key: np.stack(value) for key, value in traces.items()}
            arrays.update(success=np.asarray(success_trace, dtype=bool), grasps=np.asarray(grasp_trace, dtype=bool),
                          phase=np.asarray(phase_trace, dtype=np.int8), qpos=np.stack(qpos), qvel=np.stack(qvel))
            accepted = bool(success_trace[-1] and np.any(arrays['success'] & arrays['grasps'].all(axis=1)))
            record = episode_record(attempt, total, ever, first, success_trace[-1], env.horizon, env.control_freq, initial_hash)
            record.update(accepted=accepted, preview_video=str(video_path) if video_path else None)
            manifest['attempts'].append(record)
            if accepted:
                validate_trajectory(arrays, manifest)
                filename = f'episode_{len(manifest["episodes"]):03d}.npz'
                np.savez_compressed(output / filename, **arrays)
                manifest['episodes'].append(dict(episode=len(manifest['episodes']), attempt=attempt,
                                                file=filename, sha256=file_sha256(output / filename)))
                manifest['successful_episodes'] = len(manifest['episodes'])
            save_manifest()
            print(json.dumps(dict(attempt=attempt, accepted=accepted, collected=manifest['successful_episodes'],
                                  first_success_step=first, final_success=bool(success_trace[-1]))), flush=True)
            if manifest['successful_episodes'] == episodes:
                manifest['status'] = 'completed'
                save_manifest()
                return manifest
        raise RuntimeError(f'Only {manifest["successful_episodes"]}/{episodes} demos within {max_attempts} attempts')
    except BaseException as error:
        manifest['status'] = 'failed'
        save_manifest()
        (output / 'failure.json').write_text(json.dumps(dict(error=repr(error), traceback=traceback.format_exc()), indent=2)+'\n')
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/demonstrations/two_arm_lift_scripted'))
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--max-attempts', type=int, default=100)
    parser.add_argument('--seed', type=int, default=40000)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    os.environ['MUJOCO_GL'] = 'egl'
    collect(args.output_dir, args.episodes, args.max_attempts, args.seed, args.config, not args.no_video)


if __name__ == '__main__':
    main()
