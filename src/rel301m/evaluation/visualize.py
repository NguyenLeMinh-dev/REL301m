"""Watch decentralized checkpoint rollouts in MuJoCo or export annotated MP4."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


class VideoEncoder:
    def __init__(self, path, width, height, fps):
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            raise RuntimeError('Video export requires ffmpeg on PATH')
        self.log = path.with_suffix('.ffmpeg.log').open('w', encoding='utf-8')
        self.process = subprocess.Popen(
            [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
             '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
             '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
             '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log,
        )

    def write(self, rgb):
        self.process.stdin.write(rgb.tobytes())

    def close(self):
        try:
            self.process.stdin.close()
            result = self.process.wait(timeout=60)
            if result:
                raise RuntimeError('ffmpeg failed; inspect the .ffmpeg.log beside the MP4')
        except BaseException:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            raise
        finally:
            self.log.close()


def make_camera(base_env, name):
    import mujoco

    model = base_env.sim.model._model
    if name != 'overview':
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
            raise ValueError(f'Unknown camera {name!r}; choose overview or one of {names}')
        return name
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = base_env.sim.data.body_xpos[base_env.pot_body_id]
    camera.lookat[2] += .12
    camera.distance, camera.azimuth, camera.elevation = 2.4, 135, -25
    return camera


def video_frame(renderer, base_env, camera, option, step, reward, total, ever, success, first, policy_label):
    import cv2
    import numpy as np

    renderer.update_scene(base_env.sim.data._data, camera=camera, scene_option=option)
    rgb = renderer.render()
    footer = np.full((96, rgb.shape[1], 3), (20, 25, 32), dtype=np.uint8)
    first_label = 'none' if first is None else f'{first} ({first / base_env.control_freq:.2f}s)'
    lines = [
        f'TwoArmLift | {policy_label} | step {step}/{base_env.horizon} | t={step/base_env.control_freq:.2f}s',
        f'reward={reward:.4f}   return={total:.2f}   first_success={first_label}',
        f'ever_success={bool(ever)}   current_success={bool(success)}   state observations only',
    ]
    for i, line in enumerate(lines):
        cv2.putText(footer, line, (16, 24 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, .55, (225, 231, 239), 1, cv2.LINE_AA)
    return np.concatenate((rgb, footer), axis=0)


def run_episode(env, model, args, output_dir, episode):
    from contextlib import ExitStack
    import cv2
    import mujoco
    import numpy as np

    from .metrics import episode_record

    observations = env.reset()
    base = env._env
    initial_hash = hashlib.sha256(env.critic_state.tobytes()).hexdigest()
    success = env.check_success()
    ever, first, total = success, 0 if success else None, 0.0
    raw_model, raw_data = base.sim.model._model, base.sim.data._data
    camera = make_camera(base, args.camera)
    option = mujoco.MjvOption()
    option.geomgroup[0], option.geomgroup[1] = 0, 1
    frames, step, done = 0, 0, False
    encoder = None
    video_path = output_dir / f'episode_{episode:03d}.mp4'
    with ExitStack() as stack:
        if args.mode == 'video':
            # Resolution affects only this visualization instance's framebuffer.
            raw_model.vis.global_.offwidth = max(raw_model.vis.global_.offwidth, args.width)
            raw_model.vis.global_.offheight = max(raw_model.vis.global_.offheight, args.height)
            renderer = stack.enter_context(mujoco.Renderer(raw_model, height=args.height, width=args.width))
            encoder = VideoEncoder(video_path, args.width, args.height + 96, env.control_freq)
            stack.callback(encoder.close)
            initial = video_frame(renderer, base, camera, option, 0, 0, total, ever, success, first, args.checkpoint.name)
            cv2.imwrite(str(output_dir / f'episode_{episode:03d}_start.png'), cv2.cvtColor(initial, cv2.COLOR_RGB2BGR))
        else:
            import mujoco.viewer

            viewer = stack.enter_context(mujoco.viewer.launch_passive(raw_model, raw_data,
                                             show_left_ui=False, show_right_ui=False))
            with viewer.lock():
                if isinstance(camera, str):
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
                else:
                    viewer.cam.type = camera.type
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
                viewer.opt.geomgroup[:] = option.geomgroup
            viewer.sync()
        for next_step in range(1, env.horizon + 1):
            tick = time.perf_counter()
            if args.mode == 'viewer' and not viewer.is_running():
                break
            actions = model.act([observations[a]['actor_obs'] for a in env.agent_ids], deterministic=True)
            if args.mode == 'viewer':
                with viewer.lock():
                    observations, reward, done, _ = env.step(*actions)
            else:
                observations, reward, done, _ = env.step(*actions)
            step = next_step
            if bool(done) != (step == env.horizon):
                raise RuntimeError('Unexpected termination: visualization must preserve the horizon contract')
            total += float(reward)
            success = env.check_success()
            ever |= success
            if success and first is None:
                first = step
            if args.mode == 'video':
                frame = video_frame(renderer, base, camera, option, step, reward, total, ever, success, first, args.checkpoint.name)
                encoder.write(frame)
                frames += 1
                if step in (env.horizon // 2, env.horizon):
                    label = 'middle' if step < env.horizon else 'end'
                    cv2.imwrite(str(output_dir / f'episode_{episode:03d}_{label}.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                viewer.sync()
                remaining = 1 / env.control_freq - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)
            if done:
                break
    if step == 0:
        return None
    row = episode_record(episode, total, ever, first, success, step, env.control_freq, initial_hash)
    row.update(partial=not bool(done), frames=frames,
               video=str(video_path) if args.mode == 'video' else None)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mode', choices=('video', 'viewer'), default='video')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--camera', default='overview', help='overview, agentview (close-up), or native camera name')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.episodes <= 0 or args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error('episodes must be positive; width/height must be positive even integers')
    # Set before importing MuJoCo/robosuite: headless EGL for MP4, GLFW for desktop.
    os.environ['MUJOCO_GL'] = 'egl' if args.mode == 'video' else 'glfw'
    import numpy as np
    import torch
    import yaml

    from rel301m.algorithms.masac import MASAC
    from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
    from rel301m.envs.robosuite_factory import make_two_arm_lift
    from .metrics import summarize_episodes

    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    model, metadata = MASAC.load(args.checkpoint, device, load_optimizers=False)
    if 'env_config' not in metadata:
        raise ValueError('Checkpoint lacks env_config')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_dir or args.checkpoint.parent / f'{args.mode}_seed{args.seed}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with tempfile.TemporaryDirectory(prefix='rel301m-visualize-') as temp:
        config = Path(temp) / 'env.yaml'
        config.write_text(yaml.safe_dump(metadata['env_config']), encoding='utf-8')
        env = MultiAgentWrapper(make_two_arm_lift(config, seed=args.seed))
        try:
            if tuple(env.observation_dims[a]['actor_obs'] for a in env.agent_ids) != model.obs_dims or env.critic_state_dim != model.state_dim:
                raise ValueError('Checkpoint observation dimensions mismatch')
            for i, agent in enumerate(env.agent_ids):
                if not all(np.array_equal(a, b) for a, b in zip(env.action_specs[agent], model.action_specs[i])):
                    raise ValueError('Checkpoint action bounds mismatch')
            for episode in range(args.episodes):
                row = run_episode(env, model, args, output, episode)
                if row is None:
                    break
                rows.append(row)
                print(json.dumps(row), flush=True)
                if row['partial']:
                    break
        except BaseException as error:
            (output / 'failure.json').write_text(json.dumps({'error': repr(error)}, indent=2)+'\n')
            raise
        finally:
            env.close()
    completed = [row for row in rows if not row['partial']]
    report = dict(mode=args.mode, seed=args.seed, camera=args.camera, device=device,
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_step=metadata.get('step'),
                  policy='deterministic', observation_mode='state_only',
                  episodes=rows, completed_metrics=summarize_episodes(completed) if completed else None,
                  output_dir=str(output.resolve()))
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'OUTPUT_DIR={output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
