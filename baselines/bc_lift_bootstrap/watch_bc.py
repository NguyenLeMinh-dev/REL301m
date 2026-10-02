"""MuJoCo passive viewer on the same assisted BC environment."""
from common import device, model_path, verify_stack
from env import make_lift_env

import argparse
import time
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--headless", action="store_true", help="Check diagnostics without opening a GUI")
    args = p.parse_args()
    if args.episodes <= 0 or args.print_every <= 0:
        raise ValueError("episodes and print-every must be positive")
    verify_stack()
    model = SAC.load(model_path(args.model), device=device(args.device))
    metadata = getattr(model, "bc_bootstrap_metadata", None)
    if metadata is None:
        raise ValueError("Not a BC bootstrap checkpoint")
    for episode in range(args.episodes):
        env = make_lift_env(args.seed+episode, assist_distance=metadata["environment"]["assist_distance"])
        viewer = None
        try:
            obs, _ = env.reset()
            raw = env.unwrapped
            if not args.headless:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(raw.sim.model._model, raw.sim.data._data)
            total, previous = 0., None
            for step in range(1, 501):
                start = time.monotonic()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
                total += float(reward)
                state = (info["grasping"], info["current_success"], info["assist_active"])
                if step == 1 or step % args.print_every == 0 or state != previous:
                    print(f"step={step} distance_to_cube={info['distance_to_cube_m']:.4f} policy_gripper_action={info['policy_gripper_action']:+.3f} executed_gripper_action={info['executed_gripper_action']:+.3f} grasp={int(info['grasping'])} cube_height={info['cube_height_m']:.4f} success={int(info['current_success'])}", flush=True)
                previous = state
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    viewer.sync()
                    time.sleep(max(0., 1/20-(time.monotonic()-start)))
                if terminated or truncated:
                    break
            print(f"episode={episode+1} return={total:.6f} ever_success={info['is_success']} final_success={info['current_success']}")
        finally:
            if viewer is not None:
                viewer.close()
            env.close()


if __name__ == "__main__":
    main()
