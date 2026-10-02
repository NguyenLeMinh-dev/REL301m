"""Deterministic rollouts with native success and actual executed actions."""
from common import checked_action
from env import make_lift_env
import numpy as np


def run_episode(model, seed, *, assist_distance, horizon=500, observer=None):
    env = make_lift_env(seed, horizon=horizon, assist_distance=assist_distance)
    try:
        obs, reset_info = env.reset()
        if obs.dtype != np.float32 or not np.isfinite(obs).all():
            raise ValueError("Invalid reset observation")
        trajectory = {k: [] for k in ("observations", "actions", "executed_actions", "policy_actions", "next_observations",
                                     "rewards", "steps", "dones", "terminated", "truncated", "current_success",
                                     "distance_to_cube", "cube_height", "grasp", "assist_active")}
        first = 0 if reset_info["current_success"] else None
        ever_grasp, total, info = False, 0., reset_info
        for step in range(horizon):
            policy, _ = model.predict(obs, deterministic=True)
            policy = checked_action(policy, env.action_space)
            next_obs, reward, terminated, truncated, info = env.step(policy)
            if next_obs.dtype != np.float32 or not np.isfinite(next_obs).all() or not np.isfinite(reward):
                raise ValueError("Non-finite/wrong-dtype transition")
            if terminated or (truncated != (step == horizon-1)):
                raise RuntimeError("Native/time-limit episode contract changed")
            executed = checked_action(info["executed_action"], env.action_space)
            if info["current_success"] and first is None:
                first = step+1
            ever_grasp |= bool(info["grasping"])
            row = {"observations": obs.copy(), "actions": executed.copy(), "executed_actions": executed.copy(),
                   "policy_actions": policy.copy(), "next_observations": next_obs.copy(), "rewards": float(reward),
                   "steps": step, "dones": bool(terminated or truncated), "terminated": bool(terminated), "truncated": bool(truncated),
                   "current_success": bool(info["current_success"]), "distance_to_cube": info["distance_to_cube_m"],
                   "cube_height": info["cube_height_m"], "grasp": info["grasping"], "assist_active": info["assist_active"]}
            for key, value in row.items():
                trajectory[key].append(value)
            if observer is not None:
                observer(step+1, row, env)
            total += float(reward)
            obs = next_obs
        result = {"seed": int(seed), "episode_return": total, "ever_success": bool(info["is_success"]),
                  "final_success": bool(info["current_success"]), "first_success_step": first,
                  "episode_length": horizon, "ever_grasp": ever_grasp,
                  "premature_close_count": sum(bool(a[6] > 0 and active) for a, active in zip(trajectory["policy_actions"], trajectory["assist_active"]))}
        return trajectory, result
    finally:
        env.close()
