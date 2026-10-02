"""Use the verbatim assisted environment; record actions below assistance."""
from runtime_guard import enforce_phase3_python
enforce_phase3_python()

import gymnasium as gym
import numpy as np

from assisted_env import DEFAULT_ASSIST_DISTANCE, DEFAULT_HORIZON, OBS_KEYS
from assisted_env import OpenUntilCloseWrapper, make_lift_env as _make


class ExecutedActionRecorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_executed_action = None

    def reset(self, **kwargs):
        self.last_executed_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        # This wrapper sits BELOW OpenUntilCloseWrapper. Record the argument
        # actually sent to GymWrapper/robosuite, not a second guessed gate.
        executed = np.asarray(action)
        if executed.dtype != np.float32 or executed.shape != self.action_space.shape:
            raise ValueError("Executed action must match the live float32 action space")
        if not np.isfinite(executed).all() or not self.action_space.contains(executed):
            raise ValueError("Non-finite/out-of-bounds executed action")
        self.last_executed_action = executed.copy()
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["executed_action"] = self.last_executed_action.copy()
        return obs, reward, terminated, truncated, info


class UnassistedDiagnostics(gym.Wrapper):
    """Measure native state while forwarding the policy action unchanged."""
    def step(self, action):
        raw = self.env.unwrapped
        distance = float(raw._gripper_to_target(gripper=raw.robots[0].gripper, target=raw.cube.root_body,
                                               target_type="body", return_distance=True))
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.update(assist_active=False, distance_to_cube_m=distance,
                    policy_gripper_action=float(action[6]), executed_gripper_action=float(info["executed_action"][6]),
                    grasping=bool(raw._check_grasp(gripper=raw.robots[0].gripper, object_geoms=raw.cube)),
                    cube_height_m=float(raw.sim.data.body_xpos[raw.cube_body_id][2]))
        return obs, reward, terminated, truncated, info


def make_lift_env(seed, *, horizon=DEFAULT_HORIZON, assist_distance=DEFAULT_ASSIST_DISTANCE, mode="assisted"):
    if mode not in ("assisted", "unassisted"):
        raise ValueError(f"Unknown environment mode: {mode}")
    env = _make(seed, horizon=horizon, assist_distance=assist_distance)
    parent, cursor = None, env
    while hasattr(cursor, "env") and not isinstance(cursor, OpenUntilCloseWrapper):
        parent, cursor = cursor, cursor.env
    if not isinstance(cursor, OpenUntilCloseWrapper):
        env.close()
        raise RuntimeError("Copied source does not contain the expected assistance wrapper")
    if cursor.gripper_slice != slice(6, 7):
        env.close()
        raise RuntimeError(f"Live gripper action layout mismatch: {cursor.gripper_slice}")
    cursor.env = ExecutedActionRecorder(cursor.env)
    if mode == "unassisted":
        # Remove only OpenUntilCloseWrapper; all native environment settings and
        # outer success/TimeLimit/float32 wrappers are preserved.
        parent.env = UnassistedDiagnostics(cursor.env)
    try:
        environment_contract(env, assist_distance=assist_distance, horizon=horizon, mode=mode)
    except Exception:
        env.close()
        raise
    return env


def environment_contract(env, *, assist_distance=DEFAULT_ASSIST_DISTANCE, horizon=DEFAULT_HORIZON, mode="assisted"):
    raw = env.unwrapped
    info = raw.robots[0].composite_controller.get_action_info_dict()
    arm = raw.robots[0].part_controllers["right"]
    if raw.robots[0].composite_controller.name != "BASIC" or arm.name != "OSC_POSE":
        raise RuntimeError("Expected BASIC/OSC_POSE from live controllers")
    if env.action_space.shape != (7,) or env.action_space.dtype != np.float32:
        raise RuntimeError("Expected live 7-dimensional float32 action space")
    if not np.array_equal(env.action_space.low, -np.ones(7)) or not np.array_equal(env.action_space.high, np.ones(7)):
        raise RuntimeError("Expected live action bounds [-1,1]")
    if env.observation_space.dtype != np.float32 or len(env.observation_space.shape) != 1:
        raise RuntimeError("Expected flat float32 observation space")
    return {"env": "Lift", "robot": "Panda", "controller": "BASIC", "arm_controller": arm.name,
            "observation_keys": list(OBS_KEYS), "obs_dim": int(env.observation_space.shape[0]),
            "action_dim": int(env.action_space.shape[0]), "action_layout": {key: list(value) for key, value in info.items()},
            "action_low": env.action_space.low.tolist(), "action_high": env.action_space.high.tolist(),
            "horizon": horizon, "control_freq": int(raw.control_freq), "reward_shaping": bool(raw.reward_shaping),
            "reward_scale": float(raw.reward_scale), "use_camera_obs": bool(raw.use_camera_obs),
            "use_object_obs": bool(raw.use_object_obs), "assist_distance": float(assist_distance),
            "evaluation_mode": mode, "gripper_assistance": mode == "assisted", "termination": "TimeLimit truncation, native success does not terminate"}
