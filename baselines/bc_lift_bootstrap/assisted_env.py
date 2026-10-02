"""Stage-1 assisted robosuite Lift environment.

Standalone external sanity baseline. It imports no REL301m code.
The only assistance is gripper gating: while the gripper is farther than
``assist_distance`` from the cube, the executed gripper command is forced OPEN.
Once inside the threshold, the SAC policy controls the gripper normally.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import warnings
from pathlib import Path

from runtime_guard import enforce_phase3_python

enforce_phase3_python()
warnings.filterwarnings("ignore")

_import_stdout = io.StringIO()
_import_stderr = io.StringIO()
with contextlib.redirect_stdout(_import_stdout), contextlib.redirect_stderr(_import_stderr):
    import robosuite as suite
    from robosuite.wrappers import GymWrapper

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)
for _handler in ROBOSUITE_DEFAULT_LOGGER.handlers:
    _handler.setLevel(logging.ERROR)
    if isinstance(_handler, logging.StreamHandler):
        try:
            _handler.setStream(sys.stderr)
        except (AttributeError, ValueError):
            _handler.stream = sys.stderr

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

ENV_NAME = "Lift"
ROBOT = "Panda"
DEFAULT_HORIZON = 500
DEFAULT_CONTROL_FREQ = 20
DEFAULT_ASSIST_DISTANCE = 0.05  # 5 cm
OBS_KEYS = ["object-state", "robot0_proprio-state"]



class Float32ObservationWrapper(gym.ObservationWrapper):
    """Cast robosuite state observations to float32 for Gymnasium/SB3."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(f"Expected Box observation space, got {type(space).__name__}")
        self.observation_space = gym.spaces.Box(
            low=np.asarray(space.low, dtype=np.float32),
            high=np.asarray(space.high, dtype=np.float32),
            shape=space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32)


class OpenUntilCloseWrapper(gym.Wrapper):
    """Force the gripper open while the EEF is farther than a threshold.

    Arm actions are NEVER modified. The gripper action is only overridden while
    distance(gripper, cube) > assist_distance. Panda convention in robosuite is
    -1=open and +1=closed.
    """

    def __init__(self, env: gym.Env, assist_distance: float = DEFAULT_ASSIST_DISTANCE):
        super().__init__(env)
        self.assist_distance = float(assist_distance)
        if not (0.0 < self.assist_distance < 0.20):
            raise ValueError("assist_distance must be in (0, 0.20) meters")

        raw = self.env.unwrapped
        if len(raw.robots) != 1:
            raise RuntimeError("Stage 1 requires exactly one Panda robot")
        info = raw.robots[0].composite_controller.get_action_info_dict()
        gripper_parts = [(name, indices) for name, indices in info.items() if name != "Action Dimension" and "gripper" in name]
        if len(gripper_parts) != 1:
            raise RuntimeError(f"Expected exactly one gripper action slice, got {gripper_parts}")
        name, indices = gripper_parts[0]
        start, stop = map(int, indices)
        if stop - start != 1:
            raise RuntimeError(f"Expected 1-D Panda gripper action, got {name}: {indices}")
        self.gripper_slice = slice(start, stop)
        self.gripper_part_name = name

    def _distance_to_cube(self) -> float:
        raw = self.env.unwrapped
        return float(
            raw._gripper_to_target(
                gripper=raw.robots[0].gripper,
                target=raw.cube.root_body,
                target_type="body",
                return_distance=True,
            )
        )

    def _is_grasping(self) -> bool:
        raw = self.env.unwrapped
        return bool(raw._check_grasp(gripper=raw.robots[0].gripper, object_geoms=raw.cube))

    def _cube_height(self) -> float:
        raw = self.env.unwrapped
        return float(raw.sim.data.body_xpos[raw.cube_body_id][2])

    def step(self, action):
        action = np.asarray(action, dtype=self.action_space.dtype)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Action shape {action.shape} != {self.action_space.shape}")
        executed = action.copy()
        distance = self._distance_to_cube()
        assist_active = distance > self.assist_distance
        policy_gripper = float(action[self.gripper_slice][0])
        if assist_active:
            executed[self.gripper_slice] = -1.0

        obs, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info)
        info.update(
            assist_active=bool(assist_active),
            distance_to_cube_m=float(distance),
            policy_gripper_action=policy_gripper,
            executed_gripper_action=float(executed[self.gripper_slice][0]),
            grasping=self._is_grasping(),
            cube_height_m=self._cube_height(),
        )
        return obs, reward, terminated, truncated, info


class EverSuccessWrapper(gym.Wrapper):
    """Add current and ever-success flags without changing the task."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.ever_success = False

    def _current_success(self) -> bool:
        raw = self.env.unwrapped
        return bool(raw._check_success())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.ever_success = self._current_success()
        info = dict(info)
        info["current_success"] = self.ever_success
        info["is_success"] = self.ever_success
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_success = self._current_success()
        self.ever_success = bool(self.ever_success or current_success)
        info = dict(info)
        info["current_success"] = current_success
        info["is_success"] = self.ever_success
        return obs, reward, terminated, truncated, info


def make_lift_env(
    seed: int,
    *,
    horizon: int = DEFAULT_HORIZON,
    assist_distance: float = DEFAULT_ASSIST_DISTANCE,
    monitor_path: str | Path | None = None,
) -> gym.Env:
    if suite.__version__ != "1.5.2":
        raise RuntimeError(f"Expected robosuite==1.5.2, found {suite.__version__}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    controller = suite.load_composite_controller_config(controller="BASIC")
    raw_env = suite.make(
        env_name=ENV_NAME,
        robots=ROBOT,
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        reward_scale=1.0,
        control_freq=DEFAULT_CONTROL_FREQ,
        horizon=horizon,
        ignore_done=True,
        hard_reset=False,
        seed=int(seed),
    )

    env = GymWrapper(raw_env, keys=OBS_KEYS, flatten_obs=True)
    env = Float32ObservationWrapper(env)
    env = OpenUntilCloseWrapper(env, assist_distance=assist_distance)
    env = EverSuccessWrapper(env)
    env = TimeLimit(env, max_episode_steps=horizon)

    if monitor_path is not None:
        monitor_path = Path(monitor_path)
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=("is_success", "current_success"),
        )
    return env


def env_summary(env: gym.Env) -> dict:
    obs, _ = env.reset()
    assist = None
    cursor = env
    while hasattr(cursor, "env"):
        if isinstance(cursor, OpenUntilCloseWrapper):
            assist = cursor
            break
        cursor = cursor.env
    return {
        "stage": "stage1_assisted_lift",
        "env": ENV_NAME,
        "robot": ROBOT,
        "observation_shape": list(np.asarray(obs).shape),
        "action_shape": list(env.action_space.shape),
        "action_low": np.asarray(env.action_space.low).tolist(),
        "action_high": np.asarray(env.action_space.high).tolist(),
        "horizon": int(getattr(env, "_max_episode_steps", DEFAULT_HORIZON)),
        "control_freq": DEFAULT_CONTROL_FREQ,
        "observation_keys": list(OBS_KEYS),
        "reward_shaping": True,
        "assist_distance_m": float(assist.assist_distance if assist is not None else DEFAULT_ASSIST_DISTANCE),
        "assist_rule": "force gripper OPEN only while distance_to_cube > assist_distance; arm actions unchanged",
        "timeout_semantics": "Gymnasium TimeLimit truncation",
    }
