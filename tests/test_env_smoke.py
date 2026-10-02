"""Integration checks using the installed simulator and configured horizon."""

import numpy as np
import pytest

from rel301m.envs.contract import action_bounds, validate_action, validate_observations, validate_reward
from rel301m.envs.robosuite_factory import load_env_config
from rel301m.evaluation.rollout import run_episode


def test_construction_reset_and_state_only_settings(env):
    config = load_env_config()
    observations = env.reset()
    validate_observations(observations)
    assert len(env.robots) == len(config["robots"])
    assert env.env_configuration == config["env_configuration"]
    assert env.horizon == config["horizon"]
    assert env.control_freq == config["control_freq"]
    assert env.reward_shaping == config["reward_shaping"]
    assert env.reward_scale == config["reward_scale"]
    assert not env.has_renderer and not env.has_offscreen_renderer and not env.use_camera_obs
    assert env.use_object_obs and not env.ignore_done
    for key in ("robot0_proprio-state", "robot1_proprio-state", "object-state"):
        assert observations[key].ndim == 1 and observations[key].size > 0
    assert all(robot.composite_controller.name == config["controller"] for robot in env.robots)


def test_reset_and_several_random_steps(env):
    reference = env.reset()
    low, high = action_bounds(env)
    rng = np.random.default_rng(31)
    for _ in range(min(10, env.horizon - 1)):
        action = rng.uniform(low, high)
        validate_action(action, low, high)
        observations, reward, done, info = env.step(action)
        validate_observations(observations, reference)
        validate_reward(reward)
        assert not done
        assert isinstance(info, dict)


def test_full_horizon_episode(env):
    metrics = run_episode(env, np.random.default_rng(17))
    assert metrics["episode_length"] == env.horizon
    assert np.isfinite(metrics["episode_return"])
    assert env.done
    assert metrics["final_success"] == bool(env._check_success())
    if metrics["ever_success"]:
        assert 0 <= metrics["first_success_step"] <= env.horizon
    else:
        assert metrics["first_success_step"] is None
    assert not metrics["final_success"] or metrics["ever_success"]


def test_reject_invalid_actions_and_nonfinite_data(env):
    low, high = action_bounds(env)
    validate_action(low, low, high)
    validate_action(high, low, high)
    with pytest.raises(ValueError, match="dimensions"):
        validate_action(np.zeros(low.size + 1), low, high)
    with pytest.raises(ValueError, match="bounds"):
        validate_action(high + 1, low, high)
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_observations({"bad": np.array([np.nan])})
    with pytest.raises(ValueError, match="finite scalar"):
        validate_reward(np.inf)
