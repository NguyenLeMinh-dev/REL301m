"""Check measured contracts and task semantics without guessing dimensions."""

import json

import numpy as np
import pytest

from rel301m.envs.contract import inspect_actions, inspect_observations, source_evidence
from rel301m.envs.robosuite_factory import PROJECT_ROOT
from rel301m.evaluation.rollout import run_episode


def saved_spec(filename):
    path = PROJECT_ROOT / "artifacts" / filename
    if not path.is_file():
        pytest.fail(f"Missing {path}; run python -I scripts/inspect_env.py first")
    return json.loads(path.read_text())


def test_observation_snapshot_and_observable_status(env):
    observations = env.reset()
    spec = saved_spec("observation_spec.json")
    assert inspect_observations(env, observations) == spec
    returned_sensors = set(observations).intersection(env.observation_names)
    assert returned_sensors == env.enabled_observables.intersection(env.active_observables)
    for key, item in spec["observations"].items():
        assert list(observations[key].shape) == item["shape"]
        assert str(observations[key].dtype) == item["dtype"]
        if item["aggregation_of"]:
            combined = np.concatenate([observations[name] for name in item["aggregation_of"]])
            np.testing.assert_array_equal(observations[key], combined)


def test_action_snapshot_and_public_controller_roundtrip(env):
    env.reset()
    spec = saved_spec("action_spec.json")
    assert inspect_actions(env) == spec
    low, high = env.action_spec
    assert spec["action_dim"] == env.action_dim == low.size == high.size
    np.testing.assert_array_equal(low, spec["low"])
    np.testing.assert_array_equal(high, spec["high"])
    action = np.random.default_rng(23).uniform(low, high)
    rebuilt = []
    for robot, layout in zip(env.robots, spec["robots"]):
        start, stop = layout["joint_slice"]
        robot_action = action[start:stop]
        controller = robot.composite_controller
        part_actions = controller.create_action_dict_from_action_vector(robot_action)
        assert list(part_actions) == [part["name"] for part in layout["parts"]]
        for part in layout["parts"]:
            joint_start, joint_stop = part["joint_slice"]
            np.testing.assert_array_equal(part_actions[part["name"]], action[joint_start:joint_stop])
            assert part_actions[part["name"]].size == part["action_dim"]
        rebuilt.append(controller.create_action_vector(part_actions))
    np.testing.assert_array_equal(np.concatenate(rebuilt), action)


def test_sensor_semantics_and_aggregate_membership(env):
    observations = env.reset()
    for index in range(len(env.robots)):
        prefix = env.robots[index].robot_model.naming_prefix
        np.testing.assert_allclose(observations[prefix + "joint_pos_cos"], np.cos(observations[prefix + "joint_pos"]))
        np.testing.assert_allclose(observations[prefix + "joint_pos_sin"], np.sin(observations[prefix + "joint_pos"]))
        np.testing.assert_allclose(
            observations[f"gripper{index}_to_handle{index}"],
            observations[f"handle{index}_xpos"] - observations[prefix + "eef_pos"],
        )
    spec = saved_spec("observation_spec.json")
    object_keys = spec["observations"]["object-state"]["aggregation_of"]
    assert "gripper0_to_handle0" in object_keys and "gripper1_to_handle1" in object_keys
    assert spec["observations"]["robot0_eef_quat"]["description"] != spec["observations"]["robot0_eef_quat_site"]["description"]


def test_reward_success_and_termination_source_snapshot(env):
    spec = saved_spec("env_spec.json")
    for key, function in (("reward", type(env).reward), ("success", type(env)._check_success), ("termination", type(env)._post_action)):
        assert source_evidence(function)["sha256"] == spec["source_evidence"][key]["sha256"]
    assert env.reward_scale == spec["reward_contract"]["reward_scale"]
    assert env.reward_shaping == spec["reward_contract"]["reward_shaping"]


def set_pot_elevation(env, elevation, quaternion=(1, 0, 0, 0)):
    # Test-only state injection uses the pot's free joint and MuJoCo wxyz quaternion.
    joint = env.pot.joints[0]
    qpos = np.array(env.sim.data.get_joint_qpos(joint), copy=True)
    qpos[3:] = quaternion
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()
    current = env.sim.data.site_xpos[env.pot_center_id][2] - env.pot.top_offset[2] - env.sim.data.site_xpos[env.table_top_id][2]
    qpos[2] += elevation - current
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()


def test_real_success_height_threshold(env):
    env.reset()
    margin = saved_spec("env_spec.json")["reward_contract"]["success_height_margin_m"]
    set_pot_elevation(env, margin - 0.001)
    assert not env._check_success()
    set_pot_elevation(env, margin + 0.001)
    assert env._check_success()


def test_real_success_is_independent_of_tilt_reward_and_done(env):
    env.reset()
    set_pot_elevation(env, 0.30)
    assert env._check_success()
    assert float(env.reward()) == pytest.approx(env.reward_scale)
    # A 90-degree tilt blocks the reward while retaining height-based task success.
    half = np.sqrt(0.5)
    set_pot_elevation(env, 0.30, (half, half, 0, 0))
    assert env._check_success()
    assert float(env.reward()) == pytest.approx(0.0)
    low, high = env.action_spec
    observations, reward, done, _ = env.step((low + high) / 2)
    assert env._check_success()
    assert not done and env.timestep == 1
    assert np.isfinite(reward)
    assert all(np.isfinite(value).all() for value in observations.values())


class MetricEnv:
    """Controlled success/reward sequence to exercise transient/reset success metrics."""
    action_dim = 1
    action_spec = (np.array([-1.0]), np.array([1.0]))

    def __init__(self, successes, rewards):
        self.successes = successes
        self.rewards = rewards
        self.horizon = len(rewards)
        self.index = 0

    def reset(self):
        self.index = 0
        return {"state": np.array([0.0])}

    def _check_success(self):
        return self.successes[self.index]

    def step(self, action):
        reward = self.rewards[self.index]
        self.index += 1
        return {"state": np.array([float(self.index)])}, reward, self.index == self.horizon, {}


@pytest.mark.parametrize("successes,rewards,ever,first,final", [
    ([False, True, False, False], [0.1, 0.2, 0.3], True, 1, False),
    ([False, False, False, False], [1000, 1000, 1000], False, None, False),
    ([True, False, False, False], [0, 0, 0], True, 0, False),
    ([False, False, False, True], [0, 0, 0], True, 3, True),
])
def test_episode_metrics_use_task_success_not_returns(successes, rewards, ever, first, final):
    metrics = run_episode(MetricEnv(successes, rewards), np.random.default_rng(0))
    assert metrics == {
        "episode_return": pytest.approx(sum(rewards)), "ever_success": ever,
        "first_success_step": first, "final_success": final, "episode_length": len(rewards),
    }
