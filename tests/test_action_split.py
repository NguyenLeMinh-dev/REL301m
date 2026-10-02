"""Live bounds, correct robot ordering, and unchanged robosuite step semantics."""

import json

import numpy as np
import pytest

from rel301m.envs.contract import inspect_actions
from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
from rel301m.envs.robosuite_factory import PROJECT_ROOT


class RecordingEnv:
    """Observe the actual joint action/result while forwarding to the real simulator."""

    def __init__(self, env):
        self.base = env
        self.step_calls = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self):
        return self.base.reset()

    def step(self, action):
        self.step_calls += 1
        self.last_action = action.copy()
        self.last_result = self.base.step(action)
        self.raw_copy = {key: value.copy() for key, value in self.last_result[0].items()}
        return self.last_result


def test_measured_action_dimensions_order_and_live_bounds(env):
    wrapper = MultiAgentWrapper(env)
    metadata = inspect_actions(env)
    assert wrapper.action_dims == {"agent_0": 7, "agent_1": 7}
    low, high = env.action_spec
    actions = []
    for index, (agent, robot) in enumerate(zip(wrapper.agent_ids, metadata["robots"])):
        start, stop = robot["joint_slice"]
        agent_low, agent_high = wrapper.action_specs[agent]
        np.testing.assert_array_equal(agent_low, low[start:stop])
        np.testing.assert_array_equal(agent_high, high[start:stop])
        fraction = 0.2 if index == 0 else 0.7
        actions.append(agent_low + fraction * (agent_high - agent_low))
    joint = wrapper.join_actions(*actions)
    assert joint.shape == (14,)
    np.testing.assert_array_equal(joint, np.concatenate(actions))
    assert np.all(joint >= low) and np.all(joint <= high)
    for index, robot in enumerate(metadata["robots"]):
        start, stop = robot["joint_slice"]
        np.testing.assert_array_equal(joint[start:stop], actions[index])


def test_boundary_actions_are_valid_and_inputs_are_unchanged(env):
    wrapper = MultiAgentWrapper(env)
    specs = wrapper.action_specs
    action0 = specs["agent_0"][0]
    action1 = specs["agent_1"][1]
    before0, before1 = action0.copy(), action1.copy()
    joint = wrapper.join_actions(action0, action1)
    low, high = env.action_spec
    assert np.all(joint >= low) and np.all(joint <= high)
    np.testing.assert_array_equal(action0, before0)
    np.testing.assert_array_equal(action1, before1)
    assert not np.shares_memory(joint, action0) and not np.shares_memory(joint, action1)


def test_action_specs_are_copies(env):
    wrapper = MultiAgentWrapper(env)
    before = wrapper.action_specs
    returned = wrapper.action_specs
    returned["agent_0"][0][:] = -100
    returned["agent_1"][1][:] = 100
    current = wrapper.action_specs
    for agent in wrapper.agent_ids:
        for index in range(2):
            np.testing.assert_array_equal(current[agent][index], before[agent][index])


@pytest.mark.parametrize("bad_agent", [0, 1])
@pytest.mark.parametrize("invalid", ["wrong_dimension", "matrix", "nan", "inf", "above_bound", "below_bound", "strings"])
def test_invalid_action_is_rejected_before_simulator_step(env, bad_agent, invalid):
    recording = RecordingEnv(env)
    wrapper = MultiAgentWrapper(recording)
    wrapper.reset()
    specs = wrapper.action_specs
    actions = [(specs[agent][0] + specs[agent][1]) / 2 for agent in wrapper.agent_ids]
    low, high = specs[wrapper.agent_ids[bad_agent]]
    if invalid == "wrong_dimension":
        actions[bad_agent] = np.zeros(low.size + 1)
    elif invalid == "matrix":
        actions[bad_agent] = actions[bad_agent].reshape(1, -1)
    elif invalid == "nan":
        actions[bad_agent][0] = np.nan
    elif invalid == "inf":
        actions[bad_agent][0] = np.inf
    elif invalid == "above_bound":
        actions[bad_agent] = high + 0.01
    elif invalid == "below_bound":
        actions[bad_agent] = low - 0.01
    else:
        actions[bad_agent] = ["0"] * low.size
    with pytest.raises(ValueError, match=wrapper.agent_ids[bad_agent]):
        wrapper.step(*actions)
    assert recording.step_calls == 0


def test_changed_phase1_action_metadata_is_rejected(env, tmp_path):
    original = PROJECT_ROOT / "artifacts"
    (tmp_path / "observation_spec.json").write_bytes((original / "observation_spec.json").read_bytes())
    spec = json.loads((original / "action_spec.json").read_text())
    spec["robots"][0]["joint_slice"][1] += 1
    (tmp_path / "action_spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Live action metadata"):
        MultiAgentWrapper(env, contract_dir=tmp_path)


def test_step_preserves_joint_action_reward_done_info_and_raw_observations(env):
    recording = RecordingEnv(env)
    wrapper = MultiAgentWrapper(recording)
    wrapper.reset()
    specs = wrapper.action_specs
    action0 = specs["agent_0"][0] + 0.3 * (specs["agent_0"][1] - specs["agent_0"][0])
    action1 = specs["agent_1"][0] + 0.6 * (specs["agent_1"][1] - specs["agent_1"][0])
    agents, reward, done, info = wrapper.step(action0, action1)
    raw, expected_reward, expected_done, expected_info = recording.last_result
    np.testing.assert_array_equal(recording.last_action, np.concatenate((action0, action1)))
    assert reward is expected_reward and done is expected_done and info is expected_info
    assert wrapper.check_success() == bool(env._check_success())
    for key in raw:
        np.testing.assert_array_equal(raw[key], recording.raw_copy[key])
    assert all(np.isfinite(value).all() for agent in agents.values() for value in agent.values())


def test_success_with_zero_reward_does_not_end_wrapped_episode(env):
    wrapper = MultiAgentWrapper(env)
    wrapper.reset()
    joint = env.pot.joints[0]
    qpos = np.array(env.sim.data.get_joint_qpos(joint), copy=True)
    half = np.sqrt(0.5)
    qpos[3:] = (half, half, 0, 0)
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()
    height = env.sim.data.site_xpos[env.pot_center_id][2] - env.pot.top_offset[2] - env.sim.data.site_xpos[env.table_top_id][2]
    qpos[2] += 0.30 - height
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()
    assert wrapper.check_success() and float(env.reward()) == pytest.approx(0.0)
    specs = wrapper.action_specs
    actions = [(specs[agent][0] + specs[agent][1]) / 2 for agent in wrapper.agent_ids]
    _, reward, done, _ = wrapper.step(*actions)
    assert wrapper.check_success() == bool(env._check_success()) is True
    assert reward == pytest.approx(0.0)
    assert not done
