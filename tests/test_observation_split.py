"""Measured observation partitions, source isolation, and full-episode validation."""

import numpy as np
import pytest

from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper


def assert_agent_values_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_reset_measured_dimensions_and_exact_sources(env):
    wrapper = MultiAgentWrapper(env)
    agents = wrapper.reset()
    raw = env._get_observations()
    assert set(agents) == {"agent_0", "agent_1"}
    for index, agent in enumerate(wrapper.agent_ids):
        own = np.concatenate((raw[f"robot{index}_proprio-state"], raw[f"gripper{index}_to_handle{index}"]))
        shared = np.concatenate([raw[key] for key in ("pot_pos", "pot_quat", "handle0_xpos", "handle1_xpos")])
        assert set(agents[agent]) == {"local", "shared_object", "actor_obs"}
        assert agents[agent]["local"].shape == (53,)
        assert agents[agent]["shared_object"].shape == (13,)
        assert agents[agent]["actor_obs"].shape == (66,)
        np.testing.assert_array_equal(agents[agent]["local"], own)
        np.testing.assert_array_equal(agents[agent]["shared_object"], shared)
        np.testing.assert_array_equal(agents[agent]["actor_obs"], np.concatenate((own, shared)))
    assert wrapper.observation_dims == {
        agent: {"local": 53, "shared_object": 13, "actor_obs": 66} for agent in wrapper.agent_ids
    }
    assert wrapper.critic_state_dim == 119 and wrapper.critic_state.shape == (119,)
    np.testing.assert_array_equal(wrapper.critic_state, np.concatenate([
        raw["robot0_proprio-state"], raw["robot1_proprio-state"], raw["object-state"],
    ]))


@pytest.mark.parametrize("private_robot", [0, 1])
def test_teammate_private_and_relative_signals_do_not_reach_other_actor(env, private_robot):
    wrapper = MultiAgentWrapper(env)
    env.reset()
    raw = env._get_observations()
    baseline = wrapper.split_observations(raw)
    altered = {key: value.copy() for key, value in raw.items()}
    private_keys = [key for key in raw if key.startswith(f"robot{private_robot}_")]
    private_keys.append(f"gripper{private_robot}_to_handle{private_robot}")
    for key in private_keys:
        altered[key] += 100
    # Counterfactual sensor values test direct data access, independently of physics.
    altered["object-state"] += 500
    split = wrapper.split_observations(altered)
    other_agent = f"agent_{1 - private_robot}"
    assert_agent_values_equal(split[other_agent], baseline[other_agent])
    assert not np.array_equal(split[f"agent_{private_robot}"]["local"], baseline[f"agent_{private_robot}"]["local"])
    assert not np.array_equal(wrapper.build_critic_state(altered), wrapper.build_critic_state(raw))


def test_raw_object_aggregate_is_critic_only(env):
    wrapper = MultiAgentWrapper(env)
    env.reset()
    raw = env._get_observations()
    baseline = wrapper.split_observations(raw)
    altered = {key: value.copy() for key, value in raw.items()}
    altered["object-state"] += 1000
    split = wrapper.split_observations(altered)
    for agent in wrapper.agent_ids:
        assert_agent_values_equal(split[agent], baseline[agent])
        assert "critic_state" not in split[agent] and "object-state" not in split[agent]
    assert not np.array_equal(wrapper.build_critic_state(altered), wrapper.build_critic_state(raw))


def test_split_does_not_modify_raw_observations_or_share_storage(env):
    wrapper = MultiAgentWrapper(env)
    env.reset()
    raw = env._get_observations()
    keys_before = list(raw)
    values_before = {key: value.copy() for key, value in raw.items()}
    agents = wrapper.split_observations(raw)
    critic = wrapper.build_critic_state(raw)
    for agent in agents.values():
        for value in agent.values():
            assert all(not np.shares_memory(value, source) for source in raw.values())
            value[:] = -123
    assert all(not np.shares_memory(critic, source) for source in raw.values())
    critic[:] = -456
    assert list(raw) == keys_before
    for key in raw:
        np.testing.assert_array_equal(raw[key], values_before[key])
        assert raw[key].shape == values_before[key].shape and raw[key].dtype == values_before[key].dtype


def test_actor_outputs_and_critic_cache_have_independent_storage(env):
    wrapper = MultiAgentWrapper(env)
    with pytest.raises(RuntimeError, match="reset"):
        _ = wrapper.critic_state
    agents = wrapper.reset()
    other_before = {key: value.copy() for key, value in agents["agent_1"].items()}
    actor_before = agents["agent_0"]["actor_obs"].copy()
    critic_before = wrapper.critic_state
    agents["agent_0"]["local"][:] = 17
    agents["agent_0"]["shared_object"][:] = 19
    np.testing.assert_array_equal(agents["agent_0"]["actor_obs"], actor_before)
    agents["agent_0"]["actor_obs"][:] = 23
    assert_agent_values_equal(agents["agent_1"], other_before)
    np.testing.assert_array_equal(wrapper.critic_state, critic_before)
    temporary_critic = wrapper.critic_state
    temporary_critic[:] = 29
    np.testing.assert_array_equal(wrapper.critic_state, critic_before)


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "missing_key", "extra_key", "nonfinite"])
def test_observation_contract_mismatch_is_rejected(env, mismatch):
    wrapper = MultiAgentWrapper(env)
    raw = env.reset()
    altered = {key: value.copy() for key, value in raw.items()}
    key = "robot0_proprio-state"
    if mismatch == "shape":
        altered[key] = altered[key][:-1]
    elif mismatch == "dtype":
        altered[key] = altered[key].astype(np.float32)
    elif mismatch == "missing_key":
        del altered[key]
    elif mismatch == "extra_key":
        altered["unexpected"] = np.zeros(1)
    else:
        altered[key][0] = np.nan
    with pytest.raises(ValueError):
        wrapper.split_observations(altered)


def test_full_random_episode_finite_data_and_upstream_success(env):
    wrapper = MultiAgentWrapper(env)
    agents = wrapper.reset()
    rng = np.random.default_rng(42)
    action_specs = wrapper.action_specs
    episode_return = 0.0
    for step in range(1, wrapper.horizon + 1):
        actions = [rng.uniform(*action_specs[agent]) for agent in wrapper.agent_ids]
        agents, reward, done, info = wrapper.step(*actions)
        episode_return += float(reward)
        assert np.isfinite(reward)
        assert isinstance(info, dict)
        assert bool(done) == (step == wrapper.horizon)
        assert wrapper.check_success() == bool(env._check_success())
        for agent in wrapper.agent_ids:
            for key, value in agents[agent].items():
                assert value.shape == (wrapper.observation_dims[agent][key],)
                assert np.isfinite(value).all()
        critic = wrapper.critic_state
        assert critic.shape == (wrapper.critic_state_dim,) and np.isfinite(critic).all()
        raw = env._get_observations()
        np.testing.assert_array_equal(critic, np.concatenate([
            raw["robot0_proprio-state"], raw["robot1_proprio-state"], raw["object-state"],
        ]))
    assert step == env.horizon == 200 and env.done
    assert np.isfinite(episode_return)
