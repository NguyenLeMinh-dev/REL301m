import numpy as np
import pytest

from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
from rel301m.envs.reward_components import REWARD_COMPONENT_FIELDS, reward_components
from rel301m.imitation.scripted_teacher import ScriptedLiftTeacher


def test_reward_decomposition_matches_upstream_through_random_full_episode(env):
    wrapper = MultiAgentWrapper(env)
    wrapper.reset()
    rng = np.random.default_rng(3)
    for _ in range(wrapper.horizon):
        actions = [rng.uniform(*wrapper.action_specs[a]) for a in wrapper.agent_ids]
        _, reward, done, _ = wrapper.step(*actions)
        components = reward_components(env, reward)
        assert np.isfinite(list(components.values())).all()
        assert np.isclose(sum(components[k] for k in REWARD_COMPONENT_FIELDS), reward)
    assert done
    with pytest.raises(RuntimeError, match='upstream'):
        reward_components(env, reward + 1)


def test_components_match_real_scripted_grasp_lift_and_native_success():
    from rel301m.envs.robosuite_factory import make_two_arm_lift
    wrapper = MultiAgentWrapper(make_two_arm_lift(seed=40000))
    seen = dict(success=False, grasp=False, lift=False)
    try:
        wrapper.reset()
        teacher = ScriptedLiftTeacher(wrapper)
        for _ in range(wrapper.horizon):
            _, reward, _, _ = wrapper.step(*teacher.act())
            parts = reward_components(wrapper._env, reward)
            seen['success'] |= parts['success_reward'] > 0
            seen['grasp'] |= parts['grasp_reward_0'] + parts['grasp_reward_1'] > 0
            seen['lift'] |= parts['lift_reward'] > 0
        assert all(seen.values())
    finally:
        wrapper.close()
