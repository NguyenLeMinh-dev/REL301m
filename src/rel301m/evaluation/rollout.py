"""Roll out valid random actions and keep success separate from reward."""

import numpy as np

from rel301m.envs.contract import action_bounds, validate_action, validate_observations, validate_reward


def run_episode(env, rng):
    reference = env.reset()
    validate_observations(reference)
    low, high = action_bounds(env)
    initial_success = bool(env._check_success())
    ever_success = initial_success
    first_success_step = 0 if initial_success else None
    final_success = initial_success
    episode_return = 0.0
    for step in range(1, env.horizon + 1):
        action = rng.uniform(low, high)
        validate_action(action, low, high)
        observations, reward, done, _ = env.step(action)
        validate_observations(observations, reference)
        validate_reward(reward)
        episode_return += float(reward)
        final_success = bool(env._check_success())
        if final_success and first_success_step is None:
            first_success_step = step
        ever_success |= final_success
        if bool(done) != (step == env.horizon):
            raise RuntimeError(f"Unexpected done={done} at step {step}/{env.horizon}")
    if not np.isfinite(episode_return):
        raise ValueError("Episode return contains NaN/Inf")
    return {
        "episode_return": episode_return, "ever_success": ever_success,
        "first_success_step": first_success_step, "final_success": final_success,
        "episode_length": step,
    }
