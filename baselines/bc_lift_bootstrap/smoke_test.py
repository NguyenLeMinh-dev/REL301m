"""Real environment/actor/action-recorder/serialization smoke; no fake demos."""
from common import verify_stack, git_sha
from env import make_lift_env, environment_contract
from train_bc import make_model

import tempfile
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env


def main():
    versions = verify_stack()
    env = make_lift_env(0, horizon=8)
    try:
        check_env(env, warn=False, skip_render_check=True)
        obs, _ = env.reset()
        contract = environment_contract(env, horizon=8)
        assert obs.dtype == np.float32 and obs.shape == env.observation_space.shape
        model = make_model(env, seed=0)
        states, actions = [], []
        for step in range(8):
            action = np.zeros(7, dtype=np.float32); action[6] = .8
            current = obs.copy()
            obs, reward, terminal, timeout, info = env.step(action)
            expected = action.copy()
            if info["assist_active"]:
                expected[6] = -1
            np.testing.assert_array_equal(info["executed_action"], expected)
            assert np.array_equal(action, np.r_[np.zeros(6, dtype=np.float32), np.float32(.8)])
            assert obs.dtype == np.float32 and np.isfinite(obs).all() and np.isfinite(reward)
            assert not terminal and timeout == (step == 7)
            states.append(current); actions.append(info["executed_action"])
        inputs = torch.as_tensor(np.asarray(states, dtype=np.float32))
        targets = torch.as_tensor(np.asarray(actions, dtype=np.float32))
        before = [p.detach().clone() for p in model.actor.parameters()]
        loss = (model.actor(inputs, deterministic=True)-targets).square().mean()
        model.actor.optimizer.zero_grad(set_to_none=True); loss.backward(); model.actor.optimizer.step()
        assert torch.isfinite(loss) and any(not torch.equal(a, b) for a, b in zip(before, model.actor.parameters()))
        assert all(p.grad is None for p in model.critic.parameters())
        expected_action, _ = model.predict(obs, deterministic=True)
        with tempfile.TemporaryDirectory(prefix="bc-lift-smoke-") as directory:
            path = Path(directory)/"bc_sac_warmstart.zip"
            model.save(path)
            restored = SAC.load(path, env=env, device=model.device)
            actual, _ = restored.predict(obs, deterministic=True)
            np.testing.assert_array_equal(actual, expected_action)
            assert actual.dtype == np.float32 and env.action_space.contains(actual)
        print(f"OBS_DIM={contract['obs_dim']} ACTION_DIM={contract['action_dim']} OBS_DTYPE={obs.dtype}")
        print(f"TORCH={versions['torch']} CUDA_AVAILABLE={torch.cuda.is_available()}")
        print("EXECUTED_ACTION_CAPTURE=PASS BC_OPTIMIZER_STEP=PASS SAVE_RELOAD_PREDICT=PASS")
        print("Short real trajectory exercises collector-compatible fields; it is not labelled successful or saved as a demonstration.")
        print(f"GIT_COMMIT={git_sha()}")
        print("BC_SMOKE=PASS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
