"""Regression checks for BC labels/splitting/checkpoints. Synthetic unit arrays
are never used by the collector or represented as real demonstrations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore:.*precision lowered by casting to float32.*:UserWarning")

from common import sha256, verify_stack
from dataset import FIELDS, pack_episodes, validate_dataset, split_episodes, load_dataset, save_dataset
from env import make_lift_env, environment_contract
from assisted_env import make_lift_env as make_source_env
from train_bc import make_model, state_hash
from stable_baselines3 import SAC


def synthetic():
    horizon, d = 4, 3
    episodes = []
    for i in range(5):
        obs = np.arange(horizon*d, dtype=np.float32).reshape(horizon, d)+i*100
        nxt = np.r_[obs[1:], obs[-1:]+1]
        policy = np.zeros((horizon, 7), dtype=np.float32); policy[:, 6] = .8
        distance = np.array([.1, .1, .01, .01], dtype=np.float64)
        active = distance > .05
        executed = policy.copy(); executed[active, 6] = -1
        done = np.array([False, False, False, True])
        episode = {"observations": obs, "next_observations": nxt, "actions": executed,
                   "executed_actions": executed.copy(), "policy_actions": policy, "rewards": np.ones(horizon, dtype=np.float32),
                   "episode_ids": np.full(horizon, i, dtype=np.int32), "steps": np.arange(horizon, dtype=np.int32),
                   "dones": done, "ever_success": np.ones(horizon, dtype=bool), "terminated": np.zeros(horizon, dtype=bool),
                   "truncated": done, "current_success": done, "distance_to_cube": distance, "cube_height": np.zeros(horizon, dtype=np.float32),
                   "grasp": done, "assist_active": active}
        episodes.append(episode)
    metadata = {"obs_dim": d, "action_dim": 7, "environment": {"horizon": horizon}, "assist_distance": .05,
                "episode_seeds": list(range(5)), "source_attempts": list(range(5)),
                "number_of_successful_episodes": 5, "number_of_transitions": 20}
    return pack_episodes(episodes, metadata), metadata


def test_dataset_roundtrip_and_executed_labels(tmp_path):
    data, metadata = synthetic()
    validate_dataset(data, metadata)
    save_dataset(tmp_path/"unit.npz", data, metadata)
    loaded, _ = load_dataset(tmp_path/"unit.npz")
    assert np.all(loaded["actions"][loaded["assist_active"], 6] == -1)
    assert np.all(loaded["policy_actions"][loaded["assist_active"], 6] == np.float32(.8))
    with pytest.raises(FileExistsError):
        save_dataset(tmp_path/"unit.npz", data, metadata)


@pytest.mark.parametrize("field,change", [
    ("observations", lambda x: x.__setitem__((0,0), np.nan)),
    ("actions", lambda x: x.__setitem__((0,6), .8)),
    ("next_observations", lambda x: x.__setitem__((0,0), 100)),
    ("ever_success", lambda x: x.__setitem__(0, False)),
    ("current_success", lambda x: x.__setitem__(slice(0,4), False)),
    ("dones", lambda x: x.__setitem__(0, True)),
    ("steps", lambda x: x.__setitem__(0, 1)),
    ("policy_actions", lambda x: x.__setitem__((0,0), 1.1)),
])
def test_reject_corrupt_demonstrations(field, change):
    data, metadata = synthetic()
    change(data[field])
    with pytest.raises(ValueError):
        validate_dataset(data, metadata)


def test_reject_dtype_duplicate_seed_and_partial_episode():
    data, metadata = synthetic()
    data["actions"] = data["actions"].astype(np.float64)
    with pytest.raises(ValueError): validate_dataset(data, metadata)
    data, metadata = synthetic(); metadata["episode_seeds"][1] = metadata["episode_seeds"][0]
    with pytest.raises(ValueError): validate_dataset(data, metadata)
    data, metadata = synthetic(); data = {k: v[:-1] for k, v in data.items()}
    with pytest.raises(ValueError): validate_dataset(data, metadata)


def test_split_is_deterministic_and_has_no_episode_leakage():
    data, _ = synthetic()
    a, b, ai, bi = split_episodes(data, 42)
    same = split_episodes(data, 42)
    assert len(a) == 4 and len(b) == 1
    for x, y in zip((a,b,ai,bi), same): np.testing.assert_array_equal(x,y)
    assert not set(a).intersection(b)
    assert set(data["episode_ids"][ai]) == set(a)
    assert set(data["episode_ids"][bi]) == set(b)
    assert not set(ai).intersection(bi)


def test_recorder_is_behavior_neutral_and_logs_full_post_assistance_action():
    verify_stack()
    recorded = make_lift_env(13, horizon=5)
    source = make_source_env(13, horizon=5)
    try:
        a, _ = recorded.reset(); b, _ = source.reset()
        np.testing.assert_array_equal(a, b)
        for step in range(5):
            action = np.array([.2, -.1, .1, 0., 0., 0., .8], dtype=np.float32)
            before = action.copy()
            a, ra, ta, xa, ia = recorded.step(action)
            b, rb, tb, xb, ib = source.step(action)
            np.testing.assert_array_equal(action, before)
            np.testing.assert_array_equal(a, b)
            assert (ra, ta, xa) == (rb, tb, xb)
            np.testing.assert_array_equal(ia["executed_action"][:6], action[:6])
            assert ia["executed_action"][6] == (-1 if ia["assist_active"] else action[6])
            assert ia["current_success"] == ib["current_success"]
    finally:
        recorded.close(); source.close()


def test_bc_actor_step_preserves_critics_and_std_heads_and_reloads(tmp_path):
    verify_stack()
    env = make_lift_env(1, horizon=2)
    try:
        obs, _ = env.reset()
        model = make_model(env, seed=0)
        critic = state_hash(model.critic); std = state_hash(model.actor.log_std)
        obs_tensor = torch.as_tensor(np.repeat(obs[None], 3, axis=0))
        target = torch.zeros((3, 7))
        actor_before = state_hash(model.actor)
        loss = (model.actor(obs_tensor, deterministic=True)-target).square().mean()
        model.actor.optimizer.zero_grad(set_to_none=True); loss.backward(); model.actor.optimizer.step()
        assert state_hash(model.actor) != actor_before
        assert state_hash(model.critic) == critic and state_hash(model.actor.log_std) == std
        assert all(p.grad is None for p in model.critic.parameters())
        model.save(tmp_path/"bc.zip")
        restored = SAC.load(tmp_path/"bc.zip", env=env, device=model.device)
        assert state_hash(model.actor) == state_hash(restored.actor)
        expected, _ = model.predict(obs, deterministic=True)
        actual, _ = restored.predict(obs, deterministic=True)
        np.testing.assert_array_equal(actual, expected)
        assert callable(restored.learn)  # Actual learn() is deliberately not run.
    finally:
        env.close()


def test_live_contract_survives_json_roundtrip():
    env = make_lift_env(0)
    try:
        contract = environment_contract(env)
        assert contract == json.loads(json.dumps(contract))
    finally:
        env.close()
