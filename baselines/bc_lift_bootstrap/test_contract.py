"""Regression checks for BC labels/splitting/checkpoints. Synthetic unit arrays
are never used by the collector or represented as real demonstrations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import hashlib
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore:.*precision lowered by casting to float32.*:UserWarning")

from common import sha256, verify_stack
from dataset import FIELDS, pack_episodes, validate_dataset, split_episodes, load_dataset, save_dataset, extract_demo, first_sustained_interval
from env import make_lift_env, environment_contract
from assisted_env import make_lift_env as make_source_env, OpenUntilCloseWrapper
from train_bc import make_model, state_hash
from stable_baselines3 import SAC


def synthetic():
    horizon, d, hold = 20, 3, 2
    episodes, descriptions = [], []
    for i in range(5):
        length = 5+i
        obs = np.arange(length*d, dtype=np.float32).reshape(length, d)+i*100
        nxt = np.r_[obs[1:], obs[-1:]+1]
        policy = np.zeros((length, 7), dtype=np.float32); policy[:, 6] = .8
        distance = np.full(length, .1, dtype=np.float64); distance[-2:] = .01
        active = distance > .05
        executed = policy.copy(); executed[active, 6] = -1
        flags = np.zeros(length, dtype=bool); flags[0] = True; flags[-2:] = True
        demo_end = np.zeros(length, dtype=bool); demo_end[-1] = True
        episode = {"observations": obs, "next_observations": nxt, "actions": executed,
                   "executed_actions": executed.copy(), "policy_actions": policy, "rewards": np.ones(length, dtype=np.float32),
                   "episode_ids": np.full(length, i, dtype=np.int32), "steps": np.arange(length, dtype=np.int32),
                   "dones": np.zeros(length, dtype=bool), "demo_end": demo_end, "ever_success": np.ones(length, dtype=bool),
                   "terminated": np.zeros(length, dtype=bool), "truncated": np.zeros(length, dtype=bool), "current_success": flags,
                   "distance_to_cube": distance, "cube_height": np.zeros(length, dtype=np.float32), "grasp": flags, "assist_active": active}
        episodes.append(episode)
        descriptions.append({"episode_id": i, "seed": i, "source_attempt": i, "first_success_step": 1,
                             "sustained_success_start": length-1, "success_hold_steps": hold, "saved_transition_count": length,
                             "verified_rollout_length": horizon, "saved_end_terminated": False, "saved_end_truncated": False,
                             "saved_final_observation_sha256": hashlib.sha256(nxt[-1].tobytes()).hexdigest()})
    metadata = {"schema_version": 2, "obs_dim": d, "action_dim": 7, "environment": {"horizon": horizon}, "assist_distance": .05,
                "episode_seeds": list(range(5)), "source_attempts": list(range(5)), "success_hold_steps": hold,
                "rejected_transient_success": 0, "demonstrations": descriptions,
                "number_of_successful_episodes": 5, "number_of_transitions": sum(len(ep["steps"]) for ep in episodes)}
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
    ("current_success", lambda x: x.__setitem__(slice(0,5), False)),
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


@pytest.mark.parametrize("flags,hold,expected", [
    ([False, True, False, True, True, True, False], 3, (4,6)),
    ([True, False, True, False], 2, None),
    ([False, True], 1, (2,2)),
    ([False, False, True, True], 2, (3,4)),
    ([True, True, False, True, True], 2, (1,2)),
])
def test_first_sustained_success_and_transient_rejection(flags, hold, expected):
    assert first_sustained_interval(flags, hold) == expected


def test_extraction_cuts_post_success_without_fake_terminal():
    flags = [False, True, False, True, True, True, False, False]
    trajectory = {"current_success": flags, "steps": list(range(8)), "terminated": [False]*8,
                  "truncated": [False]*7+[True], "dones": [False]*7+[True]}
    demo, start, length = extract_demo(trajectory, 3)
    assert (start, length) == (4,6)
    assert demo["current_success"][-3:].all()
    assert not demo["dones"].any() and not demo["terminated"].any() and not demo["truncated"].any()
    np.testing.assert_array_equal(demo["demo_end"], [False]*5+[True])
    assert len(trajectory["current_success"]) == 8 and trajectory["truncated"][-1]
    assert extract_demo({"current_success": [False, True, False]}, 2) is None


def test_extraction_preserves_real_horizon_timeout():
    trajectory = {"current_success": [False, False, True, True], "steps": list(range(4)),
                  "terminated": [False]*4, "truncated": [False]*3+[True], "dones": [False]*3+[True]}
    demo, _, _ = extract_demo(trajectory, 2)
    assert demo["demo_end"][-1] and demo["truncated"][-1] and demo["dones"][-1]
    assert not demo["terminated"].any()


@pytest.mark.parametrize("field", ["dones", "truncated", "terminated"])
def test_reject_fake_env_terminal_on_demo_cut(field):
    data, metadata = synthetic()
    data[field][4] = True
    with pytest.raises(ValueError): validate_dataset(data, metadata)


def test_variable_lengths_and_metadata_consistency():
    data, metadata = synthetic()
    result = validate_dataset(data, metadata)
    assert result["saved_episode_lengths"] == [5,6,7,8,9]
    assert not data["dones"].any() and data["demo_end"].sum() == 5
    metadata["demonstrations"][0]["sustained_success_start"] = 1
    with pytest.raises(ValueError): validate_dataset(data, metadata)


def test_reject_old_schema_and_wrong_demo_boundary():
    data, metadata = synthetic()
    metadata["schema_version"] = 1
    with pytest.raises(ValueError, match="schema 2"): validate_dataset(data, metadata)
    data, metadata = synthetic(); data["demo_end"][4] = False
    with pytest.raises(ValueError): validate_dataset(data, metadata)


def test_unassisted_has_no_gripper_gate_and_same_live_contract():
    a = make_lift_env(13, horizon=3)
    b = make_lift_env(13, horizon=3, mode="unassisted")
    try:
        oa, _ = a.reset(); ob, _ = b.reset()
        np.testing.assert_array_equal(oa, ob)
        ca = environment_contract(a, horizon=3)
        cb = environment_contract(b, horizon=3, mode="unassisted")
        for key in ca:
            if key not in ("evaluation_mode", "gripper_assistance"):
                assert ca[key] == cb[key]
        cursor = b
        while hasattr(cursor, "env"):
            assert not isinstance(cursor, OpenUntilCloseWrapper)
            cursor = cursor.env
        action = np.zeros(7, dtype=np.float32); action[6] = .8
        _, _, _, _, ia = a.step(action)
        _, _, _, _, ib = b.step(action)
        assert ia["distance_to_cube_m"] > .05 and ia["executed_action"][6] == -1
        assert not ib["assist_active"] and ib["executed_action"][6] == action[6]
        np.testing.assert_array_equal(ib["executed_action"], action)
    finally:
        a.close(); b.close()


def test_dual_evaluation_reports_separate_modes_and_identical_initial_states():
    from evaluate_bc import evaluate_mode
    class ConstantPolicy:
        def predict(self, obs, deterministic):
            action = np.zeros(7, dtype=np.float32); action[6] = .8
            return action, None
    # Replace only episode length in this regression; production eval remains 500.
    import evaluate_bc
    original = evaluate_bc.run_episode
    def short(*args, **kwargs):
        return original(*args, horizon=3, **kwargs)
    from unittest.mock import patch
    with patch.object(evaluate_bc, "run_episode", short):
        a = evaluate_mode(ConstantPolicy(), [13], .05, "assisted")
        b = evaluate_mode(ConstantPolicy(), [13], .05, "unassisted")
    assert a["evaluation_mode"] == "assisted" and b["evaluation_mode"] == "unassisted"
    assert a["gripper_override_count"] > 0 and b["gripper_override_count"] == 0
    assert a["episode_results"][0]["initial_state_sha256"] == b["episode_results"][0]["initial_state_sha256"]


def test_reject_final_saved_state_leaking_from_another_episode():
    data, metadata = synthetic()
    data["next_observations"][4] = data["observations"][5]
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_dataset(data, metadata)
