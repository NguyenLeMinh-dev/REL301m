"""Successful full episodes, raw observations and executed action labels."""
from common import write_json

import json
from pathlib import Path
import numpy as np

FIELDS = {"observations": np.float32, "actions": np.float32, "executed_actions": np.float32,
          "policy_actions": np.float32, "next_observations": np.float32, "rewards": np.float32,
          "episode_ids": np.int32, "steps": np.int32, "dones": np.bool_, "ever_success": np.bool_,
          "terminated": np.bool_, "truncated": np.bool_, "current_success": np.bool_,
          "distance_to_cube": np.float64, "cube_height": np.float32, "grasp": np.bool_, "assist_active": np.bool_}


def pack_episodes(episodes, metadata):
    d = int(metadata["obs_dim"])
    data = {}
    for key, dtype in FIELDS.items():
        if episodes:
            data[key] = np.concatenate([np.asarray(ep[key], dtype=dtype) for ep in episodes])
        else:
            shape = (0, d) if key in ("observations", "next_observations") else ((0, 7) if key in ("actions", "executed_actions", "policy_actions") else (0,))
            data[key] = np.empty(shape, dtype=dtype)
    return data


def save_dataset(path, data, metadata):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # An open handle prevents numpy silently appending another .npz suffix.
    with path.open("wb") as stream:
        np.savez_compressed(stream, **data, metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    write_json(path.with_suffix(".json"), metadata)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            raise ValueError("Dataset missing metadata_json")
        metadata = json.loads(str(archive["metadata_json"].item()))
        data = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
    validate_dataset(data, metadata)
    return data, metadata


def validate_dataset(data, metadata, *, allow_empty=False):
    missing = set(FIELDS) - set(data)
    if missing:
        raise ValueError(f"Missing dataset fields: {sorted(missing)}")
    n = len(data["rewards"])
    d = int(metadata["obs_dim"])
    if metadata["action_dim"] != 7 or d <= 0:
        raise ValueError("Invalid metadata dimensions")
    for key, dtype in FIELDS.items():
        value = data[key]
        shape = (n, d) if key in ("observations", "next_observations") else ((n, 7) if key in ("actions", "executed_actions", "policy_actions") else (n,))
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"{key}: expected {shape}/{np.dtype(dtype)}, got {value.shape}/{value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"NaN/Inf in {key}")
    if n == 0:
        if not allow_empty:
            raise ValueError("Dataset contains no successful episodes")
        return {"episodes": 0, "transitions": 0, "obs_dim": d, "action_dim": 7}
    for key in ("actions", "executed_actions", "policy_actions"):
        if (data[key] < -1).any() or (data[key] > 1).any():
            raise ValueError(f"Out-of-bounds {key}")
    if not np.array_equal(data["actions"], data["executed_actions"]):
        raise ValueError("BC actions differ from executed_actions")
    if not data["ever_success"].all():
        raise ValueError("Failed episode marked as demonstration")
    ids = data["episode_ids"]
    cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, n]
    blocks = [ids[start] for start in cuts[:-1]]
    if len(set(map(int, blocks))) != len(blocks):
        raise ValueError("An episode ID appears in multiple disjoint blocks")
    seeds = metadata["episode_seeds"]
    source_attempts = metadata["source_attempts"]
    if len(seeds) != len(blocks) or len(set(seeds)) != len(seeds) or len(source_attempts) != len(blocks) or len(set(source_attempts)) != len(blocks):
        raise ValueError("Duplicate/missing episode seed or source attempt")
    horizon = int(metadata["environment"]["horizon"])
    for start, stop in zip(cuts[:-1], cuts[1:]):
        if stop - start != horizon or not np.array_equal(data["steps"][start:stop], np.arange(horizon)):
            raise ValueError("Malformed/truncated demonstration episode")
        if not np.array_equal(data["next_observations"][start:stop-1], data["observations"][start+1:stop]):
            raise ValueError("Transition leakage or next-observation misalignment")
        if not data["current_success"][start:stop].any():
            raise ValueError("No native success measurement in accepted episode")
        expected_done = np.zeros(horizon, dtype=bool); expected_done[-1] = True
        if not np.array_equal(data["dones"][start:stop], expected_done):
            raise ValueError("Malformed done boundary")
        if data["terminated"][start:stop].any() or not np.array_equal(data["truncated"][start:stop], expected_done):
            raise ValueError("Native termination/time-limit semantics changed")
    assisted = data["distance_to_cube"] > metadata["assist_distance"]
    if not np.array_equal(assisted, data["assist_active"]):
        raise ValueError("Assistance flag/pre-step distance mismatch")
    expected_actions = data["policy_actions"].copy()
    expected_actions[assisted, 6] = -1
    if not np.array_equal(expected_actions, data["actions"]):
        raise ValueError("Executed actions do not match source assistance rule")
    if metadata["number_of_successful_episodes"] != len(blocks) or metadata["number_of_transitions"] != n:
        raise ValueError("Metadata episode/transition count mismatch")
    return {"episodes": len(blocks), "transitions": n, "obs_dim": d, "action_dim": 7,
            "obs_dtype": str(data["observations"].dtype), "action_dtype": str(data["actions"].dtype),
            "action_range": [float(data["actions"].min()), float(data["actions"].max())],
            "successful_episodes": len(blocks)}


def split_episodes(data, seed, fraction=0.8):
    ids = np.unique(data["episode_ids"])
    if len(ids) < 2:
        raise ValueError("At least two successful episodes are required for train/validation splitting")
    shuffled = np.random.default_rng(seed).permutation(ids)
    count = min(len(ids)-1, max(1, int(np.floor(len(ids) * fraction))))
    train, val = shuffled[:count], shuffled[count:]
    return train, val, np.flatnonzero(np.isin(data["episode_ids"], train)), np.flatnonzero(np.isin(data["episode_ids"], val))
