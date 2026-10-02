"""Sustained-success prefixes with truthful environment flags and BC labels."""
from common import write_json

import json
import hashlib
from pathlib import Path
import numpy as np

FIELDS = {"observations": np.float32, "actions": np.float32, "executed_actions": np.float32,
          "policy_actions": np.float32, "next_observations": np.float32, "rewards": np.float32,
          "episode_ids": np.int32, "steps": np.int32, "dones": np.bool_, "demo_end": np.bool_, "ever_success": np.bool_,
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


def first_sustained_interval(success, hold_steps):
    """Return 1-based native-success start and prefix length, or None."""
    if not isinstance(hold_steps, int) or isinstance(hold_steps, bool) or hold_steps < 1:
        raise ValueError("success_hold_steps must be a positive integer")
    count = 0
    for index, flag in enumerate(success):
        count = count + 1 if flag else 0
        if count == hold_steps:
            return index - hold_steps + 2, index + 1
    return None


def extract_demo(trajectory, hold_steps):
    interval = first_sustained_interval(trajectory["current_success"], hold_steps)
    if interval is None:
        return None
    start, stop = interval
    demo = {key: np.asarray(value)[:stop].copy() for key, value in trajectory.items()}
    demo["demo_end"] = np.zeros(stop, dtype=bool)
    demo["demo_end"][-1] = True
    # Never change dones/terminated/truncated to represent a data extraction cut.
    return demo, start, stop


def validate_dataset(data, metadata, *, allow_empty=False):
    if metadata.get("schema_version") != 2:
        raise ValueError("Expected sustained-demo schema 2; recollect legacy full-episode datasets")
    missing = set(FIELDS) - set(data)
    if missing:
        raise ValueError(f"Missing dataset fields: {sorted(missing)}")
    hold = metadata["success_hold_steps"]
    first_sustained_interval([], hold)  # Validate configuration even for empty files.
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
    if metadata["number_of_transitions"] != n:
        raise ValueError("Metadata transition count mismatch")
    if n == 0:
        if metadata["number_of_successful_episodes"] or metadata["demonstrations"]:
            raise ValueError("Empty dataset has nonempty episode metadata")
        if not allow_empty:
            raise ValueError("Dataset contains no sustained-success episodes")
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
    blocks = [int(ids[start]) for start in cuts[:-1]]
    if min(blocks) < 0 or len(set(blocks)) != len(blocks):
        raise ValueError("Negative/disjoint duplicate episode IDs")
    seeds, attempts, descriptions = metadata["episode_seeds"], metadata["source_attempts"], metadata["demonstrations"]
    if any(len(values) != len(blocks) for values in (seeds, attempts, descriptions)) or len(set(seeds)) != len(seeds) or len(set(attempts)) != len(attempts):
        raise ValueError("Duplicate/missing episode identity or metadata")
    horizon = int(metadata["environment"]["horizon"])
    lengths = []
    for ep, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
        length = stop - start
        lengths.append(int(length))
        if not 0 < length <= horizon or not np.array_equal(data["steps"][start:stop], np.arange(length)):
            raise ValueError("Malformed demonstration prefix/steps")
        if not np.array_equal(data["next_observations"][start:stop-1], data["observations"][start+1:stop]):
            raise ValueError("Transition leakage or next-observation misalignment")
        flags = data["current_success"][start:stop]
        interval = first_sustained_interval(flags, hold)
        if interval is None or interval[1] != length:
            raise ValueError("Demo must end at the first completed sustained-success interval")
        expected_end = np.zeros(length, dtype=bool); expected_end[-1] = True
        if not np.array_equal(data["demo_end"][start:stop], expected_end):
            raise ValueError("Malformed demo_end boundary")
        expected_timeout = data["steps"][start:stop] == horizon-1
        if data["terminated"][start:stop].any() or not np.array_equal(data["truncated"][start:stop], expected_timeout):
            raise ValueError("Demo cut falsely changes native termination/time-limit semantics")
        if not np.array_equal(data["dones"][start:stop], expected_timeout):
            raise ValueError("Demo boundary falsely represented as an MDP terminal")
        record = descriptions[ep]
        first = 0 if record.get("initial_success", False) else int(np.flatnonzero(flags)[0]+1)
        if (record["episode_id"] != blocks[ep] or record["seed"] != seeds[ep] or record["source_attempt"] != attempts[ep]
                or record["first_success_step"] != first or record["sustained_success_start"] != interval[0]
                or record["success_hold_steps"] != hold or record["saved_transition_count"] != length
                or record["verified_rollout_length"] != horizon
                or record["saved_end_terminated"] != bool(data["terminated"][stop-1])
                or record["saved_end_truncated"] != bool(data["truncated"][stop-1])
                or record["saved_final_observation_sha256"] != hashlib.sha256(data["next_observations"][stop-1].tobytes()).hexdigest()):
            raise ValueError("Sustained-success episode metadata mismatch")
    assisted = data["distance_to_cube"] > metadata["assist_distance"]
    if not np.array_equal(assisted, data["assist_active"]):
        raise ValueError("Assistance flag/pre-step distance mismatch")
    expected_actions = data["policy_actions"].copy()
    expected_actions[assisted, 6] = -1
    if not np.array_equal(expected_actions, data["actions"]):
        raise ValueError("Executed actions do not match source assistance rule")
    if metadata["number_of_successful_episodes"] != len(blocks):
        raise ValueError("Metadata episode count mismatch")
    return {"episodes": len(blocks), "transitions": n, "obs_dim": d, "action_dim": 7,
            "obs_dtype": str(data["observations"].dtype), "action_dtype": str(data["actions"].dtype),
            "action_range": [float(data["actions"].min()), float(data["actions"].max())],
            "successful_episodes": len(blocks), "saved_episode_lengths": lengths,
            "success_hold_steps": hold, "demo_boundary_semantics": "demo_end separate from truthful env flags"}


def split_episodes(data, seed, fraction=0.8):
    ids = np.unique(data["episode_ids"])
    if len(ids) < 2:
        raise ValueError("At least two successful episodes are required for train/validation splitting")
    shuffled = np.random.default_rng(seed).permutation(ids)
    count = min(len(ids)-1, max(1, int(np.floor(len(ids) * fraction))))
    train, val = shuffled[:count], shuffled[count:]
    return train, val, np.flatnonzero(np.isin(data["episode_ids"], train)), np.flatnonzero(np.isin(data["episode_ids"], val))
