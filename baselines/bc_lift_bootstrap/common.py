"""Small shared runtime/provenance helpers; no REL301m imports."""
from runtime_guard import enforce_phase3_python, project_root
enforce_phase3_python()

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = project_root()
BASE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ("robosuite", "mujoco", "torch", "stable-baselines3", "gymnasium", "numpy", "h5py", "tqdm")}


def git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def device(value):
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)


def verify_stack():
    versions = package_versions()
    expected = {"robosuite": "1.5.2", "mujoco": "3.9.0"}
    for name, version in expected.items():
        if versions[name] != version:
            raise RuntimeError(f"Expected {name}=={version}, got {versions[name]}")
    if not versions["torch"].startswith("2.7.1"):
        raise RuntimeError(f"Expected existing Torch 2.7.1 build, got {versions['torch']}")
    torch.set_num_threads(1)
    return versions


def default_source_model():
    for name in ("robosuite_sac_lift_stage1_assisted", "robosuite_official_sac_lift"):
        p = ROOT / "baselines" / name / "runs/stage1_assisted_seed0_100k/best_model.zip"
        if p.exists():
            return p
    return ROOT / "baselines/robosuite_sac_lift_stage1_assisted/runs/stage1_assisted_seed0_100k/best_model.zip"


def model_path(value):
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    # Allow the request's runs/... shorthand without guessing a different run.
    if not path.is_absolute() and path.parts and path.parts[0] == "runs":
        for name in ("robosuite_sac_lift_stage1_assisted", "robosuite_official_sac_lift"):
            candidate = ROOT / "baselines" / name / path
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"Model not found: {path}")


def checked_action(action, space):
    value = np.asarray(action, dtype=np.float32)
    if value.shape != space.shape or not np.isfinite(value).all():
        raise ValueError(f"Invalid action shape/values: {value.shape}")
    if np.any(value < space.low) or np.any(value > space.high):
        raise ValueError("Action outside live environment bounds")
    return value
