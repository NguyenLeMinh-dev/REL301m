#!/usr/bin/env python3
"""Freeze host/Python evidence and verify state-only MuJoCo without requiring torch."""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import site
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def command_output(command, *, cwd=None):
    """Retain return codes and stderr so failures cannot appear as a PASS."""
    try:
        process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
        return {"command": command, "returncode": process.returncode,
                "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "returncode": -1, "stdout": "", "stderr": str(error)}


def check_python_path():
    ros_pattern = re.compile(r"/opt/ros(?:/|$)|/(?:ros2?_ws|catkin_ws)(?:/|$)", re.IGNORECASE)
    ros_paths = [path for path in sys.path if ros_pattern.search(path)]
    ros_modules = {}
    for name in ("rclpy", "rospy", "ament_index_python", "launch_ros"):
        spec = importlib.util.find_spec(name)
        if spec is not None:
            ros_modules[name] = spec.origin or list(spec.submodule_search_locations or [])
    config_path = Path(sys.prefix) / "pyvenv.cfg"
    venv_config = config_path.read_text() if config_path.exists() else ""
    isolated = (
        sys.prefix != sys.base_prefix
        and not site.ENABLE_USER_SITE
        and "include-system-site-packages = false" in venv_config.lower()
    )
    external_sites = [
        path for path in sys.path
        if ("site-packages" in path or "dist-packages" in path)
        and not Path(path).resolve().is_relative_to(Path(sys.prefix).resolve())
    ]
    return {
        "sys_path": sys.path, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "venv_config": venv_config.strip(), "user_site_enabled": site.ENABLE_USER_SITE,
        "python_isolated_flag": bool(sys.flags.isolated),
        "inherited_PYTHONPATH": os.environ.get("PYTHONPATH"),
        "ros_paths": ros_paths, "ros_modules": ros_modules,
        "external_site_packages": external_sites,
        "isolated_venv": isolated,
        "ros_pollution": "none" if not ros_paths and not ros_modules else "detected",
    }


def verify_simulator(config_path):
    import mujoco
    import numpy as np
    import robosuite
    import yaml

    if not Path(robosuite.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Standalone workspace requires robosuite installed inside its venv, not an external editable checkout")
    if robosuite.__version__ != "1.5.2":
        raise RuntimeError(f"Expected robosuite 1.5.2, got {robosuite.__version__}")
    version = tuple(int(part) for part in mujoco.__version__.split(".")[:2])
    if not (3, 3) <= version < (3, 10):
        raise RuntimeError(f"Expected MuJoCo >=3.3.0,<3.10, got {mujoco.__version__}")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Environment YAML must be a mapping")
    if config.get("env_name") != "TwoArmLift" or config.get("robots") != ["Panda", "Panda"]:
        raise ValueError("Phase 0 requires TwoArmLift with two independent Panda robots")
    for name in ("has_renderer", "has_offscreen_renderer", "use_camera_obs", "ignore_done"):
        if config.get(name) is not False:
            raise ValueError(f"Phase 0 requires {name}: false")
    if config.get("controller") != "BASIC":
        raise ValueError("Phase 0 requires the BASIC composite controller")
    params = dict(config)
    controller_name = params.pop("controller")
    params["controller_configs"] = robosuite.load_composite_controller_config(controller=controller_name)
    env = robosuite.make(**params)
    try:
        reference = env.reset()

        def validate_observations(observations):
            if not observations or observations.keys() != reference.keys():
                raise ValueError("Empty or changed observation keys")
            for name, value in observations.items():
                array = np.asarray(value)
                if array.ndim > 1 or not np.isfinite(array).all():
                    raise ValueError(f"Invalid state observation: {name}")
                if array.shape != reference[name].shape or array.dtype != reference[name].dtype:
                    raise ValueError(f"Observation shape/dtype changed: {name}")

        validate_observations(reference)
        low, high = env.action_spec
        if low.ndim != 1 or low.shape != high.shape or low.size != env.action_dim:
            raise ValueError("Action dimensions disagree")
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
            raise ValueError("Invalid action bounds")
        robot_dims = [robot.action_dim for robot in env.robots]
        if sum(robot_dims) != low.size:
            raise ValueError("Per-robot action dimensions disagree with joint space")
        rng = np.random.default_rng(config["seed"])
        episode_return = 0.0
        success_ever = bool(env._check_success())
        for step in range(1, env.horizon + 1):
            action = rng.uniform(low, high)
            if action.shape != low.shape or np.any(action < low) or np.any(action > high):
                raise ValueError("Invalid random action")
            observations, reward, done, _ = env.step(action)
            validate_observations(observations)
            if np.asarray(reward).shape != () or not np.isfinite(reward):
                raise ValueError("Nonfinite or nonscalar reward")
            episode_return += float(reward)
            success_ever |= bool(env._check_success())
            if bool(done) != (step == env.horizon):
                raise ValueError(f"Unexpected done={done} at step {step}/{env.horizon}")
        if not np.isfinite(episode_return):
            raise ValueError("Nonfinite episode return")
        return {
            "status": "PASS", "config": config, "reset": "PASS", "step": "PASS",
            "steps": step, "episode_return": episode_return, "success_ever": success_ever,
            "done_at_horizon": bool(done), "action_dimension": int(low.size),
            "robot_action_dimensions": robot_dims,
            "action_low": low.tolist(), "action_high": high.tolist(),
            "observations": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                             for name, value in reference.items()},
        }
    finally:
        env.close()


def write_report(path, report):
    hardware = report["hardware"]
    packages = report["packages"]
    gpu_output = hardware["nvidia_smi"]["stdout"] or hardware["nvidia_smi"]["stderr"]
    lines = [
        "# Phase 0 — System Freeze", "",
        f"Measured at: {report['measured_at']}. Result: **{report['result']}**.", "",
        "This report is generated by `scripts/check_install.py` from the selected Python",
        "environment and workstation. PyTorch and CUDA toolkit selection remains unspecified",
        "until baseline RL. GPU presence is measured through NVIDIA's driver utility;",
        "it does not imply PyTorch CUDA availability.", "",
        "## Detected workstation", "",
        f"- OS: {hardware['os']}", f"- Kernel: `{hardware['kernel']}`",
        f"- CPU: {hardware['cpu']}",
        f"- RAM exposed by Linux: {hardware['ram_bytes']} bytes ({hardware['ram_bytes'] / 2**30:.2f} GiB)",
        "- GPU, driver, and memory (queried by `nvidia-smi`):", "", "```text", gpu_output, "```", "",
        "The CUDA column in the full `nvidia-smi` output indicates driver support, not an",
        "installed CUDA toolkit. This phase does not query, require, install, or pin a toolkit.", "",
        "## Research Python", "",
        f"- Python: {report['python_version']}", f"- Executable: `{report['python_executable']}`",
        f"- Isolated venv: {report['python_path']['isolated_venv']}",
        f"- ROS pollution in research Python path: **{report['python_path']['ros_pollution']}**",
    ]
    for name, info in packages.items():
        lines.append(f"- {name}: {info['version']}" + (f"; imported from `{info['path']}`" if info.get("path") else ""))
    lines += [
        f"- PyTorch selection: unspecified; detected installation: {report['pytorch']['version']}",
        f"- `torch.cuda.is_available()`: {report['pytorch']['cuda_available']}",
        "- CUDA toolkit selection: unspecified", "",
        "Effective `sys.path` and isolation checks:", "", "```json",
        json.dumps(report["python_path"], indent=2), "```", "",
        "Use `python -I` for installation/verification commands to ignore inherited ROS",
        "`PYTHONPATH`, user site-packages, and Python environment overrides. Shell ROS",
        "variables may still exist; the checks above inspect the effective Python path",
        "and discoverable ROS modules, rather than assuming an activated venv is clean.", "",
        "## Simulator verification", "", "```json",
        json.dumps(report["simulator"], indent=2), "```", "",
        "This is a state-only reset and full-horizon random-action check. No GUI/offscreen",
        "rendering is requested. Task success is separate from valid reset/step execution.", "",
        "## Source snapshot", "",
        f"Git SHA: `{report['git_sha'].get('description', report['git_sha']['stdout'])}`", "",
        "`git status --short --untracked-files=all` at verification time:", "", "```text",
        report["git_status"]["stdout"] or "(clean)", "```", "",
        "A recorded SHA does not include uncommitted edits shown above. Commit the reviewed",
        "Phase 0 artifacts separately when you want a single immutable source reference.", "",
        "## Installed package freeze", "",
        "Dependency consistency (`python -m pip check`):", "", "```text",
        report["pip_check"]["stdout"] or report["pip_check"]["stderr"], "```", "",
        "Captured with this interpreter's `python -m pip freeze`; no inherited Conda/system packages.",
        "robosuite is installed from its released package; only REL301m is editable.", "",
        "```text", report["pip_freeze"]["stdout"], "```", "",
        "Recreate dependencies from `experiments/phase0/requirements.freeze.txt` on the same",
        "Python/platform. The editable REL301m entry may refer to a local path or Git URL.",
        "To reuse the reviewed project checkout, replace that one editable REL301m entry",
        "with `-e .` and run from the REL301m root. Keep robosuite as a released package.", "",
        "## Checks and limitations", "",
        "```json", json.dumps({"errors": report["errors"], "notes": report["notes"]}, indent=2), "```", "",
        "The upstream BASIC config includes parts absent from Panda; robosuite skips them.",
        "Missing optional `robosuite_models` / Mink warnings do not block the two Panda",
        "robots or BASIC controller. No robosuite source modification is made by this phase.", "",
        "Sandbox restrictions can block dependency downloads and NVIDIA driver access.",
        "A sandbox-related failure requires host access before a full hardware PASS.",
        "This verification does not require renderer initialization. Such restrictions",
        "alone do not establish a simulator or driver fault.", "",
        "robosuite officially supports Linux/Python 3, and recommends an isolated environment",
        "with a Python 3.10 Conda example: [official installation documentation](https://robosuite.ai/docs/installation.html).", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env/two_arm_lift.yaml")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/reproducibility.md")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/phase0")
    args = parser.parse_args()
    cpu_line = next(line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    memory_line = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:"))
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "hardware": {
            "os": platform.freedesktop_os_release().get("PRETTY_NAME", platform.system()),
            "kernel": platform.release(), "cpu": cpu_line.split(":", 1)[1].strip(),
            "ram_bytes": int(memory_line.split()[1]) * 1024,
            "nvidia_smi": command_output([
                "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
            ]),
        },
        "python_path": check_python_path(), "packages": {},
        "pytorch": {"version": "unspecified (not installed)", "cuda_available": "unspecified (torch not installed)"},
        "git_sha": command_output(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "git_status": command_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT),
        "pip_freeze": command_output([sys.executable, "-I", "-m", "pip", "freeze"]),
        "pip_check": command_output([sys.executable, "-I", "-m", "pip", "check"]),
        "errors": [], "notes": [],
    }
    errors = report["errors"]
    if report["git_sha"]["returncode"] != 0:
        git_repo = command_output(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT)
        if git_repo["returncode"] == 0 and git_repo["stdout"] == "true":
            report["git_sha"]["description"] = "uncommitted (no HEAD yet)"
            report["notes"].append("New research repository has no commit yet; rerun after committing to record its SHA.")
    if sys.version_info[:2] != (3, 10):
        errors.append("Expected Python 3.10")
    path_info = report["python_path"]
    if not path_info["isolated_venv"] or path_info["external_site_packages"]:
        errors.append("Research Python must be an isolated venv without external site-packages")
    if path_info["ros_pollution"] != "none":
        errors.append("ROS pollution detected in research Python")
    if report["hardware"]["nvidia_smi"]["returncode"] != 0:
        errors.append("nvidia-smi could not record GPU/driver (sandbox may restrict driver access)")
    for name in ("robosuite", "mujoco", "numpy", "yaml"):
        try:
            module = importlib.import_module(name)
            report["packages"][name] = {"version": str(module.__version__), "path": module.__file__}
        except Exception as error:
            report["packages"][name] = {"version": "unavailable", "error": str(error)}
            errors.append(f"Cannot import {name}: {type(error).__name__}: {error}")
    if importlib.util.find_spec("torch") is not None:
        try:
            torch = importlib.import_module("torch")
            available = torch.cuda.is_available()
            report["pytorch"] = {"version": str(torch.__version__), "cuda_available": available,
                                 "build_cuda": torch.version.cuda,
                                 "gpu_name": torch.cuda.get_device_name(0) if available else None}
        except Exception as error:
            report["pytorch"] = {"version": "detected but import/check failed", "cuda_available": "unspecified"}
            report["notes"].append(f"Optional PyTorch check failed: {error}")
    for name in ("git_sha", "git_status", "pip_freeze", "pip_check"):
        if report[name]["returncode"] != 0:
            if name == "git_sha" and report[name].get("description") == "uncommitted (no HEAD yet)":
                continue
            errors.append(f"Cannot capture {name}: {report[name]['stderr']}")
    try:
        report["simulator"] = verify_simulator(args.config)
    except Exception as error:
        report["simulator"] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        errors.append(f"Simulator verification failed: {error}")
    report["result"] = "FAIL" if errors else "PASS"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "system_freeze.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "requirements.freeze.txt").write_text(report["pip_freeze"]["stdout"] + "\n", encoding="utf-8")
    write_report(args.report, report)
    rows = {
        "Python": report["python_version"],
        "robosuite": report["packages"]["robosuite"]["version"],
        "MuJoCo": report["packages"]["mujoco"]["version"],
        "PyTorch": report["pytorch"]["version"],
        "CUDA": report["pytorch"]["cuda_available"],
        "CUDA toolkit": "unspecified", "GPU / driver": report["hardware"]["nvidia_smi"]["stdout"] or "unavailable",
        "ROS pollution": path_info["ros_pollution"] + " in research Python path",
        "Simulator": report["simulator"]["status"], "RESULT": report["result"],
    }
    for label, value in rows.items():
        print(f"{label:<15} {value}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Report: {args.report.resolve()}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
