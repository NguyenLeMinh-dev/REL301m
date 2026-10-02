"""Inspect live observations, observables, and public composite-controller metadata."""

from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
import robosuite


SCHEMA_VERSION = 1


def validate_observations(observations, reference=None):
    if not isinstance(observations, dict) or not observations:
        raise ValueError("Expected a nonempty observation dictionary")
    if reference is not None and list(observations) != list(reference):
        raise ValueError("Observation keys/order changed during the episode")
    for name, value in observations.items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"Observation {name!r} contains nonnumeric values or NaN/Inf")
        if array.ndim > 1:
            raise ValueError(f"Observation {name!r} is not low-dimensional state")
        if reference is not None:
            expected = np.asarray(reference[name])
            if array.shape != expected.shape or array.dtype != expected.dtype:
                raise ValueError(f"Observation shape/dtype changed for {name!r}")


def action_bounds(env):
    low, high = (np.asarray(value) for value in env.action_spec)
    if low.ndim != 1 or low.size == 0 or low.shape != high.shape or low.size != env.action_dim:
        raise ValueError("action_spec vectors disagree with action_dim")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        raise ValueError("Invalid action bounds")
    return low, high


def validate_action(action, low, high):
    action = np.asarray(action)
    if action.shape != low.shape or not np.isfinite(action).all():
        raise ValueError("Action has incorrect dimensions or NaN/Inf")
    if np.any(action < low) or np.any(action > high):
        raise ValueError("Action is outside action_spec bounds")


def validate_reward(reward):
    array = np.asarray(reward)
    if array.shape != () or not np.isfinite(array).all():
        raise ValueError("Reward must be a finite scalar")


def observation_semantics(name):
    object_semantics = {
        "pot_pos": ("shared_object", "Pot body position in MuJoCo world coordinates (m)."),
        "pot_quat": ("shared_object", "Pot body orientation in world coordinates, quaternion xyzw."),
        "handle0_xpos": ("shared_object", "Handle 0 site position in world coordinates (m)."),
        "handle1_xpos": ("shared_object", "Handle 1 site position in world coordinates (m)."),
        "gripper0_to_handle0": ("robot_0_object_relation", "Handle 0 minus robot 0 EEF site position, world-frame vector (m)."),
        "gripper1_to_handle1": ("robot_1_object_relation", "Handle 1 minus robot 1 EEF site position, world-frame vector (m)."),
        "object-state": ("joint_object_aggregate", "Concatenation of active/enabled object sensors; includes both robots' relative-handle vectors."),
    }
    if name in object_semantics:
        group, description = object_semantics[name]
        return {"information_group": group, "description": description}
    suffix_semantics = {
        "joint_pos": "Arm joint positions (rad).",
        "joint_pos_cos": "Cosine of arm joint positions (unitless).",
        "joint_pos_sin": "Sine of arm joint positions (unitless).",
        "joint_vel": "Arm joint velocities (rad/s).",
        "joint_acc": "Arm joint accelerations (rad/s^2), simulator-derived.",
        "eef_pos": "EEF site position in world coordinates (m).",
        "eef_quat": "Legacy EEF BODY quaternion in world coordinates, xyzw; not the EEF site orientation.",
        "eef_quat_site": "EEF SITE quaternion in world coordinates, xyzw; consistent with eef_pos.",
        "gripper_qpos": "Gripper finger joint positions (m for Panda's prismatic finger joints).",
        "gripper_qvel": "Gripper finger joint velocities (m/s).",
        "proprio-state": "Upstream concatenation of active/enabled proprioceptive sensors, including redundant encodings.",
    }
    for index in range(2):
        prefix = f"robot{index}_"
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            return {"information_group": f"robot_{index}_local", "description": suffix_semantics.get(suffix, "Unclassified robot sensor; inspect its upstream sensor before choosing policy access.")}
    return {"information_group": "unclassified", "description": "Unclassified upstream sensor; no policy access is assumed."}


def inspect_observations(env, observations):
    validate_observations(observations)
    # Public properties expose names; registry access is needed for per-sensor modality/status.
    enabled = sorted(env.enabled_observables)
    active = sorted(env.active_observables)
    registry = {
        name: {"modality": sensor.modality, "enabled": bool(sensor.is_enabled()),
               "active": bool(sensor.is_active()), "returned": name in observations}
        for name, sensor in env._observables.items()
    }
    specs = {}
    for name, value in observations.items():
        array = np.asarray(value)
        spec = {"shape": list(array.shape), "dtype": str(array.dtype), "size": int(array.size),
                **observation_semantics(name)}
        if name in registry:
            spec.update(registry[name])
            spec["aggregation_of"] = []
        else:
            components = [key for key in observations if key in registry and registry[key]["modality"] + "-state" == name]
            if not components:
                raise ValueError(f"Returned observation {name!r} has no registered sensor or aggregate provenance")
            spec.update({"modality": name.removesuffix("-state"), "enabled": None, "active": None,
                         "returned": True, "aggregation_of": components})
        specs[name] = spec
    return {"schema_version": SCHEMA_VERSION, "returned_keys": list(observations),
            "enabled_observables": enabled, "active_observables": active,
            "observables": registry, "observations": specs}


def inspect_actions(env):
    low, high = action_bounds(env)
    robots = []
    offset = 0
    for index, robot in enumerate(env.robots):
        controller = robot.composite_controller
        # These public methods expose the actual body-part order and robot-local slices.
        info = controller.get_action_info_dict()
        index_info, dimension_info = controller.get_action_info()
        robot_low, robot_high = robot.action_limits
        stop = offset + robot.action_dim
        if tuple(info["Action Dimension"]) != robot_low.shape:
            raise ValueError("Controller action information disagrees with robot action limits")
        if not np.array_equal(low[offset:stop], robot_low) or not np.array_equal(high[offset:stop], robot_high):
            raise ValueError("Per-robot limits disagree with the joint action_spec")
        parts = []
        next_start = 0
        for name, indices in info.items():
            if name == "Action Dimension":
                continue
            start, end = map(int, indices)
            if start != next_start or end < start or end > robot.action_dim:
                raise ValueError("Noncontiguous or invalid controller action partition")
            part = controller.part_controllers[name]
            part_spec = {
                "name": name, "controller_type": controller.part_controller_config[name]["type"],
                "controller_class": type(part).__name__, "action_dim": end - start,
                "robot_slice": [start, end], "joint_slice": [offset + start, offset + end],
                "low": np.asarray(robot_low[start:end]).tolist(),
                "high": np.asarray(robot_high[start:end]).tolist(),
            }
            for attribute in ("input_type", "input_ref_frame", "impedance_mode"):
                if hasattr(part, attribute):
                    part_spec[attribute] = getattr(part, attribute)
            parts.append(part_spec)
            next_start = end
        if next_start != robot.action_dim:
            raise ValueError("Body-part action slices do not cover the robot action")
        robots.append({
            "robot_id": index, "robot_name": robot.robot_model.name,
            "naming_prefix": robot.robot_model.naming_prefix, "action_dim": robot.action_dim,
            "joint_slice": [offset, stop], "composite_controller": controller.name,
            "composite_controller_class": type(controller).__name__,
            "action_index_info": index_info, "action_dimension_info": dimension_info,
            "action_info": {key: list(value) for key, value in info.items()}, "parts": parts,
        })
        offset = stop
    if offset != env.action_dim:
        raise ValueError("Robot slices do not cover the joint action")
    return {"schema_version": SCHEMA_VERSION, "action_dim": int(env.action_dim),
            "shape": list(low.shape), "dtype": str(low.dtype), "low": low.tolist(),
            "high": high.tolist(), "slice_convention": "start inclusive, stop exclusive", "robots": robots}


def source_evidence(function):
    lines, start = inspect.getsourcelines(function)
    source = "".join(lines)
    return {"path": inspect.getsourcefile(function), "start_line": start,
            "sha256": hashlib.sha256(source.encode()).hexdigest(), "source": source}


def build_environment_spec(env, observations, config):
    observation_spec = inspect_observations(env, observations)
    action_spec = inspect_actions(env)
    env_spec = {
        "schema_version": SCHEMA_VERSION, "measured_at": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "robosuite": robosuite.__version__,
                     "mujoco": mujoco.__version__, "numpy": np.__version__},
        "config": config, "environment_class": type(env).__name__,
        "robot_count": len(env.robots), "control_freq": env.control_freq, "horizon": env.horizon,
        "observation_spec_file": "observation_spec.json", "action_spec_file": "action_spec.json",
        "reward_contract": {
            "reward_shaping": env.reward_shaping, "reward_scale": env.reward_scale,
            "success_method": "env._check_success()", "success_height_margin_m": 0.10,
            "success_requires_tilt_check": False,
            "reward_tilt_gate_degrees": 30, "reward_normalization_divisor": 3.0,
            "raw_reward_when_success": "3 * tilt_gate",
            "raw_shaped_reward_when_not_success": "10 * tilt_gate * clip(elevation - 0.05, 0, 0.15) + 0.25 * (grasp0 + grasp1) + 0.5 * (1 - tanh(10 * distance0)) + 0.5 * (1 - tanh(10 * distance1))",
            "termination": "done = (timestep >= horizon) and not ignore_done; success does not terminate",
            "success_is_not_return_threshold": True,
        },
        "episode_metrics": {
            "episode_return": "Sum of step rewards; independent of success.",
            "ever_success": "True if upstream task success is seen at reset or after any step.",
            "first_success_step": "0 for success at reset, otherwise first successful 1-based step; null if never successful.",
            "final_success": "Upstream task success after the final step.",
            "episode_length": "Number of control steps actually executed.",
        },
        "source_evidence": {
            "reward": source_evidence(type(env).reward),
            "success": source_evidence(type(env)._check_success),
            "termination": source_evidence(type(env)._post_action),
        },
    }
    return env_spec, observation_spec, action_spec


def write_specs(output_dir, env_spec, observation_spec, action_spec):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, spec in (("env_spec.json", env_spec), ("observation_spec.json", observation_spec), ("action_spec.json", action_spec)):
        (output_dir / filename).write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n", encoding="utf-8")
