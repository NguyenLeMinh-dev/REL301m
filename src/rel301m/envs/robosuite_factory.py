"""Build the initial environment from YAML without changing robosuite."""

from copy import deepcopy
from pathlib import Path

import robosuite
import yaml


# The documented workflow installs REL301m editable from this repository.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/env/two_arm_lift.yaml"


def load_env_config(config_path=DEFAULT_CONFIG):
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Environment YAML must be a mapping")
    required = {
        "env_name", "robots", "env_configuration", "controller", "has_renderer",
        "has_offscreen_renderer", "use_camera_obs", "use_object_obs",
        "reward_shaping", "reward_scale", "control_freq", "horizon", "ignore_done", "seed",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing environment parameters: {sorted(missing)}")
    if config["env_name"] != "TwoArmLift" or config["robots"] != ["Panda", "Panda"]:
        raise ValueError("This contract requires TwoArmLift with two independent Panda robots")
    if config["env_configuration"] != "opposed" or config["controller"] != "BASIC":
        raise ValueError("This contract requires opposed robots and the BASIC composite controller")
    for name in ("has_renderer", "has_offscreen_renderer", "use_camera_obs", "ignore_done"):
        if config[name] is not False:
            raise ValueError(f"State-only horizon contract requires {name}: false")
    if config["use_object_obs"] is not True:
        raise ValueError("Object observations must be enabled")
    if not isinstance(config["horizon"], int) or isinstance(config["horizon"], bool) or config["horizon"] <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(config["control_freq"], (int, float)) or config["control_freq"] <= 0:
        raise ValueError("control_freq must be positive")
    return config


def make_two_arm_lift(config_path=DEFAULT_CONFIG, *, seed=None):
    if robosuite.__version__ != "1.5.2":
        raise RuntimeError(f"Expected robosuite 1.5.2, got {robosuite.__version__} from {robosuite.__file__}")
    config = load_env_config(config_path)
    controller = robosuite.load_composite_controller_config(controller=config.pop("controller"))
    # robosuite initializes/mutates controller settings independently for each robot.
    config["controller_configs"] = [deepcopy(controller) for _ in config["robots"]]
    if seed is not None:
        config["seed"] = seed
    return robosuite.make(**config)
