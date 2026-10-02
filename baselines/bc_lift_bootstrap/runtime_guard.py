"""Require the existing Phase-3 interpreter and exclude user/ROS packages."""
import os
import site
import sys
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[2]


def enforce_phase3_python():
    required = project_root() / ".venv-phase3"
    if Path(sys.prefix).resolve() != required.resolve():
        raise SystemExit(f"Wrong Python environment: {sys.executable}. Required: {required / 'bin/python'}")
    user_site = Path(site.getusersitepackages()).resolve()
    if site.ENABLE_USER_SITE or any(Path(p).resolve() == user_site for p in sys.path if p):
        raise SystemExit("User site is enabled; use the shell wrappers with PYTHONNOUSERSITE=1")
    if any("/opt/ros/" in p or "/ros2/" in p for p in sys.path):
        raise SystemExit("ROS packages in research Python path; unset PYTHONPATH before launch")
    os.environ["PYTHONNOUSERSITE"] = "1"
