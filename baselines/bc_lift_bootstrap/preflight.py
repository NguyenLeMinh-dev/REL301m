"""Verify existing dependencies only; never install/upgrade Torch or CUDA."""
from common import verify_stack, sha256
from env import environment_contract, make_lift_env
from pathlib import Path
import json
import torch


def main():
    versions = verify_stack()
    source = json.loads((Path(__file__).parent/"environment_source.json").read_text())
    assert sha256(Path(__file__).parent/"assisted_env.py") == source["sha256"], "Environment snapshot changed"
    env = make_lift_env(0)
    try:
        contract = environment_contract(env)
    finally:
        env.close()
    print(json.dumps({"packages": versions, "cuda_available": torch.cuda.is_available(), "environment": contract}, indent=2))
    print("INSTALL=PASS (existing dependencies verified; nothing installed/upgraded)")


if __name__ == "__main__":
    main()
