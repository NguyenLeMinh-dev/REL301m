"""Try the saved failure pose in an unchanged environment, without running RL."""
import argparse
import json
from pathlib import Path
import sys

from rel301m.utils.console import install_panda_warning_filter
install_panda_warning_filter()
import mujoco
import numpy as np
import yaml
from rel301m.envs.robosuite_factory import make_two_arm_lift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    failure = json.loads((args.run_dir / 'failure.json').read_text())
    config = yaml.safe_load((args.run_dir / 'resolved_config.yaml').read_text())
    base = make_two_arm_lift(args.run_dir / 'env.yaml', seed=config['seed'])
    result = dict(mujoco=mujoco.__version__, original_error=failure['error'], exact_resume=False,
                  snapshot=str(args.run_dir / 'failure_state.npz'), collision_fix_verified=False)
    try:
        base.reset()
        model, data = base.sim.model._model, base.sim.data._data
        with np.load(args.run_dir / 'failure_state.npz', allow_pickle=False) as state:
            if state['qpos'].shape != data.qpos.shape or state['qvel'].shape != data.qvel.shape:
                raise ValueError('Snapshot dimensions differ from the compiled model')
            if not np.isfinite(state['qpos']).all() or not np.isfinite(state['qvel']).all():
                raise ValueError('Snapshot contains nonfinite simulator state')
            data.qpos[:] = state['qpos']
            data.qvel[:] = state['qvel']
        try:
            mujoco.mj_forward(model, data)
            result.update(status='not_reproduced', contacts=int(data.ncon))
        except mujoco.FatalError as error:
            result.update(status='reproduced', error=str(error))
            # Geom names/types help identify the failing pair without changing collisions.
            import re
            pair = re.search(r'geom pair \((\d+), (\d+)\)', str(error))
            if pair:
                geoms = [int(g) for g in pair.groups()]
                result['geoms'] = [dict(id=g, name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
                                        type=int(model.geom_type[g])) for g in geoms]
    finally:
        base.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    # The diagnostic completed; status reports whether the historical failure reproduced.
    return 0


if __name__ == '__main__':
    sys.exit(main())
