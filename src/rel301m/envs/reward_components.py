"""Diagnostic decomposition of the pinned TwoArmLift reward; never replace reward."""
import numpy as np
import robosuite
from robosuite.utils import transform_utils as T


REWARD_COMPONENT_FIELDS = ['success_reward', 'lift_reward', 'reach_reward_0', 'reach_reward_1',
                           'grasp_reward_0', 'grasp_reward_1']


def reward_components(base, observed_reward):
    if robosuite.__version__ != '1.5.2' or type(base).__name__ != 'TwoArmLift' or base.env_configuration != 'opposed':
        raise ValueError('Reward diagnostics require the pinned opposed TwoArmLift contract')
    components = dict.fromkeys(REWARD_COMPONENT_FIELDS, 0.0)
    upright = int((T.quat2mat(base._pot_quat) @ np.array([0., 0., 1.]))[2] >= np.cos(np.pi / 6))
    if base._check_success():
        components['success_reward'] = 3.0 * upright
    elif base.reward_shaping:
        bottom = base.sim.data.site_xpos[base.pot_center_id][2] - base.pot.top_offset[2]
        elevation = bottom - base.sim.data.site_xpos[base.table_top_id][2]
        components['lift_reward'] = 10.0 * upright * min(max(elevation - .05, 0), .15)
        relatives = (base._gripper0_to_handle0, base._gripper1_to_handle1)
        handles = (base.pot.handle0_geoms, base.pot.handle1_geoms)
        for i in range(2):
            components[f'reach_reward_{i}'] = .5 * (1 - np.tanh(10 * np.linalg.norm(relatives[i])))
            components[f'grasp_reward_{i}'] = .25 * bool(base._check_grasp(base.robots[i].gripper, handles[i]))
    scale = base.reward_scale / 3 if base.reward_scale is not None else 1.0
    components = {key: float(value * scale) for key, value in components.items()}
    total = sum(components.values())
    if not np.isfinite([observed_reward, total]).all() or not np.isclose(total, observed_reward, atol=1e-8, rtol=1e-6):
        raise RuntimeError('Reward decomposition differs from upstream reward')
    return dict(components, tilt_valid=upright, total_reward=total)
