"""Privileged reach/grasp/lift expert, used only to collect demonstrations."""

import numpy as np
from scipy.spatial.transform import Rotation


class ScriptedLiftTeacher:
    phases = ('approach', 'descend', 'grasp', 'lift')

    def __init__(self, env):
        from rel301m.envs.contract import inspect_actions

        self.env = env
        self.base = env._env
        self.layout = inspect_actions(self.base)['robots']
        self.bounds = [env.action_specs[a] for a in env.agent_ids]
        self.handles = np.stack((self.base._handle0_xpos, self.base._handle1_xpos)).copy()
        pot_rotation = Rotation.from_quat(self.base._pot_quat).as_matrix()
        # Panda finger closing direction must be perpendicular to the handle bar.
        self.orientations = [pot_rotation @ np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]),
                             pot_rotation @ np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])]
        self.controllers = []
        self.arm_parts, self.grip_parts = [], []
        for robot, layout in zip(self.base.robots, self.layout):
            arms = [p for p in layout['parts'] if p['controller_type'] == 'OSC_POSE']
            grips = [p for p in layout['parts'] if p['controller_type'] == 'GRIP']
            if len(arms) != 1 or len(grips) != 1 or arms[0]['action_dim'] != 6 or grips[0]['action_dim'] != 1:
                raise ValueError('Script requires the measured OSC_POSE + one-DOF Panda gripper layout')
            controller = robot.part_controllers[arms[0]['name']]
            if controller.input_type != 'delta' or controller.input_ref_frame not in ('base', 'world'):
                raise ValueError('Script requires live delta OSC controller metadata')
            self.controllers.append(controller)
            self.arm_parts.append(arms[0])
            self.grip_parts.append(grips[0])
        self.phase = 'approach'

    def grasps(self):
        return np.asarray([self.base._check_grasp(robot.gripper[part['name']],
                           getattr(self.base.pot, f'handle{i}_geoms'))
                           for i, (robot, part) in enumerate(zip(self.base.robots, self.arm_parts))], dtype=bool)

    def act(self):
        if self.phase == 'approach' and all(
                np.linalg.norm(self.handles[i] + [0, 0, .07] - c.ref_pos) < .012 and
                np.linalg.norm(Rotation.from_matrix(self.orientations[i] @ c.ref_ori_mat.T).as_rotvec()) < .08
                for i, c in enumerate(self.controllers)):
            self.phase = 'descend'
        if self.phase == 'descend' and all(np.linalg.norm(self.handles[i] + [0, 0, -.004] - c.ref_pos) < .013
                                           for i, c in enumerate(self.controllers)):
            self.phase = 'grasp'
        if self.phase == 'grasp' and self.grasps().all():
            self.phase = 'lift'
        actions = []
        live_handles = (self.base._handle0_xpos, self.base._handle1_xpos)
        for i, controller in enumerate(self.controllers):
            if self.phase == 'approach':
                target, closing = self.handles[i] + [0, 0, .07], False
            elif self.phase in ('descend', 'grasp'):
                target, closing = self.handles[i] + [0, 0, -.004], self.phase == 'grasp'
            else:
                target = live_handles[i].copy()
                target[2] = min(target[2] + .012, self.handles[i, 2] + .22)
                closing = True
            frame = controller.origin_ori if controller.input_ref_frame == 'base' else np.eye(3)
            delta = frame.T @ (target - controller.ref_pos)
            rotation = Rotation.from_matrix(frame.T @ self.orientations[i] @ controller.ref_ori_mat.T @ frame).as_rotvec()
            motion = np.concatenate((.5 * delta, .5 * rotation))
            scale = (controller.output_max - controller.output_min) / (controller.input_max - controller.input_min)
            commands = (motion - (controller.output_max + controller.output_min) / 2) / scale
            commands += (controller.input_max + controller.input_min) / 2
            low, high = self.bounds[i]
            action = np.zeros_like(low)
            start, stop = self.arm_parts[i]['robot_slice']
            action[start:stop] = np.clip(commands, controller.input_min, controller.input_max)
            start, stop = self.grip_parts[i]['robot_slice']
            action[start:stop] = high[start:stop] if closing else low[start:stop]
            actions.append(action)
        return tuple(actions)
