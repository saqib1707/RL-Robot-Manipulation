import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import utils

import robosuite as suite
from robosuite.wrappers import GymWrapper

np.set_printoptions(precision=3)

class BaseHumanPolicy:
    def __init__(self, env):
        self.env = env
        self.pos_tol = 2e-2     # positional tolerance
        self.rot_tol = 5e-2     # rotational tolerance in radian

        # Check environment name, controller type
        assert self.env.robot_configs[0]['controller_config']['type'] == 'OSC_POSE'

        # direction / rotation symbol to index
        self.dir2ind = {'x': 0, 'y': 1, 'z': 2}
        self.rot2ind = {'ax': 3, 'ay': 4, 'az': 5}

    def _setup_stages(self):
        raise NotImplementedError()

    def reset(self):
        self.completed = {stage: False for stage in self.stages}

    def normalize_angle(self, x):
        # Normalize angle between -pi to pi
        return np.arctan2(np.sin(x), np.cos(x))

    def move(self, ds, action, direction=None, tol=None):
        """
        Translational motion of ds, in x / y / z direction
        action is modified in-place
        """
        completed = True
        ind = self.dir2ind[direction] 
        tol = tol if tol is not None else self.pos_tol
        if np.abs(ds) > tol:
            action[ind] = -1 if ds < 0 else 1
            completed = False
        return action, completed

    def rotate(self, da, action, direction=None, tol=None):
        """
        Rotational motion of da (in radian), in ax / ay / az direction
        action is modified in-place
        In robosuite OSC_POSE controller, rotation axes are taken relative to the global 
        world coordinate frame, c.f. https://robosuite.ai/docs/modules/controllers.html
        """
        completed = True
        ind = self.rot2ind[direction]
        tol = tol if tol is not None else self.rot_tol
        if np.abs(da) > tol:
            action[ind] = -0.15 if da < 0 else 0.15
            completed = False
        return action, completed

    def predict(self, obs):
        raise NotImplementedError()


class ReachPolicy(BaseHumanPolicy):
    def __init__(self, env):
        super().__init__(env)

        # Assert observations used for control exist in the env
        self.obs_names = [
            'target_to_robot0_eef_pos',
        ]

        env_obs_names = env.observation_names
        for name in self.obs_names:
            assert name in env_obs_names, f"{name} not in env {env}"

        self._setup_stages()

    def _setup_stages(self):
        self.stages = [
            'move_x', 
            'move_y', 
            'move_z', 
        ]   

    def move_x(self, obs, action):
        """Negate the obs since we are moving eef towards object"""
        rel_pos = -obs[f'target_to_robot0_eef_pos']
        return self.move(rel_pos[0], action, direction='x')

    def move_y(self, obs, action):
        rel_pos = -obs[f'target_to_robot0_eef_pos']
        return self.move(rel_pos[1], action, direction='y')

    def move_z(self, obs, action):
        rel_pos = -obs[f'target_to_robot0_eef_pos']
        return self.move(rel_pos[2], action, direction='z')

    def predict(self, obs):
        action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for curr_stage, completed in self.completed.items():
            if not completed: break

        action, completed = getattr(self, curr_stage)(obs, action)
        self.completed[curr_stage] = completed
        return action, {'stage': curr_stage, 'completed': completed}


class LiftPolicy(BaseHumanPolicy):
    def __init__(self, env):
        super().__init__(env)
        self.obj_name = 'cube'

        # Assert observations used for control exist in the env
        self.obs_names = [
            'robot0_eef_pos',
            'robot0_eef_quat',
            f'{self.obj_name}_to_robot0_eef_pos',
            f'robot0_eef_to_{self.obj_name}_yaw',
            'robot0_touch',
        ]

        env_obs_names = env.observation_names
        for name in self.obs_names:
            assert name in env_obs_names, f"{name} not in env {env}"

        self._setup_stages()

    def _setup_stages(self):
        self.stages = [
            'rotate_z', 
            'move_x', 
            'move_y', 
            'lower_gripper', 
            'close_gripper', 
            'lift_gripper',
        ]

    def rotate_z(self, obs, action):
        rel_rot = obs[f'robot0_eef_to_{self.obj_name}_yaw']
        return self.rotate(rel_rot, action, direction='az')

    def move_x(self, obs, action):
        """Negate the obs since we are moving eef towards object"""
        rel_pos = -obs[f'{self.obj_name}_to_robot0_eef_pos']
        return self.move(rel_pos[0], action, direction='x')

    def move_y(self, obs, action):
        rel_pos = -obs[f'{self.obj_name}_to_robot0_eef_pos']
        return self.move(rel_pos[1], action, direction='y')

    def lower_gripper(self, obs, action):
        rel_pos = -obs[f'{self.obj_name}_to_robot0_eef_pos']
        return self.move(rel_pos[2], action, direction='z')

    def close_gripper(self, obs, action):
        completed = True
        touch_obs = obs['robot0_touch']
        action[-1] = 1
        if not np.all(touch_obs > 0.9):
            completed = False
        return action, completed

    def lift_gripper(self, obs, action):
        target_z = 0.9
        eef_pos = obs['robot0_eef_pos']
        return self.move(target_z - eef_pos[2], action, direction='z')

    def predict(self, obs):
        action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for curr_stage, completed in self.completed.items():
            if curr_stage == 'close_gripper' and completed:
                action[-1] = 1
            if not completed:
                break

        action, completed = getattr(self, curr_stage)(obs, action)
        self.completed[curr_stage] = completed
        return action, {'stage': curr_stage, 'completed': completed}


class PickPlacePolicy(BaseHumanPolicy):
    def __init__(self, env):
        super().__init__(env)

        # Currently only supports single object in task
        assert self.env.single_object_mode in [1, 2]
        self.obj_names = ["Milk", "Bread", "Cereal", "Can"]
        self.obj_name = self.obj_names[self.env.object_id]

        if self.obj_name == 'Milk':
            self.obj_height = 0.083
        elif self.obj_name == 'Bread':
            self.obj_height = 0.04
        elif self.obj_name == 'Cereal':
            self.obj_height = 0.1
        elif self.obj_name == 'Can':
            self.obj_height = 0.06

        self.obs_names = [
            'robot0_eef_pos',
            f'{self.obj_name}_to_robot0_eef_pos',
            f'robot0_eef_to_{self.obj_name}_yaw',
            f'{self.obj_name}_to_{self.obj_name}_bin_pos',
            'robot0_touch',
        ]

        env_obs_names = env.observation_names
        for name in self.obs_names:
            assert name in env_obs_names, f"{name} not in env {env}"

        self._setup_stages()

    def _setup_stages(self):
        self.stages = [
            'rotate_z',
            'move_to_obj_x',
            'move_to_obj_y',
            'move_to_obj_z',
            'close_gripper',
            'lift_gripper',
            'move_to_bin_y',
            'move_to_bin_x',
            'move_to_bin_z',
            'drop_obj',
        ]

    def rotate_z(self, obs, action):
        rel_rot = obs[f'robot0_eef_to_{self.obj_name}_yaw']
        return self.rotate(rel_rot, action, direction='az')

    def move_to_obj_x(self, obs, action):
        rel_pos = -obs[f'{self.obj_name}_to_robot0_eef_pos']
        return self.move(rel_pos[0], action, direction='x')

    def move_to_obj_y(self, obs, action):
        rel_pos = -obs[f'{self.obj_name}_to_robot0_eef_pos']
        return self.move(rel_pos[1], action, direction='y')

    def move_to_obj_z(self, obs, action):
        rel_pos = -obs[f'{self.obj_name}_to_robot0_eef_pos']
        return self.move(rel_pos[2]+self.obj_height-0.04, action, direction='z')

    def close_gripper(self, obs, action):
        completed = True
        touch_obs = obs['robot0_touch']
        action[-1] = 1
        if not np.all(touch_obs > 0.9):
            completed = False
        return action, completed

    def lift_gripper(self, obs, action):
        target_z = 1
        eef_pos = obs['robot0_eef_pos']
        rel_pos = target_z - eef_pos[2] + self.obj_height
        return self.move(rel_pos, action, direction='z')

    def move_to_bin_x(self, obs, action):
        """
        Increase tolerance since the arm is hard to stretch out
        """
        rel_pos = obs[f'{self.obj_name}_to_{self.obj_name}_bin_pos']
        if self.obj_name in ['Cereal', 'Can']:
            tol = 0.03
        else:
            tol = self.pos_tol
        return self.move(rel_pos[0], action, direction='x', tol=tol)
    
    def move_to_bin_y(self, obs, action):
        rel_pos = obs[f'{self.obj_name}_to_{self.obj_name}_bin_pos']
        if self.obj_name in ['Cereal', 'Can']:
            tol = 0.03
        else:
            tol = self.pos_tol
        return self.move(rel_pos[1], action, direction='y', tol=tol)

    def move_to_bin_z(self, obs, action):
        rel_pos = obs[f'{self.obj_name}_to_{self.obj_name}_bin_pos']
        if self.obj_name in ['Cereal', 'Can']:
            tol = 0.03
        else:
            tol = self.pos_tol
        return self.move(rel_pos[2]+self.obj_height+0.1, action, direction='z', tol=tol)

    def drop_obj(self, obs, action):
        completed = True
        touch_obs = obs['robot0_touch']
        if not np.all(touch_obs < 0.05):
            action[-1] = -1
            completed = False
        return action, completed

    def predict(self, obs):
        action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for curr_stage, completed in self.completed.items():
            if curr_stage == 'close_gripper' and completed:
                action[-1] = 1
            if curr_stage == 'drop_obj' and completed:
                action[-1] = -1
            if not completed: break

        action, completed = getattr(self, curr_stage)(obs, action)
        self.completed[curr_stage] = completed
        return action, {curr_stage: completed}


def main():
    # Use default OSC_POSE controller
    
    # env = utils.make_robosuite_env("PickPlaceCereal", render=True)
    # policy = PickPlacePolicy(env)

    env = utils.make_robosuite_env("Lift", robots="Sawyer", render=True)
    policy = LiftPolicy(env)

    for i in range(10):
        obs = env.reset()

        policy.reset()
        done = False
        while not done:
            action, completed = policy.predict(obs)
            obs, rew, done, info = env.step(action)
            # print(rew, obs['cube_to_robot0_eef_pos'])
            # print(obs[f'{policy.obj_name}_to_{policy.obj_name}_bin_pos'])
            env.render()
            if env._check_success():
                print("Episode solved successfully")
                break
    env.close()


if __name__ == '__main__':
    main()