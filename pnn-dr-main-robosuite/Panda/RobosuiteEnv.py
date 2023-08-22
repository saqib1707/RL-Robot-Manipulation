import os
import mujoco_py
# import mujoco
import numpy as np
import gym
from gym.utils import seeding

DEFAULT_SIZE = 64


class RobosuiteEnv:
  def __init__(self, name, horizon=1000, size=(64, 64), camview="agentview", use_depth_obs=False, use_object_obs=True, use_tactile_obs=False, use_touch_obs=True):
    domain, task = name.split('_', 1)
    self._size = size
    self._camview_rgb = camview+'_image'
    self._camview_depth = camview+'_depth'
    self._use_object_obs = use_object_obs
    self._use_depth_obs = use_depth_obs
    self._use_tactile_obs = use_tactile_obs
    self._use_touch_obs = use_touch_obs

    if isinstance(domain, str):
      import robosuite as suite
      from robosuite.wrappers import GymWrapper
      from robosuite.controllers import load_controller_config

      # load default controller parameters for Operational Space Control (OSC)
      controller_config = load_controller_config(default_controller="OSC_POSE")

      # create a robosuite environment to visualize on-screen
      self._env = suite.make(
          env_name=domain, 
          robots="Panda", 
          gripper_types="default",
          controller_configs=controller_config,
          reward_shaping=True, 
          has_renderer=False, 
          has_offscreen_renderer=True, 
          use_camera_obs=True, 
          use_object_obs=self._use_object_obs,
          camera_depths=self._use_depth_obs,
          control_freq=20, 
          horizon=horizon, 
          camera_names=camview, 
          camera_heights=self._size[0], 
          camera_widths=self._size[1], 
          use_tactile_obs=self._use_tactile_obs,
          use_touch_obs=self._use_touch_obs
      )

  @property
  def observation_space(self):
    spaces = {}
    # print("Observation space keys:", self._env.observation_spec.keys())
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
    
    spaces[self._camview_rgb] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec
    shp = spec[0].shape
    return gym.spaces.Box(spec[0].min(), spec[1].max(), (shp[0],), dtype=np.float32)

  def step(self, action):
    next_obs, reward, done, _ = self._env.step(action)
    return next_obs, reward, done

  def reset(self):
    obs = self._env.reset()
    return obs
