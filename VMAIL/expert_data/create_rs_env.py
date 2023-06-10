import numpy as np
import torch
import argparse
import os
import sys
import random
import time
import math
import matplotlib.pyplot as plt
import imageio
import pprint

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper
# from gym_wrapper import GymWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("Number of GPU devices:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    print("Device:", device)

env_name = "Lift"
# with open("controller_config/robomimic.json", 'r') as f:
#     controller_config = json.load(f)

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")
# controller_config = None

train_camera_names = "robot0_eye_in_hand"
# train_camera_names = "agentview"
horizon = 100
image_size = 84
use_camera_depth = False
use_tactile_obs = False
use_touch_obs = True

# create an environment to visualize on-screen
env = suite.make(
    env_name=env_name,
    robots="Panda",                # load a Sawyer robot and a Panda robot
    gripper_types="default",         # use default grippers for robot arm
    controller_configs=controller_config,      # each arm is controlled using OSC
    reward_shaping=True, 
    has_renderer=False,                         # on-screen rendering
    has_offscreen_renderer=True,              # no off-screen rendering
    control_freq=20,                    
    horizon=horizon,             # each episode terminates after 200 steps
    use_object_obs=True,         # no observations needed
    use_camera_obs=True,         # don't provide camera/image observations to agent
    camera_depths=use_camera_depth,
    camera_heights=image_size, 
    camera_widths=image_size, 
    camera_names=train_camera_names, 
    use_tactile_obs=use_tactile_obs,
    use_touch_obs=use_touch_obs
)
print("Environment created")

obs = env.reset()
for k, v in obs.items():
    print(k, v.shape)