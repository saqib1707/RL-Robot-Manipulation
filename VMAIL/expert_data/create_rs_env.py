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
# env_name = "PickPlaceBread"
# with open("controller_config/robomimic.json", 'r') as f:
#     controller_config = json.load(f)

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")
# controller_config = None

# train_camera_names = "robot0_eye_in_hand"
train_camera_names = "frontview"
image_size = 84

# create an environment to visualize on-screen
env = suite.make(
    env_name=env_name,
    robots="Panda",
    gripper_types="default",
    controller_configs=controller_config,
    reward_shaping=True, 
    has_renderer=False,
    has_offscreen_renderer=True,
    control_freq=20,                    
    horizon=100,
    use_object_obs=True,
    use_camera_obs=True,
    camera_depths=True,
    camera_heights=image_size, 
    camera_widths=image_size, 
    camera_names=train_camera_names, 
    use_tactile_obs=False,
    use_touch_obs=True
)
print("Environment created")

obs = env.reset()
for k, v in obs.items():
    print(k, v.shape)


# model_data = np.load("../robosuite_task/log_20230704_164646/model_data/20230628T100123-5c202889259044b1be711866ddb7b3ce-100.npz")
# policy_data = np.load("../robosuite_task/log_20230704_164646/policy_data/20230704T165324-426efc5479b844b8908d70737512d46f-101.npz")
# print(model_data, policy_data)
# print(model_data.files, policy_data.files)

# print("model data")
# for item in model_data.files:
#     print(item, model_data[item].shape)

# print("Policy data")
# for item in policy_data.files:
#     print(item, policy_data[item].shape)