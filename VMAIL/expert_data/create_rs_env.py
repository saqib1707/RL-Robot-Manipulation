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

camera_names = "robot0_eye_in_hand"
# camera_names = "frontview"
# camera_names = "agentview"
image_size = 256
horizon = 100

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
    horizon=horizon,
    use_object_obs=True,
    use_camera_obs=True,
    camera_depths=True,
    camera_heights=image_size, 
    camera_widths=image_size, 
    camera_names=camera_names, 
    use_tactile_obs=False,
    use_touch_obs=True
)
print("Environment created")

obs = env.reset()
index = 1
for k, v in obs.items():
    print(index, k, v.shape, v.min(), v.max())
    index += 1

def get_policy_action(obs=None):
    low, high = env.action_spec
    action = np.random.uniform(low, high)
    return action

camera_rgb = camera_names + "_image"
camera_depth = camera_names + "_depth"

frames_rgb = [obs[camera_rgb]]
depth_image = np.uint8(obs[camera_depth] * 255)
frames_depth = [depth_image]

done = False
episode_reward = 0
start_time = time.time()
while not done:
    action = get_policy_action()         # use observation to decide on an action
    # action = action_lst[i]
    # print(action)
    obs, reward, done, _ = env.step(action)    # play action
    episode_reward += reward

    # print(obs["agentview_image"].min(), obs["agentview_image"].max())
    # print(obs[camera_depth].min(), obs[camera_depth].max())
    frames_rgb.append(obs[camera_rgb])
    depth_image = np.uint8(obs[camera_depth] * 255)
    # print(np.min(depth_image), np.max(depth_image))
    frames_depth.append(depth_image)
    
    # obs1 = np.flip(obs[:image_size*image_size*3].reshape(image_size, image_size, 3), axis=0)
    # obs1 = obs1.astype(np.uint8)
    # frames_rgb.append(obs1)

    # obs2 = 255.0 - np.flip(obs[image_size*image_size*3:image_size*image_size*4].reshape(image_size, image_size), axis=0) * 255.0
    # obs2 = obs2.astype(np.uint8)
    # frames_depth.append(obs2)
    # i += 1

    proprio_state = np.concatenate((obs["robot0_joint_pos_cos"], obs["robot0_joint_pos_sin"], obs["robot0_joint_vel"], obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], obs["robot0_gripper_qvel"]))
    proprio_error = np.sum(proprio_state - obs["robot0_proprio-state"])

    object_state = np.concatenate((obs["cube_pos"], obs["cube_quat"], obs["cube_to_robot0_eef_pos"], obs["cube_to_robot0_eef_quat"], obs["robot0_eef_to_cube_yaw"]))
    object_error = np.sum(object_state - obs["object-state"])

    touch_error = np.sum(obs["robot0_touch"] - obs["robot0_touch-state"])

    # print(proprio_error, object_error, touch_error)

env.close()
print("rollout completed with return {}".format(episode_reward))
print(f"Spend {time.time() - start_time:.3f} s to run {horizon} steps")

path = "test_images/view_rgb.mp4"
imageio.mimsave(path, frames_rgb, fps=30)
path = "test_images/view_depth.mp4"
imageio.mimsave(path, frames_depth, fps=30)

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