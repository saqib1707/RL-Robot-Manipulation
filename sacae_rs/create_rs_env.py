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
# from robosuite.wrappers import GymWrapper
from gym_wrapper import GymWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Number of GPU devices:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
    # print('Allocated memory:', round(torch.cuda.memory_allocated(0)/1024**3, 3), 'GB')
    # print('Cached memory:   ', round(torch.cuda.memory_reserved(0)/1024**3, 3), 'GB')
else:
    print("Device:", device)

env_name = "Lift"
# with open("controller_config/robomimic.json", 'r') as f:
#     controller_config = json.load(f)

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")
# controller_config = None
train_camera_names = ["agentview", "robot0_eye_in_hand"]
horizon = 250
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
    # env_configuration="single-arm-opposed",    # arms face each other
    reward_shaping=True, 
    has_renderer=False,                         # on-screen rendering
    # render_camera="frontview",                 # visualize the frontview camera
    has_offscreen_renderer=True,              # no off-screen rendering
    control_freq=20,                    
    horizon=horizon,             # each episode terminates after 200 steps
    use_object_obs=False,        # no observations needed
    use_camera_obs=True,         # don't provide camera/image observations to agent
    camera_depths=use_camera_depth,
    camera_heights=image_size, 
    camera_widths=image_size, 
    camera_names=train_camera_names, 
    use_tactile_obs=use_tactile_obs,
    use_touch_obs=use_touch_obs
)

print(env.camera_names)
print("Robot type:", env.robots[0], len(env.robots))
print("Environment created")
print("Use camera depths:", env.camera_depths[0])

obs = env.reset()
for k, v in obs.items():
    print(k, v.shape)

def get_policy_action(obs=None):
    low, high = env.action_spec
    action = np.random.uniform(low, high)
    return action

env2 = GymWrapper(env)

frames_rgb = []
frames_depth = []
obs = env2.reset()
# print(obs.shape, obs[:image_size*image_size*3].min(), obs[:image_size*image_size*3].max(), obs[image_size*image_size*3:].min(), obs[image_size*image_size*3:].max())
# print(obs.shape, obs[:image_size*image_size*3].min(), obs[:image_size*image_size*3].max(), obs[image_size*image_size*3:image_size*image_size*4].min(), obs[image_size*image_size*3:image_size*image_size*4].max(), obs[image_size*image_size*4:].min(), obs[image_size*image_size*4:].max())

rgb_dim = image_size * image_size * 3
depth_dim = image_size * image_size if env.camera_depths[0] == True else 0

rgb_obs = np.flip(obs[:rgb_dim].reshape(image_size, image_size, 3), axis=0)
if depth_dim > 0:
    depth_obs = np.flip(obs[rgb_dim:rgb_dim+depth_dim].reshape(image_size, image_size, 1), axis=0)
    rgb_obs = np.concatenate([rgb_obs, depth_obs], axis=2)

print(rgb_obs[:,:,0].min(), rgb_obs[:,:,0].max(), rgb_obs[:,:,1].min(), rgb_obs[:,:,1].max(), rgb_obs[:,:,2].min(), rgb_obs[:,:,2].max(), rgb_obs[:,:,3].min(), rgb_obs[:,:,3].max())
rgb_obs = rgb_obs.astype(np.uint8)
frames_rgb.append(rgb_obs)

obs2 = 255.0 - np.flip(obs[image_size*image_size*3:image_size*image_size*4].reshape(image_size, image_size), axis=0) * 255.0
obs2 = obs2.astype(np.uint8)
frames_depth.append(obs2)

done = False
ret = 0.
i = 0
start_time = time.time()
while not done:
    action = get_policy_action()         # use observation to decide on an action
    # action = action_lst[i]
    # print(action)
    obs, reward, done, _ = env2.step(action) # play action
    ret += reward
    
    obs1 = np.flip(obs[:image_size*image_size*3].reshape(image_size, image_size, 3), axis=0)
    obs1 = obs1.astype(np.uint8)
    frames_rgb.append(obs1)

    obs2 = 255.0 - np.flip(obs[image_size*image_size*3:image_size*image_size*4].reshape(image_size, image_size), axis=0) * 255.0
    obs2 = obs2.astype(np.uint8)
    frames_depth.append(obs2)
    i += 1

print("rollout completed with return {}".format(ret))
print(f"Spend {time.time() - start_time:.3f} s to run 1000 steps")

path = "images/view_rgb.mp4"
imageio.mimsave(path, frames_rgb, fps=90)

path = "images/view_depth.mp4"
imageio.mimsave(path, frames_depth, fps=90)

# find the difference between the two set of frames
# for i in range(horizon):
#     err = np.linalg.norm(frames[i] - frames2[i])
#     print(err)
#     assert(err < 1e-10)

env.close()
env2.close()

# Get the camera object
# camera = env.sim
# pprint.pprint(vars(camera))
# print("Camera object:", camera)

# Set the camera position and orientation
# camera.set_pos([0.5, 0.5, 1.0])
# camera.set_quat([1, 0, 0, 0])  # Identity quaternion to face the origin
# camera.set_pos([1.6, 0, 1.45])
# camera.set_quat([0.56, 0.43, 0.43, 0.56])


'''
low, high = env.action_spec
action_lst = []
for _ in range(horizon):
    action_lst.append(np.random.uniform(low, high))

# frames = []

# # reset the environment to prepare for a rollout
# obs = env.reset()
# frame = np.flip(obs[train_camera_names+"_image"], axis=0)
# frames.append(frame)
# # plt.imsave("images/testview.png", frame)

# # import ipdb; ipdb.set_trace()
# done = False
# ret = 0.

# i = 0
# start_time = time.time()
# while not done:
#     # action = get_policy_action(obs)         # use observation to decide on an action
#     action = action_lst[i]
#     # print(action)
#     obs, reward, done, _ = env.step(action) # play action
#     ret += reward
#     frame = np.flip(obs[train_camera_names+"_image"], axis=0)
#     frames.append(frame)
#     i += 1

# print("rollout completed with return {}".format(ret))
# print(f"Spend {time.time() - start_time:.3f} s to run 1000 steps")

# path = "images/testvideo.mp4"
# imageio.mimsave(path, frames, fps=30)

# ----------------------------------------------------------------------------------------
frames1 = []
frames2 = []
obs = env2.reset()
print(obs[:image_size*image_size*3].min(), obs[:image_size*image_size*3].max(), obs[image_size*image_size*3:image_size*image_size*3*2].min(), obs[image_size*image_size*3:image_size*image_size*3*2].max(), obs[image_size*image_size*3*2:].min(), obs[image_size*image_size*3*2:].max())
print(obs.shape)
obs1 = np.flip(obs[:image_size*image_size*3].reshape(image_size, image_size, 3), axis=0)
obs1 = obs1.astype(np.uint8)
frames1.append(obs1)

obs2 = np.flip(obs[image_size*image_size*3:image_size*image_size*3*2].reshape(image_size, image_size, 3), axis=0)
obs2 = obs2.astype(np.uint8)
frames2.append(obs2)

done = False
ret = 0.
i = 0
start_time = time.time()
while not done:
    # action = get_policy_action(obs)         # use observation to decide on an action
    action = action_lst[i]
    # print(action)
    obs, reward, done, _ = env2.step(action) # play action
    ret += reward
    
    obs1 = np.flip(obs[:image_size*image_size*3].reshape(image_size, image_size, 3), axis=0)
    obs1 = obs1.astype(np.uint8)
    frames1.append(obs1)

    if i > 200:
        obs2 = np.flip(obs[image_size*image_size*3:image_size*image_size*3*2].reshape(image_size, image_size, 3), axis=0)
        obs2 = obs2.astype(np.uint8)
        frames2.append(obs2)
    else:
        obs2 = obs[image_size*image_size*3:image_size*image_size*3*2].reshape(image_size, image_size, 3)
        obs2 = obs2.astype(np.uint8)
        frames2.append(obs2)
    i += 1

print("rollout completed with return {}".format(ret))
print(f"Spend {time.time() - start_time:.3f} s to run 1000 steps")

path = "images/view1.mp4"
imageio.mimsave(path, frames1, fps=90)

path = "images/view2.mp4"
imageio.mimsave(path, frames2, fps=90)

# find the difference between the two set of frames
# for i in range(horizon):
#     err = np.linalg.norm(frames[i] - frames2[i])
#     print(err)
#     assert(err < 1e-10)

env.close()
env2.close()
'''