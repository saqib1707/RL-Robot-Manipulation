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

# import tensorflow as tf
# import tensorflow.keras.mixed_precision as prec
# tf.get_logger().setLevel('ERROR')
# from tensorflow_probability import distributions as tfd

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Number of GPU devices:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
    print('Allocated memory:', round(torch.cuda.memory_allocated(0)/1024**3, 3), 'GB')
    print('Cached memory:   ', round(torch.cuda.memory_reserved(0)/1024**3, 3), 'GB')
else:
    print("Device:", device)

# load default controller parameters for Operational Space Control (OSC)
# controller_config = load_controller_config(default_controller="OSC_POSE")
# controller_config = load_controller_config(default_controller=None)
controller_config = None
train_camera_names = "agentview"

# create an environment to visualize on-screen
env = suite.make(
    env_name="Lift",
    robots=["Panda"],                # load a Sawyer robot and a Panda robot
    gripper_types="default",         # use default grippers for robot arm
    controller_configs=controller_config,      # each arm is controlled using OSC
    # env_configuration="single-arm-opposed",    # arms face each other
    reward_shaping=True, 
    has_renderer=False,                         # on-screen rendering
    render_camera="frontview",                 # visualize the frontview camera
    has_offscreen_renderer=True,              # no off-screen rendering
    control_freq=20,                    
    horizon=100,               # each episode terminates after 200 steps
    use_object_obs=False,     # no observations needed
    use_camera_obs=True,     # don't provide camera/image observations to agent
    camera_names=train_camera_names, 
)
print("Robot type:", env.robots[0], len(env.robots))
print("Environment created")

# env2 = GymWrapper(env)
# env2.seed(0)

# Get the camera object
camera = env.sim
pprint.pprint(vars(camera))
# print("Camera object:", camera)

# Set the camera position and orientation
# camera.set_pos([0.5, 0.5, 1.0])
# camera.set_quat([1, 0, 0, 0])  # Identity quaternion to face the origin
# camera.set_pos([1.6, 0, 1.45])
# camera.set_quat([0.56, 0.43, 0.43, 0.56])


# def get_policy_action(obs):
#     low, high = env.action_spec
#     action = np.random.uniform(low, high)
#     return action

# low, high = env.action_spec
# action_lst = []
# for _ in range(100):
#     action_lst.append(np.random.uniform(low, high))

# frames = []
# frames2 = []

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

# # ----------------------------------------------------------------------------------------
# obs = env2.reset()
# # print(next_obs[:256*256*3].min(), next_obs[:256*256*3].max(), next_obs[256*256*3:].min(), next_obs[256*256*3:].max())
# obs = np.flip(obs[:256*256*3].reshape(256, 256, 3), axis=0)
# obs = obs.astype(np.uint8)
# frames2.append(obs)

# done = False
# ret = 0.

# i = 0
# start_time = time.time()
# while not done:
#     # action = get_policy_action(obs)         # use observation to decide on an action
#     action = action_lst[i]
#     # print(action)
#     obs, reward, done, _ = env2.step(action) # play action
#     ret += reward
#     obs = np.flip(obs[:256*256*3].reshape(256, 256, 3), axis=0)
#     obs = obs.astype(np.uint8)
#     frames2.append(obs)
#     i += 1

# print("rollout completed with return {}".format(ret))
# print(f"Spend {time.time() - start_time:.3f} s to run 1000 steps")

# path = "images/testvideo2.mp4"
# imageio.mimsave(path, frames2, fps=30)

# env.close()
# env2.close()

# # find the difference between the two set of frames
# for i in range(100):
#     err = np.linalg.norm(frames[i] - frames2[i])
#     print(err)
#     assert(err < 1e-10)