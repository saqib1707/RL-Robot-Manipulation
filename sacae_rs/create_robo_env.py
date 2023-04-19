import numpy as np
import torch
import argparse
import os
import sys
import random
import time
import math

print("stage-1")
# import tensorflow as tf
# import tensorflow.keras.mixed_precision as prec
# tf.get_logger().setLevel('ERROR')
# from tensorflow_probability import distributions as tfd

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

print("stage-2")
# load default controller parameters for Operational Space Control (OSC)
# controller_config = load_controller_config(default_controller="OSC_POSE")

# print("stage-3")
# create an environment to visualize on-screen
# env = suite.make(
#     env_name="Lift",
#     robots=["Panda"],                # load a Sawyer robot and a Panda robot
#     gripper_types="default",                   # use default grippers for robot arm
#     controller_configs=controller_config,      # each arm is controlled using OSC
#     env_configuration="single-arm-opposed",    # arms face each other
#     has_renderer=True,                         # on-screen rendering
#     render_camera="frontview",                 # visualize the frontview camera
#     has_offscreen_renderer=False,              # no off-screen rendering
#     control_freq=20,                    
#     horizon=200,                               # each episode terminates after 200 steps
#     use_object_obs=False,                      # no observations needed
#     use_camera_obs=False,                      # don't provide camera/image observations to agent
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Number of GPU devices:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
    print('Allocated memory:', round(torch.cuda.memory_allocated(0)/1024**3, 3), 'GB')
    print('Cached memory:   ', round(torch.cuda.memory_reserved(0)/1024**3, 3), 'GB')
else:
    print("Device:", device)

env = suite.make(
    env_name="Lift", 
    robots="Panda",   
    has_renderer=False,
    use_camera_obs=True, 
    has_offscreen_renderer=True, 
    horizon=1000)

print("Robot type:", env.robots[0], len(env.robots))
print("Environment created")

def get_policy_action(obs):
    low, high = env.action_spec
    action = np.random.uniform(low, high)
    return action

# reset the environment to prepare for a rollout
obs = env.reset()

# import ipdb; ipdb.set_trace()
done = False
ret = 0.

start_time = time.time()
while not done:
    action = get_policy_action(obs)         # use observation to decide on an action
    # print(action)
    obs, reward, done, _ = env.step(action) # play action
    ret += reward

print("rollout completed with return {}".format(ret))
print(f"Spend {time.time() - start_time:.3f} s to run 1000 steps")

env.close()