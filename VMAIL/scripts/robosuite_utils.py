import os
import sys
import argparse
import uuid
import io
import random
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import imageio
import json
import pprint
import pathlib
from tqdm import tqdm
import colored_traceback
colored_traceback.add_hook()

import torch

import robosuite as suite
from robosuite.wrappers import GymWrapper
suite.macros.IMAGE_CONVENTION = "opencv"    # Set the image convention to opencv so the images are automatically rendered "right side up" when using imageio (which uses opencv convention)

from human_policy import ReachPolicy, BaseHumanPolicy, LiftPolicy, PickPlacePolicy
# from gym_wrapper import GymWrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("Number of GPU devices:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    print("Device:", device)


class AttrDict(dict):
    """
    This class allows you to access python dictionary elements using dot 
    notation `obj.key` instead of the usual square brackets `obj['key']`.
    """
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def args_type(default):
    if isinstance(default, bool):
        return lambda x: bool(['False', 'True'].index(x))
    if isinstance(default, int):
        return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    return type(default)


def define_config():
    config = AttrDict()

    config.env_name = "PickPlaceCereal"    # [PickPlaceBread, PickPlaceMilk, PickPlaceCereal, Lift, Reach]
    config.robots = "Panda"     # "Sawyer", "UR5e"
    config.controller = "OSC_POSE"
    if config.env_name.startswith("PickPlace"):
        config.horizon = 250
    elif config.env_name == "Lift" or config.env_name == "Reach":
        config.horizon = 100
    else:
        config.horizon = 250

    # config.camera_names = "frontview"
    config.camera_names = ["frontview", "agentview", "robot0_eye_in_hand"]

    config.use_camera_obs = True
    config.use_depth_obs = True
    config.use_object_obs = True
    config.use_proprio_obs = True
    config.use_touch_obs = True
    config.use_tactile_obs = False
    config.use_shape_obs = False

    config.img_width = 256
    config.img_height = 256
    
    return config


def random_policy(env, obs=None):
    low, high = env.action_spec
    action = np.random.uniform(low, high)
    return action


def create_env(config, verbose=False):
    assert isinstance(config.controller, str), "controller must be a str"
    if config.controller == "robomimic":
        with open("controller_config/robomimic.json", 'r') as f:
            controller_config = json.load(f)
    elif config.controller is not None:
        # load default controller parameters for Operational Space Control (OSC)
        # Only OSC_POSE controller is available for human expert
        controller_config = suite.controllers.load_controller_config(default_controller=config.controller)
    else:
        controller_config = None

    # create an environment to visualize on-screen
    env = suite.make(
        env_name=config.env_name,
        robots=config.robots,
        gripper_types="default",
        controller_configs=controller_config,
        reward_shaping=True, 
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=20,
        horizon=config.horizon,
        use_object_obs=config.use_object_obs,
        use_camera_obs=config.use_camera_obs,
        camera_depths=config.use_depth_obs,
        camera_heights=config.img_height, 
        camera_widths=config.img_width, 
        camera_names=config.camera_names, 
        use_tactile_obs=config.use_tactile_obs,
        use_touch_obs=config.use_touch_obs
    )
    # env._max_episode_steps = env.horizon
    print(f"Environment created")
    if verbose:
        obs = env.reset()    # reset the env
        for idx, (key, val) in enumerate(obs.items()):
            print(f"{idx} Var: {key}, shape: {val.shape}, range: ({val.min()}, {val.max()})")
    return env


def save_frames_to_video(frames, savepath, fps=30, verbose=False):
    if not str(savepath).endswith(".mp4"):
        print("Error: savepath must have extension '.mp4'")
        return False

    with imageio.get_writer(savepath, fps=fps) as writer:
        for i in range(len(frames)):
            # frame = np.array(Image.fromarray(frames[i]).resize((512, 512)))
            writer.append_data(frames[i])
    # imageio.mimsave(savepath, frames_depth, fps=30)
    if verbose:
        print(f"Saved frames to video at {savepath}")
    return True


def episode_rollout(env, config, save_frames=False):
    """
    Rolls out an episode using some policy and saves RGB + depth (maybe) 
    frames in a video
    """
    if isinstance(config.camera_names, str):
        rgb_cam = config.camera_names + "_image"
        depth_cam = config.camera_names + "_depth"
    elif isinstance(config.camera_names, list):
        rgb_cam = [camera_name + "_image" for camera_name in config.camera_names]
        depth_cam = [camera_name + "_depth" for camera_name in config.camera_names]

    obs = env.reset()    # reset the env
    rgb_frames = [obs[rgb_cam]]
    depth_frames = [np.uint8(obs[depth_cam] * 255)]

    ep_reward = 0
    start_time = time.time()

    # episode rollout
    done = False
    while not done:
        action = random_policy(env)         # use observation to decide on an action
        obs, reward, done, _ = env.step(action)    # play action
        ep_reward += reward

        rgb_frames.append(obs[rgb_cam])
        depth_image = np.uint8(obs[depth_cam] * 255)
        depth_frames.append(depth_image)

        # proprio_state = np.concatenate((obs["robot0_joint_pos_cos"], obs["robot0_joint_pos_sin"], obs["robot0_joint_vel"], obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], obs["robot0_gripper_qvel"]))
        # proprio_err = np.sum(proprio_state - obs["robot0_proprio-state"])
        # assert proprio_err == 0, f"Proprio-state error: {proprio_err}"

        # object_state = np.concatenate((obs["cube_pos"], obs["cube_quat"], obs["cube_to_robot0_eef_pos"], obs["cube_to_robot0_eef_quat"], obs["robot0_eef_to_cube_yaw"]))
        # object_err = np.sum(object_state - obs["object-state"])
        # assert object_err == 0, f"Object-state error: {object_err}"

        # touch_err = np.sum(obs["robot0_touch"] - obs["robot0_touch-state"])
        # assert touch_err == 0, f"Object-state error: {touch_err}"
    
    print(f"rollout completed with return {ep_reward}")
    print(f"Spent {time.time() - start_time:.3f} s to rollout {config.horizon} steps")
    if save_frames:
        save_frames_to_video(rgb_frames, "../demo_videos/rgb_video.mp4")
        save_frames_to_video(depth_frames, "../demo_videos/depth_video.mp4")
    return rgb_frames, depth_frames


def load_episodes(dirpath, capacity=None):
    """
    Returns a dict, with each key corr to a file in dirpath containing 
    one episode. The returned directory from filenames to episodes is guaranteed to be in
    temporally sorted order.
    """
    directory = pathlib.Path(dirpath)
    filepaths = sorted(directory.glob('*.npz'))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split('-')[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    
    episodes = {}
    for filepath in filepaths:
        try:
            with filepath.open('rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}    # this line is necessary for .npz file
        except Exception as e:
            print(f'Could not load episode {str(filepath)}: {e}')
            continue
        episodes[str(filepath)] = episode
    return episodes


def save_episode(env, dirpath, episode):
    """
    Saves an episode in a '.npz' file. Saves image observations in a video.
    """
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)[:8]
    
    data_dir = pathlib.Path(f"{dirpath}/episodes/")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=False)
    filename = data_dir / f'{timestamp}-{identifier}.npz'
    
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    
    # save the image observations in a video
    video_dir = pathlib.Path(f"{dirpath}/videos/")
    if not video_dir.exists():
        video_dir.mkdir(parents=True, exist_ok=False)

    # check whether the episode dict has any "_image" key
    img_keys = [key for key in episode.keys() if "_image" in key]
    for key in img_keys:
        frames = episode[key]
        video_path = video_dir / f"{timestamp}-{identifier}-{''.join(key.split('_')[0:-1])}.mp4"
        save_frames_to_video(frames, video_path)


def sample_episodes(env, policy, dirpath, num_eps=1, policy_obs_keys=None):
    eps_rewards = []
    for _ in tqdm(range(num_eps)):
        obs = env.reset()
        if isinstance(policy, BaseHumanPolicy):
            policy.reset()

        # Save all observation keys from env
        obs_keys = list(obs.keys())
        episode = {k : [obs[k]] for k in obs_keys}
        episode['action'] = []
        episode['reward'] = []

        # rollout an episode
        done = False
        while not done:
            if policy_obs_keys is not None:
                policy_obs = np.concatenate([obs[k] for k in policy_obs_keys])
            else:
                policy_obs = obs
            action, _ = policy.predict(policy_obs)
            obs, rew, done, _ = env.step(action)
            for k in obs_keys:
                episode[k].append(obs[k])
            episode['action'].append(action)
            episode['reward'].append(rew)

        # print(f"Episode total reward: {np.sum(episode['reward']):.2f}")
        eps_rewards.append(np.sum(episode['reward']))
        save_episode(env, dirpath, episode)
    
    # plot episode total rewards
    plt.figure()
    plt.plot(eps_rewards)
    plt.axhline(y=env.horizon, color='r', linestyle='-')
    plt.gca().set_ylim([0, env.horizon+20])
    plt.gca().set_xlim([0, len(eps_rewards)])
    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward")
    plt.grid(linestyle="--")
    savepath = dirpath / f'reward_plot.png'
    plt.savefig(savepath)


if __name__ == "__main__":
    config = define_config()
    env = create_env(config, verbose=True)

    if config.env_name == "Lift":
        policy = LiftPolicy(env)
    elif config.env_name == "Reach":
        policy = ReachPolicy(env)
    elif config.env_name.startswith("PickPlace"):
        policy = PickPlacePolicy(env)
    print("Policy object created")

    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    dirpath = pathlib.Path(f"../expert_data/robosuite/{config.env_name.lower()}/{config.robots.lower()}/{config.controller.lower()}/{timestamp}-{config.horizon}/")
    print("Store directory:", dirpath)
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)

    sample_episodes(env, policy, dirpath, num_eps=24, policy_obs_keys=None)
    env.close()