import pathlib
import datetime
import uuid
import io
import imageio

import numpy as np
import matplotlib.pyplot as plt
import torch

import utils
import robosuite as suite
# from lapal.models.sac_ae import SacAeAgent
from human_policy import ReachPolicy, BaseHumanPolicy, LiftPolicy, PickPlacePolicy


class AttrDict(dict):
    """
    This class allows you to access python dictionary elements using dot 
    notation `obj.key` instead of the usual square brackets `obj['key']`.
    """
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def eplen(episode):
    return len(episode['action'])


def sample_episodes(env, policy, directory, num_episodes=1, policy_obs_keys=None):
    # Save all observation keys from environment
    episodes_saved = 0
    episodes_reward_lst = []
    while episodes_saved < num_episodes:
        obs = env.reset()
        if isinstance(policy, BaseHumanPolicy):
            policy.reset()

        obs_keys = list(obs.keys())
        done = False
        episode = {}
        for k in obs_keys:
            episode[k] = [obs[k]]
        episode['action'] = []
        episode['reward'] = []

        while not done:
            if policy_obs_keys is not None:
                policy_obs = np.concatenate([obs[k] for k in policy_obs_keys])
            else:
                policy_obs = obs
            action, _ = policy.predict(policy_obs)
            obs, rew, done, info = env.step(action)

            for k in obs_keys:
                episode[k].append(obs[k])
            episode['action'].append(action)
            episode['reward'].append(rew)

        print(f"Episode return: {np.sum(episode['reward']):.2f}")
        episodes_reward_lst.append(np.sum(episode['reward']))
        save_episode(env, directory, episode)
        episodes_saved += 1
        # if np.sum(episode['reward']) < 160:
        #     print("Discarding current episode")
        #     continue
        # else:
        #     save_episode(env, directory, episode)
        #     episodes_saved += 1
    
    plt.figure()
    plt.plot(episodes_reward_lst)
    plt.axhline(y=env.horizon, color='r', linestyle='-')
    plt.gca().set_ylim([0, env.horizon+20])
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    savepath = directory / f'reward_plot.png'
    plt.savefig(savepath)


def save_episode(env, directory, episode):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    data_dir = pathlib.Path(f"{directory}/expert_data/")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=False)
    filename = data_dir / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    # save the image observations from each episode in a video to verify whether the
    # lifting process works correctly
    video_dir = pathlib.Path(f"{directory}/expert_videos/")
    if not video_dir.exists():
        video_dir.mkdir(parents=True, exist_ok=False)
    video_path = video_dir / f'{timestamp}-{identifier}-{length}.mp4'
    video_writer = imageio.get_writer(video_path, fps=30)
    camera_names = env.camera_names[0]+"_image"
    for i in range(eplen(episode)):
        video_writer.append_data(episode[camera_names][i])
    video_writer.close()
    return filename


def save_human_episodes():
    env_name = "Lift"
    # env_name = "PickPlaceBread"
    # env_name = "Reach"

    robots = "Panda"
    # robots = "Sawyer"
    # robots = "UR5e"
    
    image_size = 256
    horizon = 250
    # camera_names = "robot0_eye_in_hand"
    camera_names = "frontview"

    controller_type = "OSC_POSE"    # Only OSC_POSE controller is available for human expert
    controller_config = suite.load_controller_config(default_controller=controller_type)

    env = suite.make(
        env_name=env_name, 
        robots=robots, 
        gripper_types="default",
        controller_configs=controller_config,
        reward_shaping=True, 
        has_renderer=False, 
        has_offscreen_renderer=True, 
        use_camera_obs=True, 
        use_object_obs=True,
        camera_depths=True,
        control_freq=20, 
        horizon=horizon, 
        camera_names=camera_names, 
        camera_heights=image_size, 
        camera_widths=image_size, 
        use_tactile_obs=False,
        use_touch_obs=True
    )
    env._max_episode_steps = env.horizon
    print("Environment created")

    policy = LiftPolicy(env)
    # policy = ReachPolicy(env)
    # policy = PickPlacePolicy(env)
    print("Policy object created")

    # policy_obs_keys = ['target_to_robot0_eef_pos']

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # policy_obs_keys = [
    #     'robot0_eef_pos',
    #     'target_to_robot0_eef_pos',
    # ]

    # ob_encoder_params = utils.AttrDict(
    #     type='identity',
    #     input_dim=6,
    #     feature_dim=6,
    #     n_layers=3,
    #     hidden_dim=512,
    #     activation='relu',
    #     output_activation='identity',
    #     lr=3e-4,
    #     update_freq=1,
    # )
    # actor_critic_params = utils.AttrDict(
    #     lr=3e-4, 
    #     n_layers=3, 
    #     hidden_dim=512,
    #     update_freq=2,
    #     tau=0.005,
    # )
    # disc_params = utils.AttrDict(
    #     reward_type=None, 
    #     n_layers=3,
    #     hidden_dim=256,
    #     activation='relu',
    #     spectral_norm=True,
    #     lr=3e-5,
    #     update_freq=1,
    # )

    # policy = SacAeAgent(
    #     (6,),
    #     (7,),
    #     device=device,
    #     ob_encoder_params=ob_encoder_params,
    #     actor_critic_params=actor_critic_params,
    #     disc_params=disc_params,
    # )
    # model_dir = "./data/05.01.2023/14-42-59_Reach_OSC_POSE_SAC_OSC/models/step_0050000"
    # policy.load(model_dir)

    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    directory = pathlib.Path(f"../../../VMAIL/expert_data/robosuite_expert/{env_name}/{robots}/{controller_type}/{timestamp}-{camera_names}-{horizon}/")
    print("Store directory:", directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    sample_episodes(env, policy, directory, num_episodes=64, policy_obs_keys=None)

    env.close()


if __name__ == '__main__':
    save_human_episodes()