import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
from datetime import datetime, timezone
import pytz
import json
import dmc2gym
import copy
from tqdm import tqdm
import imageio
import pdb

import utils
from logger import Logger
from video import VideoRecorder

from sac_ae import SacAeAgent

import robosuite as suite
# from robosuite.wrappers import GymWrapper
from gym_wrapper import GymWrapper
from robosuite.controllers import load_controller_config

utc_dt = datetime.now(timezone.utc).astimezone(pytz.timezone('US/Pacific'))


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default="Lift", type=str)
    parser.add_argument('--robots', default="Panda", type=str, nargs='+')
    parser.add_argument('--controller', default="", type=str)
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--train_camera_names', default="agentview", type=str, nargs='+', help="Cameras used to generate views for training")
    parser.add_argument('--render_camera_names', default="frontview", type=str, nargs='+', help="Names of camera to render")
    parser.add_argument('--horizon', type=int, default=1000, help="every episode lasts for exactly horizon timesteps")
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    parser.add_argument('--reduce_rb_size', default=False, action='store_true')
    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--video_save_freq', default=50000, type=int)
    parser.add_argument('--video_height', default=512, type=int)
    parser.add_argument('--video_width', default=512, type=int)
    parser.add_argument('--video_fps', default=90, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--use_camera_depth', default=False, action='store_true')
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='./logdir/', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--load_saved_logdir', default="", type=str)
    parser.add_argument('--start_step', default=0, type=int)

    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, video_save_freq):
    start_time = time.time()
    for i in tqdm(range(num_episodes)):
        obs = env.reset()
        video_filename = '%d.mp4' % step
        video.init(enabled=(i == 0 and step % video_save_freq == 0), filename=video_filename)
        done = False
        episode_reward = 0

        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            video.record(env, obs)
            episode_reward += reward

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
    L.log('eval/duration', time.time() - start_time, step)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'sac_ae':
        return SacAeAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_lambda=args.decoder_weight_lambda,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def print_env_info(env):
    print("Env Name:", env.name)
    print("Env horizon:", env.horizon)
    print("Camera resolution:", env.camera_heights[0], env.camera_widths[0])
    print("Observation space:", env.observation_space.shape, env.observation_space.low.min(), env.observation_space.high.max(), np.array(env.observation_space.low).shape, np.array(env.observation_space.high).shape)
    print("Action space:", env.action_space.shape, env.action_space.low.min(), env.action_space.high.max(), np.array(env.action_space.low).shape, np.array(env.action_space.high).shape, "\n")


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("Number of GPU devices:", torch.cuda.device_count())
        print("GPU device name:", torch.cuda.get_device_name(0))
        print('Allocated memory:', round(torch.cuda.memory_allocated(0)/1024**3, 3), 'GB')
        print('Cached memory:   ', round(torch.cuda.memory_reserved(0)/1024**3, 3), 'GB')
    else:
        print("Device:", device)

    if args.controller == "OSC_POSE":
        # load default controller parameters for Operational Space Control (OSC)
        controller_config = load_controller_config(default_controller="OSC_POSE")
    elif args.controller == "robomimic":
        with open("controller_config/robomimic.json", 'r') as f:
            controller_config = json.load(f)
    else:
        controller_config = None

    # create robosuite environment and wrap using gym library
    print("Creating robosuite environment ...")
    env = suite.make(
        env_name=args.domain_name, 
        robots=args.robots, 
        controller_configs=controller_config, 
        reward_shaping=True,          # if True, uses dense rewards else sparse 
        has_renderer=False, 
        has_offscreen_renderer=True, 
        use_camera_obs=(args.encoder_type == 'pixel'), 
        use_object_obs=False, 
        camera_depths=args.use_camera_depth, 
        horizon=args.horizon, 
        camera_names=args.train_camera_names, 
        camera_heights=args.image_size, 
        camera_widths=args.image_size, 
    )
    print("Robosuite env created !!!")
    num_cameras = len(env.camera_names)
    num_channels = 4 if env.camera_depths[0] == True else 3

    env = GymWrapper(env)
    print_env_info(env)
    print("Robosuite env wrapped in gym !!!")

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, num_frames=args.frame_stack, img_shape=(env.camera_heights[0], env.camera_widths[0], num_channels), action_repeat=args.action_repeat, num_cameras=num_cameras)
        print("Frames stacked !!!")
    print_env_info(env)

    if args.load_saved_logdir == "":
        args.work_dir = os.path.join(args.work_dir, 'log_'+utc_dt.strftime('%Y%m%d_%H%M%S'))
    else:
        args.work_dir = os.path.join(args.work_dir, args.load_saved_logdir)
        # train_log_filepath = os.path.join(args.work_dir, 'train.log')
        # eval_log_filepath = os.path.join(args.work_dir, 'eval.log')
        # if os.path.exists(train_log_filepath):
        #     with open(train_log_filepath, 'r') as f:
        #         train_log = json.loads(f)
        #         print("Train log:", train_log)
        # if os.path.exists(eval_log_filepath):
        #     with open(eval_log_filepath, 'r') as f:
        #         eval_log = json.load(f)
    
    utils.make_dir(args.work_dir)
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, height=args.video_height, width=args.video_width, fps=args.video_fps, render_camera_names=args.render_camera_names)

    if args.load_saved_logdir == "":
        with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    print("Creating replay buffer...")
    num_ch = num_channels * num_cameras if args.reduce_rb_size == True else num_channels * args.frame_stack * num_cameras
    replay_buffer = utils.ReplayBuffer(
        obs_shape=(num_ch, env.camera_heights[0], env.camera_widths[0]) if args.encoder_type == 'pixel' else env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        args=args
    )
    print("Replay buffer created !!!")

    if args.load_saved_logdir != "":
        replay_buffer.load(buffer_dir)
        print("Replay buffer loaded")

    print("Creating Agent...")
    agent = make_agent(
        obs_shape=(num_channels * args.frame_stack * num_cameras, env.camera_heights[0], env.camera_widths[0]) if args.encoder_type == 'pixel' else env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )
    print("Agent created !!!")

    if args.load_saved_logdir != "":
        agent.load(model_dir=model_dir, step=args.start_step)
        print("Agent model loaded")

    if args.load_saved_logdir == "":
        L = Logger(args.work_dir, use_tb=args.save_tb, resume_training=False)
    else:
        L = Logger(args.work_dir, use_tb=args.save_tb, resume_training=True)

    if args.load_saved_logdir == "":
        episode_idx, episode_reward, done = 0, 0, True
        start_itr = 0
    else:
        episode_idx = args.start_step // args.horizon + 1
        done = False
        episode_reward = 0
        start_itr = args.start_step
        obs = env.reset()
        episode_step = 0
        L.log('train/episode_idx', episode_idx, args.start_step+1)
    
    print("Starting training loop...")
    start_time = time.time()
    for itr in tqdm(range(start_itr, args.num_train_steps)):
        if done:
            if itr > 0:
                L.log('train/duration', time.time() - start_time, itr)
                L.log('train/episode_reward', episode_reward, itr)
                L.dump(itr)
                start_time = time.time()

            # evaluate agent periodically
            if itr % args.eval_freq == 0:
                L.log('eval/episode_idx', episode_idx, itr)
                
                print("Starting evaluation...")
                evaluate(env, agent, video, args.num_eval_episodes, L, itr, args.video_save_freq)
                print("Evaluation complete !!!")
                
                if args.save_model:
                    agent.save(model_dir, itr)
                    print("Saved model and tensorboard data!!!")

                if args.save_buffer:
                    replay_buffer.save(buffer_dir)
                    print("Saved replay buffer!!!")

            # reset the environment, episode reward, episode step, and update the episode index
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode_idx += 1

            L.log('train/episode_idx', episode_idx, itr)

        # sample action for data collection
        if itr < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)
        
        # run training update
        if itr >= args.init_steps:
            num_updates = args.init_steps if itr == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, itr)

        next_obs, reward, done, _ = env.step(action)

        # allow infinite bootstrap
        episode_reward += reward
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)

        if args.reduce_rb_size == True:
            st = (args.frame_stack-1) * 3
            ed = args.frame_stack * 3
            replay_buffer.add(obs[st:ed], action, reward, next_obs[st:ed], done_bool)
        else:
            replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1

    env.close()


if __name__ == '__main__':
    main()