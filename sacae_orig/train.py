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
import pdb

import utils
from logger import Logger
from video import VideoRecorder

from sac_ae import SacAeAgent

utc_dt = datetime.now(timezone.utc).astimezone(pytz.timezone('US/Pacific'))


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
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

    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step):
    for i in tqdm(range(num_episodes)):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
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


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("Number of GPU devices:", torch.cuda.device_count())
        print("GPU device name:", torch.cuda.get_device_name(0))
        print('Allocated memory:', round(torch.cuda.memory_allocated(0)/1024**3, 3), 'GB')
        print('Cached memory:   ', round(torch.cuda.memory_reserved(0)/1024**3, 3), 'GB')
    else:
        print("Device:", device)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat
    )
    env.seed(args.seed)
    print("Robosuite env created !!!")

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)
        print("Frames stacked !!!")

    args.work_dir = os.path.join(args.work_dir, 'log_'+utc_dt.strftime('%Y%m%d_%H%M%S'))
    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    print("Creating replay buffer...")
    if args.reduce_rb_size == True:
        replay_buffer = utils.ReplayBuffer(
        obs_shape=(3, args.image_size, args.image_size) if args.encoder_type == 'pixel' else env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        args=args
    )
    else:
        replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        args=args
    )
    print("Replay buffer created !!!")

    print("Creating Agent...")
    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )
    print("Agent created !!!")

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    print("Starting training loop...")
    for itr in range(args.num_train_steps):
        if done:
            if itr > 0:
                L.log('train/duration', time.time() - start_time, itr)
                L.log('train/episode_reward', episode_reward, itr)
                start_time = time.time()
                L.dump(itr)

            # evaluate agent periodically
            if itr % args.eval_freq == 0:
                L.log('eval/episode', episode, itr)
                print("Starting evaluation...")
                evaluate(env, agent, video, args.num_eval_episodes, L, itr)
                print("Evaluation complete !!!")
                if args.save_model:
                    agent.save(model_dir, itr)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, itr)

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

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        if args.reduce_rb_size == True:
            start_idx = (args.frame_stack - 1) * 3
            end_idx = args.frame_stack * 3
            replay_buffer.add(obs[start_idx:end_idx], action, reward, next_obs[start_idx:end_idx], done_bool)
        else:
            replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = np.copy(next_obs)
        episode_step += 1
    
    env.close()


if __name__ == '__main__':
    main()
