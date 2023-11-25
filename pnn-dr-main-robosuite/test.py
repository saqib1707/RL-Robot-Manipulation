# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from datetime import datetime, timezone
import pytz
import torch
import cv2
import matplotlib.pyplot as plt

# from irb120 import IRB120Env
from Panda import RobosuiteEnv
from model import ActorCritic
from utils import state_to_tensor, plot_line, set_seed_everywhere

tmz = pytz.timezone('US/Pacific')


def test(rank, args, T, shared_model, device):
    """
    Validation method that test the model during training.

    Args:
        rank (int): variable to set the seed.
        args (argparse.Namespace): arguments set by the user.
        T (Counter): global shared counter.
        shared_model (model.ActorCritic): current global model.
    """
    # If fine rendering enabled, increase the image size
    # if args.fine_render:
    #     args.height = 640
    #     args.width = 640

    camview_rgb = args.camviews + '_image'

    # instantiate the robosuite environment
    set_seed_everywhere(args.seed + rank)
    env = RobosuiteEnv(
        args=args, 
        task=args.task, 
        horizon=args.max_episode_length, 
        size=(args.width, args.height), 
        camviews=args.camviews, 
        reward_shaping=args.reward_continuous
    )
    # env.seed(args.seed + rank)

    # If fine rendering enabled, Visualization parameters
    if args.fine_render:
        # (_, obs_rgb) = env.reset()
        obs_rgb = env.reset()[camview_rgb]
        plt.ion()
        f, ax = plt.subplots()
        im = ax.imshow(obs_rgb)

    model = ActorCritic(args.hidden_size, rgb_width=args.width, rgb_height=args.height).to(device)    # Instantiate the model
    model.eval()     # setting the model to evaluation mode

    can_test = True    # Test flag
    t_start = 1        # Test step counter to check against global counter
    
    rewards, steps = [], []        # Rewards and steps for plotting
    # reward_step, steps_count = [], []     # Rewards and steps for metrics
    
    done = True          # Start new episode
    while T.value() <= args.T_max - 1:
        if can_test:
            t_start = T.value()     # Reset counter
            
            # Evaluate over several episodes and average results
            avg_rewards, avg_episode_lengths = [], []
            
            for _ in range(args.evaluation_episodes):
                while True:
                    # Reset or pass on hidden state
                    if done:
                        model.load_state_dict(shared_model.state_dict())     # Sync with shared model every episode
                        with torch.no_grad():
                            hx = torch.zeros(1, args.hidden_size, device=device)      # LSTM hidden state
                            cx = torch.zeros(1, args.hidden_size, device=device)      # LSTM cell state
                        
                        # Reset environment and done flag
                        if args.fine_render:
                            state = state_to_tensor(env.reset()[camview_rgb], device)
                            # state = state_to_tensor((obs, cv2.resize(obs_rgb, (64, 64))))
                        else:
                            # state = state_to_tensor(env.reset())
                            state = state_to_tensor(env.reset()[camview_img], device)
                        
                        action, reward, done, episode_length = (0, 0, 0, 0, 0, 0, 0), 0, False, 0
                        episode_reward = 0

                    # Calculate policy
                    with torch.no_grad():
                        policy, _, (hx, cx) = model(state, (hx.detach(), cx.detach()))    # Break graph for memory efficiency

                    # Choose action greedily
                    action = [p.max(1)[1].data[0].cpu() for p in policy]

                    # Step
                    if args.fine_render:
                        # state, reward, done = env.step(action, episode_length)
                        state, reward, done = env.step(action)
                        # obs_rgb = state[1]
                        # state = state_to_tensor((state[0], cv2.resize(obs_rgb, (64, 64))))
                        state = state_to_tensor(state[camview_rgb], device)

                    # Save outcomes
                    # reward_step.append(reward)
                    # steps_count.append(episode_length)

                    episode_reward += reward
                    done = done or episode_length >= args.max_episode_length - 1    # Stop episodes at a max length
                    episode_length += 1      # Increase episode counter

                    # Optionally render validation states (uncomment to see it)
                    if args.fine_render:
                        im.set_data(obs_rgb)
                        plt.draw()
                        plt.pause(0.001)

                    # Log and reset statistics at the end of every episode
                    if done:
                        avg_rewards.append(episode_reward)
                        avg_episode_lengths.append(episode_length)
                        break

            print("Evaluation during training")
            print(
                ("[{}] Step: {}/{}  Avg.Reward: {}   Avg.Episode Length:{}").format(
                    datetime.now(timezone.utc).astimezone(tmz).strftime('%Y-%m-%d %H:%M:%S'), 
                    t_start,
                    args.T_max, 
                    sum(avg_rewards) / args.evaluation_episodes,
                    sum(avg_episode_lengths) / args.evaluation_episodes,
                )
            )

            # Keep all evaluations
            rewards.append(avg_rewards)
            steps.append(t_start)

            plot_line(steps, rewards)    # Plot rewards
            torch.save(model.state_dict(), os.path.join(args.logdir, str(t_start) + "_model.pth"))    # Checkpoint model params
            can_test = False    # Finish testing

            if args.evaluate:
                return
        else:
            if T.value() - t_start >= args.evaluation_interval:
                can_test = True

        time.sleep(0.001)  # Check if available to test every millisecond
