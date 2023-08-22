# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# from irb120 import IRB120Env
from Panda import RobosuiteEnv
from model import ActorCritic
from utils import state_to_tensor


def _transfer_grads_to_shared_model(model, shared_model):
    """
    Transfers gradients from process-specific model to shared model.

    Args:
        model (model.ActorCritic): neural network model.
        shared_model (model.ActorCritic): current global model.
    """
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def _adjust_learning_rate(optimiser, lr):
    """
    Adjusts learning rate.

    Args:
        optimiser (optim.SharedRMSprop): network optimiser.
        lr (float): learning rate.
    """
    for param_group in optimiser.param_groups:
        param_group["lr"] = lr


def _update_networks(args, T, model, shared_model, loss, optimiser, loss_values):
    """
    Update network parameters.

    Args:
        args (argparse.Namespace): arguments set by the user.
        T (Counter): global shared counter.
        model (model.ActorCritic): neural network model.
        shared_model (model.ActorCritic): current global model.
        loss (float): loss value.
        optimiser (optim.RMSprop): network optimiser.
        loss_values (list):list of loss values.
    """
    # Zero shared and local grads
    optimiser.zero_grad()
    # Calculate gradients (not losses defined as negatives of normal update rules for gradient descent)
    loss.backward()
    # Gradient L2 norm clipping
    nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm, 2)
    loss_values.append(loss.item())
    # Transfer gradients to shared model and update
    _transfer_grads_to_shared_model(model, shared_model)
    optimiser.step()
    if args.lr_decay:
        # Linearly decay learning rate
        _adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))


def _train(args, T, model, shared_model, optimiser, policies, Vs, actions, rewards, R, loss_values):
    """
    Train the model parameters.

    Args:
        args (argparse.Namespace): arguments set by the user.
        T (Counter): global shared counter.
        model (model.ActorCritic): neural network model.
        shared_model (model.ActorCritic): current global model.
        optimiser (optim.SharedRMSprop): network optimiser.
        policies (list): policies performed during the episode.
        Vs (list): values of the value state function obtained during the episode.
        actions (list): actions performed during the episode.
        rewards (list): rewards obtained during the episode.
        R (torch.Tensor): last inmediate reward obtained.
        loss_values (list):list of loss values.
    """
    policy_loss, value_loss = 0, 0
    # Generalised advantage estimator Ψ
    A_GAE = torch.zeros(1, 1)
    # Calculate n-step returns in forward view, stepping backwards from the last state
    t = len(rewards)
    for i in reversed(range(t)):
        # R ← r_i + γR
        R = rewards[i] + args.discount * R
        # Advantage A ← R - V(s_i; θ)
        A = R - Vs[i]
        # dθ ← dθ - ∂A^2/∂θ
        value_loss += 0.5 * A**2  # Least squares error
        # TD residual δ = r + γV(s_i+1; θ) - V(s_i; θ)
        td_error = rewards[i] + args.discount * Vs[i + 1].data - Vs[i].data
        # Generalised advantage estimator Ψ (roughly of form ∑(γλ)^t∙δ)
        A_GAE = A_GAE * args.discount * args.trace_decay + td_error
        # dθ ← dθ - ∇θ∙log(π(a_i|s_i; θ))∙Ψ - β∙∇θH(π(s_i; θ))
        for j, p in enumerate(policies[i]):
            policy_loss -= p.gather(1, actions[i][j].detach().unsqueeze(0).unsqueeze(0)).log() * Variable(A_GAE)
            policy_loss -= args.entropy_weight * -(p.log() * p).sum(1)
    # Optionally normalise loss by number of time steps
    if not args.no_time_normalisation:
        policy_loss /= t
        value_loss /= t
    # Update networks
    _update_networks(args, T, model, shared_model, policy_loss + value_loss, optimiser, loss_values)


def train(rank, args, T, shared_model, optimiser):
    """
    Act and train the model.

    Args:
        rank (int): variable to set the seed.
        args (argparse.Namespace): arguments set by the user.
        T (Counter): global shared counter.
        shared_model (model.ActorCritic): current global model.
        optimiser (optim.SharedRMSprop): network optimiser.
    """
    # Instantiate the environment
    # env = IRB120Env(
    #     args.width,
    #     args.height,
    #     args.frame_skip,
    #     args.rewarding_distance,
    #     args.control_magnitude,
    #     args.reward_continuous,
    #     args.max_episode_length,
    # )

    task = "Lift_task"
    env = RobosuiteEnv(task, horizon=args.max_episode_length)
    # env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    
    # Instantiate the model
    model = ActorCritic(args.hidden_size)
    model.train()
    
    loss_values = []     # Losses list
    t = 1                # Thread step counter
    # Start new episode
    done = True
    
    # Start training
    while T.value() <= args.T_max - 1:
        # print("Inside outer while")
        model.load_state_dict(shared_model.state_dict())    # Sync with shared model at least every t_max steps
        t_start = t      # Get starting timestep
        # Reset or pass on hidden state
        if done:
            hx = Variable(torch.zeros(1, args.hidden_size))    # LSTM hidden state
            cx = Variable(torch.zeros(1, args.hidden_size))    # LSTM cell state
            # Reset environment and done flag
            state = state_to_tensor(env.reset()["agentview_image"])
            action, reward, done, episode_length = (0, 0, 0, 0, 0, 0, 0), 0, False, 0
        else:
            # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
            hx = hx.detach()
            cx = cx.detach()
        
        # Lists of outputs for training
        policies, Vs, actions, rewards = [], [], [], []

        while not done and t - t_start < args.t_max - 1:
            # print("Inside inner while")
            # Calculate policy and value
            # policy, V, (hx, cx) = model(Variable(state[1]), (hx, cx))
            policy, V, (hx, cx) = model(Variable(state), (hx, cx))
            
            # Sample action
            # Graph broken as loss for stochastic action calculated manually
            action = [p.multinomial(num_samples=1).data[0] for p in policy]
            # state, reward, done = env.step(action, episode_length)    # Step into the environment
            # print("Action before step:", action)
            action = np.array(torch.stack(action).squeeze())
            # print("Action after:", action)

            state, reward, done = env.step(action)
            state = state_to_tensor(state["agentview_image"])

            done = done or episode_length >= args.max_episode_length - 1   # Stop episodes at a max length
            episode_length += 1      # Increase episode counter
            # Save outputs for online training
            [
                arr.append(el)
                for arr, el in zip(
                    (policies, Vs, actions, rewards), (policy, V, Variable(torch.LongTensor(action)), reward)
                )
            ]
            # Increment counters
            t += 1
            T.increment()

        # Break graph for last values calculated (used for targets, not directly as model outputs)
        if done:
            R = Variable(torch.zeros(1, 1))
        else:
            # R = V(s_i; θ) for non-terminal s
            # _, R, _ = model(Variable(state[1]), (hx, cx))
            _, R, _ = model(Variable(state), (hx, cx))
            R = R.detach()
        Vs.append(R)

        # Train the network
        _train(args, T, model, shared_model, optimiser, policies, Vs, actions, rewards, R, loss_values)
