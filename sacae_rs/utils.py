import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
import matplotlib.pyplot as plt
import pdb


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, args=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.obs_shape = obs_shape
        self.reduce_rb_size = args.reduce_rb_size
        self.frame_stack = args.frame_stack

    def add(self, obs, action, reward, next_obs, done):
        # print("stage-1:", obs.shape, next_obs.shape)  # (3,84,84)
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        low = 0
        high = self.capacity if self.full else self.idx
        if self.reduce_rb_size == True:
            # high = high - self.frame_stack
            idxs = np.random.randint(low, high, size=self.batch_size)
            idxs3 = np.max(0, np.vstack([idxs-2, idxs-1, idxs]).T.flatten())
            idxs4 = np.max(0, np.vstack([idxs-1, idxs, idxs+1]).T.flatten())

            obses = torch.as_tensor(self.obses[idxs3].reshape((-1, 3*self.frame_stack, *self.obs_shape[-2:])), device=self.device).float()
            next_obses = torch.as_tensor(self.next_obses[idxs3].reshape((-1, 3*self.frame_stack, *self.obs_shape[-2:])), device=self.device).float()
        else:
            idxs = np.random.randint(low, high, size=self.batch_size)
            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, img_shape, action_repeat):
        gym.Wrapper.__init__(self, env)
        self._num_frames = num_frames
        self._action_repeat = action_repeat
        self._frames = deque([], maxlen=num_frames)
        # self._max_episode_steps = env._max_episode_steps
        self._max_episode_steps = (env.horizon - action_repeat + 1) // action_repeat

        if img_shape is not None:
            self._img_height, self._img_width, self._img_channels = img_shape
            self.cam_obs_dim = img_shape[0] * img_shape[1] * img_shape[2]
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(img_shape[2] * num_frames, img_shape[0], img_shape[1]),
                dtype=env.observation_space.dtype
            )
        else:
            shp = env.observation_space.shape
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=((shp[0] * num_frames,) + shp[1:]),
                dtype=env.observation_space.dtype
            )

    def get_observation(self, obs):
        # print("Stage:", obs.shape, obs[:self.cam_obs_dim].min(), obs[:self.cam_obs_dim].max(), obs[self.cam_obs_dim:].min(), obs[self.cam_obs_dim:].max())
        if self.cam_obs_dim > 0:
            obs = np.flip(obs[:self.cam_obs_dim].reshape(self._img_height, self._img_width, self._img_channels), axis=0)
            obs = obs.astype(np.uint8)
            # plt.imsave('images/test1.png', obs)
            obs = np.transpose(obs, (2,0,1))
        return obs

    def reset(self):
        obs = self.env.reset()
        obs = self.get_observation(obs)
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        reward = 0
        for _ in range(self._action_repeat):    
            obs, rd, done, info = self.env.step(action)
            reward += rd
            if done:
                break
        obs = self.get_observation(obs)
        # print("stage-step:", obs.shape, np.max(obs), np.min(obs), reward)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)
