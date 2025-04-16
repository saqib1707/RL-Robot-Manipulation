import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lapal.models.encoder import make_encoder
from lapal.utils import utils

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(self, obs_dim, action_dim, actor_params):
        super().__init__()

        self.log_std_min = -10
        self.log_std_max = 2

        n_layers = actor_params.n_layers
        hidden_dim = actor_params.hidden_dim
        self.trunk = utils.build_mlp(obs_dim, action_dim*2, n_layers, hidden_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, compute_log_pi=False): 

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()
        noise = torch.randn_like(mu)
        pi = mu + noise * std

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        outputs = {'mu': mu, 'pi': pi, 'log_pi': log_pi, 'log_std': log_std}
        return outputs


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, n_layers, hidden_dim):
        super().__init__()

        self.trunk = utils.build_mlp(obs_dim+action_dim, 1, n_layers, hidden_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(self, obs_dim, action_dim, critic_params):
        super().__init__()

        n_layers = critic_params.n_layers
        hidden_dim = critic_params.hidden_dim
        self.Q1 = QFunction(obs_dim, action_dim, n_layers, hidden_dim)
        self.Q2 = QFunction(obs_dim, action_dim, n_layers, hidden_dim)

        self.apply(utils.weight_init)

    def forward(self, obs, action):

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)
        return q1, q2
