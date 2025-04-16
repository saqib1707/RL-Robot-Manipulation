import torch
import torch.nn as nn
import torch.nn.functional as F

from lapal.utils import utils

class Discriminator(nn.Module):
    """Discriminator network, classifying latent observation and actions"""
    def __init__(self, obs_dim, action_dim, disc_params):
        super().__init__()

        self.reward_type = disc_params.reward_type

        self.disc = utils.build_mlp(
            obs_dim+action_dim, 
            1, 
            disc_params.n_layers,
            disc_params.hidden_dim,
            activation=disc_params.activation,
            output_activation='identity',
            spectral_norm=disc_params.spectral_norm,
        )


    def forward(self, obs, action):
        logit = self.disc(torch.cat([obs, action], dim=-1))
        return logit, torch.sigmoid(logit)

    def reward(self, obs, action):
        with torch.no_grad():
            logit, prob = self(obs, action)
            if self.reward_type == 'GAIL':
                reward = -torch.log(prob)
            elif self.reward_type == 'SOFTPLUS':
                reward = -F.softplus(logit)
            elif self.reward_type == 'AIRL':
                reward = -logit
            else:
                assert False
        return reward