import torch
import torch.nn as nn

from lapal.utils import utils


class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

class IdentityEncoder(BaseEncoder):
    """
    Non-trainable identity encoder
    """
    def __init__(
        self, input_dim, feature_dim, n_layers=2, hidden_dim=256, 
        activation='relu', output_activation='identity'
    ):
        super().__init__()

        self.feature_dim = input_dim

    def forward(self, x, **kwargs):
        return x

class VectorEncoder(BaseEncoder):
    """
    MLP encoder for vector input
    """
    def __init__(
        self, input_dim, feature_dim, n_layers=2, hidden_dim=256, 
        activation='relu', output_activation='identity',
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.mlp_layers = utils.build_mlp(
            input_dim, 
            feature_dim, 
            n_layers=n_layers, 
            size=hidden_dim, 
            activation=activation,
            output_activation=output_activation
        )

    def forward(self, x):
        h = self.mlp_layers(x)
        return h

class TouchEncoder(VectorEncoder):
    """
    MLP encoder for vector input
    assumes last two dimensions are touch observations which are not encoded
    """
    def __init__(  
        self, input_dim, feature_dim, n_layers=2, hidden_dim=256, 
        activation='relu', output_activation='identity',
    ):
        super().__init__(
            input_dim-2, feature_dim-2, n_layers, hidden_dim, 
            activation, output_activation
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        x, touch = x[:, :-2], x[:, -2:]
        h = self.mlp_layers(x)
        return torch.cat([h, touch], dim=-1)

class ConditionalEncoder(BaseEncoder):
    
    def __init__(self, input_dim, feature_dim, cond_feature_dim, n_layers=2, hidden_dim=256,
        activation='relu', output_activation='tanh'
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.cond_feature_dim = cond_feature_dim

        self.log_std_min = -10
        self.log_std_max = 2

        self.mlp_layers = utils.build_mlp(
            input_dim + cond_feature_dim, 
            feature_dim * 2,            # mu and log_std
            n_layers=n_layers, 
            size=hidden_dim, 
            activation=activation,
            output_activation=output_activation
        )

    def forward(self, x, cond=None):
        x = torch.cat([x, cond], dim=-1)
        h = self.mlp_layers(x)
        mu, log_std = h[:, :self.feature_dim], h[:, self.feature_dim:]
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)
        z = mu + torch.randn_like(std) * std
        # return mu
        return z

    def compute_latent_dist_and_sample(self, x, cond=None):
        x = torch.cat([x, cond], dim=-1)
        h = self.mlp_layers(x)
        mu, log_std = h[:, :self.feature_dim], h[:, self.feature_dim:]
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)
        z = mu + torch.randn_like(std) * std
        outputs = {'z': z, 'mu': mu, 'std': std}
        return outputs

_AVAILABLE_ENCODERS = {
    'identity': IdentityEncoder,
    'vector': VectorEncoder,
    'touch': TouchEncoder,
    'conditional': ConditionalEncoder,
}

def make_encoder(params):
    assert params.type in _AVAILABLE_ENCODERS

    args = [params.input_dim, params.feature_dim]
    if params.type == 'conditional':
        args.append(params.cond_feature_dim)

    kwargs = dict(n_layers=params.n_layers, hidden_dim=params.hidden_dim, 
        activation=params.activation, output_activation=params.output_activation)

    return _AVAILABLE_ENCODERS[params.type](*args, **kwargs)
