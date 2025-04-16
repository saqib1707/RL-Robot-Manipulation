import torch
import torch.nn as nn

from lapal.utils import utils

class BaseDecoder(nn.Module):
    def __init__(self):
        super().__init__()


class IdentityDecoder(BaseDecoder):
    """
    Non-trainable identity decoder
    """
    def __init__(
        self, output_dim, feature_dim, n_layers=2, hidden_dim=256, 
        activation='relu', output_activation='identity',
    ):

        super().__init__()

        self.feature_dim = output_dim

    def forward(self, x, **kwargs):
        return x

class VectorDecoder(BaseDecoder):
    """
    MLP decoder for vector output
    """
    def __init__(
        self, output_dim, feature_dim, n_layers=2, hidden_dim=256, 
        activation='relu', output_activation='identity',
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.mlp_layers = utils.build_mlp(
            feature_dim, 
            output_dim, 
            n_layers=n_layers, 
            size=hidden_dim, 
            activation=activation,
            output_activation=output_activation
        )

    def forward(self, x):
        x = self.mlp_layers(x)
        return x


class ConditionalDecoder(BaseDecoder):
    def __init__(self, output_dim, feature_dim, cond_feature_dim, 
        n_layers=2, hidden_dim=256, activation='relu', output_activation='identity',
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.cond_feature_dim = cond_feature_dim

        self.mlp_layers = utils.build_mlp(
            feature_dim + cond_feature_dim, 
            output_dim, 
            n_layers=n_layers, 
            size=hidden_dim, 
            activation=activation,
            output_activation=output_activation
        )

    def forward(self, x, cond=None):
        x = torch.cat([x, cond], dim=-1)
        h = self.mlp_layers(x)
        return h


_AVAILABLE_DECODERS = {
    'identity': IdentityDecoder,
    'vector': VectorDecoder,
    'conditional': ConditionalDecoder,
}

def make_decoder(params):
    assert params.type in _AVAILABLE_DECODERS

    args = [params.input_dim, params.feature_dim]
    if params.type == 'conditional':
        args.append(params.cond_feature_dim)
        
    kwargs = dict(n_layers=params.n_layers, hidden_dim=params.hidden_dim, 
        activation=params.activation, output_activation=params.output_activation)

    return _AVAILABLE_DECODERS[params.type](*args, **kwargs)


