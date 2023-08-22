# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import init


class ActorCritic(nn.Module):
    def __init__(self, hidden_size):
        """
        Model constructor.

        Args:
            hidden_size (int): Hidden size of LSTM cell.
        """
        super(ActorCritic, self).__init__()
        self.rgb_state_size = (3, 64, 64)      # Observation shape
        self.action_size = 7
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        
        # The archtecture is adapted from Sim2Real (Rusu et. al 2016)
        self.conv1 = nn.Conv2d(self.rgb_state_size[0], 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.fc1 = nn.Linear(1152, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc_actor1 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor2 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor3 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor4 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor5 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor6 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor7 = nn.Linear(hidden_size, self.action_size)
        self.fc_critic = nn.Linear(hidden_size, 1)
        # Orthogonal weight initialization
        for name, p in self.named_parameters():
            if "weight" in name:
                init.orthogonal_(p)
            elif "bias" in name:
                init.constant_(p, 0)

    def forward(self, rgb_state, h):
        """
        Forward method of the nn.Module.

        Args:
            rgb_state (torch.Tensor): state observation from the environment.
            h (tuple): (hidden state, cell state).

        Returns:
            tuple of tensors, tensor, tuple of tensors: policies for each joint, value of the value state function and LSTM hidden state and cell state.
        """
        x = self.relu(self.conv1(rgb_state))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        h = self.lstm(x, h)
        x = h[0]
        policy1 = self.softmax(self.fc_actor1(x)).clamp(max=1 - 1e-20)
        policy2 = self.softmax(self.fc_actor2(x)).clamp(max=1 - 1e-20)
        policy3 = self.softmax(self.fc_actor3(x)).clamp(max=1 - 1e-20)
        policy4 = self.softmax(self.fc_actor4(x)).clamp(max=1 - 1e-20)
        policy5 = self.softmax(self.fc_actor5(x)).clamp(max=1 - 1e-20)
        policy6 = self.softmax(self.fc_actor6(x)).clamp(max=1 - 1e-20)
        policy7 = self.softmax(self.fc_actor7(x)).clamp(max=1 - 1e-20)
        V = self.fc_critic(x)
        return (policy1, policy2, policy3, policy4, policy5, policy6, policy7), V, h
