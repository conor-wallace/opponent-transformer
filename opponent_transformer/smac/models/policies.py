import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


class NAM(nn.Module):
    def __init__(self, envs):
        super(NAM, self).__init__()

        self.num_agents = len(envs.action_space)
        self.num_opponents = self.num_agents - 1
        self.act_dim = envs.action_space[-1].n
        self.obs_dim = envs.observation_space[-1][0]
        self.opp_act_dim = envs.action_space[0].n
        self.opp_obs_dim = envs.observation_space[0][0]

        self.lstm = nn.LSTM(self.obs_dim + self.act_dim, 128)
        self.fc1 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, self.act_dim)
        self.critic = nn.Linear(128, 1)

    @torch.no_grad()
    def act(self, x, lstm_state, available_actions):
        x = x.unsqueeze(0)
        available_actions = available_actions.unsqueeze(0)

        h, lstm_state = self.lstm(x, lstm_state)
        h = F.relu(self.fc1(h))

        logits = self.actor(h)
        values = self.critic(h)

        logits[available_actions == 0] = -1e10

        policy_probs = F.softmax(logits, dim=-1)
        dist = OneHotCategorical(policy_probs)
        actions = dist.sample()

        actions = actions.squeeze(0)
        values = values.squeeze(0)

        return actions, values, lstm_state

    def get_value(self, x, lstm_state):
        x = x.unsqueeze(0)

        h, lstm_state = self.lstm(x, lstm_state)
        h = F.relu(self.fc1(h))
        values = self.critic(h)
        return values
    
    def evaluate(self, x, lstm_state, action):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        h, lstm_state = self.lstm(x, lstm_state)
        h = F.relu(self.fc1(h))

        logits = self.actor(h)
        values = self.critic(h)

        policy_probs = F.softmax(logits, dim=-1)
        log_probs = torch.sum(torch.log(policy_probs + 1e-20) * action, dim=-1)
        entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-20), dim=-1).mean()
        return log_probs, entropy, values