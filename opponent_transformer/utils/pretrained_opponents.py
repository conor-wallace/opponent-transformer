import os
import copy
import numpy as np
import itertools
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, OneHotCategorical


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs


class RMAPPONet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(RMAPPONet, self).__init__()

        self.tpdv = dict(dtype=torch.float32, device='cpu')

        self.feature_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden)
        )
        self.rnn = nn.GRU(hidden, hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.Linear(hidden, output_dim)
    
    def forward(self, x, hxs, mask, available_actions):
        x = torch.from_numpy(x).to(**self.tpdv).unsqueeze(0)
        hxs = torch.from_numpy(hxs).to(**self.tpdv).unsqueeze(0)
        mask = torch.from_numpy(mask).to(**self.tpdv).unsqueeze(0)
        available_actions = torch.from_numpy(available_actions).to(**self.tpdv).unsqueeze(0)

        x = self.feature_norm(x)
        x = self.fc1(x)
        x = self.fc2(x)

        batch_size = hxs.shape[0]

        x, hxs = self.rnn(
            x.unsqueeze(0),
            (hxs * mask.repeat(1, 1).unsqueeze(-1)).transpose(0, 1).contiguous()
        )
        x = x.squeeze(0)
        hxs = hxs.transpose(0, 1)

        x = self.norm(x)

        logits = self.act(x)
        if available_actions is not None:
            logits[available_actions == 0] = -1e10
        action = onehot_from_logits(logits)
        return action, hxs


class OpponentPolicy:
    def __init__(self, env_name: str, map_name: str, envs):
        self.pretrained_dir = os.path.join('../../opponent_transformer/', env_name, map_name)

        pretrained_models = os.listdir(self.pretrained_dir)
        self.num_opponent_policies = pretrained_models