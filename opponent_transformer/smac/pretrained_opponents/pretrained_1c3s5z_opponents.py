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

params_dir = '../../opponent_transformer/smac/pretrained_opponents/pretrained_parameters/1c3s5z'
num_rmappo_models = 1
rmappo_agents = [RMAPPONet(310, 15, 64) for _ in range(num_rmappo_models)]
for i in range(num_rmappo_models):
    print("CWD: ", os.getcwd())
    params_path = os.path.join(params_dir, 'rmappo_agent_' + str(i) + '.pt')
    save_dict = torch.load(params_path, map_location='cpu')
    modified_save_dict = {
        'feature_norm.weight': save_dict['base.feature_norm.weight'],
        'feature_norm.bias': save_dict['base.feature_norm.bias'],
        'fc1.0.bias': save_dict['base.mlp.fc1.0.bias'],
        'fc1.0.weight': save_dict['base.mlp.fc1.0.weight'],
        'fc1.2.weight': save_dict['base.mlp.fc1.2.weight'],
        'fc1.2.bias': save_dict['base.mlp.fc1.2.bias'],
        'fc2.0.weight': save_dict['base.mlp.fc2.0.0.weight'],
        'fc2.0.bias': save_dict['base.mlp.fc2.0.0.bias'],
        'fc2.2.weight': save_dict['base.mlp.fc2.0.2.weight'],
        'fc2.2.bias': save_dict['base.mlp.fc2.0.2.bias'],
        'rnn.weight_ih_l0': save_dict['rnn.rnn.weight_ih_l0'],
        'rnn.weight_hh_l0': save_dict['rnn.rnn.weight_hh_l0'],
        'rnn.bias_ih_l0': save_dict['rnn.rnn.bias_ih_l0'],
        'rnn.bias_hh_l0': save_dict['rnn.rnn.bias_hh_l0'],
        'norm.weight': save_dict['rnn.norm.weight'],
        'norm.bias': save_dict['rnn.norm.bias'],
        'act.weight': save_dict['act.action_out.linear.weight'],
        'act.bias': save_dict['act.action_out.linear.bias']
    }
    rmappo_agents[i].load_state_dict(modified_save_dict)
    rmappo_agents[i].eval()


opp = []
opp += rmappo_agents

index = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0]
])

pretrained_agents = []

for i in range(len(index)):
    pretrained_agents.append([
        opp[index[i][0]],
        opp[index[i][1]],
        opp[index[i][2]],
        opp[index[i][3]],
        opp[index[i][4]],
        opp[index[i][5]],
        opp[index[i][6]],
        opp[index[i][7]]
    ])


@torch.no_grad()
def get_opponent_actions(obs, hxs, masks, available_actions, task_id):
    actions = []
    hxs_out = []

    for i in range(len(obs)):
        actions_i, hxs_out_i = pretrained_agents[task_id][i](obs[i], hxs[i], masks[i], available_actions[i])
        actions.append(actions_i)
        hxs_out.append(hxs_out_i)

    actions = torch.cat(actions)
    hxs_out = torch.cat(hxs_out)
    return actions, hxs_out
