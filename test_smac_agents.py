import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from tqdm import tqdm

from opponent_transformer.envs import DummyVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv, StarCraft2Env
from opponent_transformer.smac.pretrained_opponents.pretrained_1c3s5z_opponents import get_opponent_actions


def make_env(env_id, rank):
    def init_env():
        env = StarCraft2Env(
            map_name=env_id
        )
        env.seed(50000 + rank * 10000)
        return env
    
    return init_env


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs


class PPONet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(PPONet, self).__init__()

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
    
    def forward(self, x, hxs, masks, available_actions=None):
        # print("X shape: ", x.shape)
        # print("Hxs shape: ", hxs.shape)
        # print("Masks shape: ", masks.shape)
        # print("Available actions shape: ", available_actions.shape)

        x = self.feature_norm(x)
        x = self.fc1(x)
        x = self.fc2(x)

        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0),
                              (hxs * masks.repeat(1, 1).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(1, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)

        logits = self.act(x)
        if available_actions is not None:
            logits[available_actions == 0] = -1e10
        action = onehot_from_logits(logits)
        return action, hxs

# TODO: Test with NAM Agent (basically just random policy)
class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()

        self.num_agents = len(envs.action_space)
        self.act_dim = envs.action_space[-1].n
        self.obs_dim = envs.observation_space[-1][0]
        self.opp_act_dim = sum([envs.action_space[i].n for i in range(self.num_agents)])
        self.opp_obs_dim = sum([envs.observation_space[i][0] for i in range(self.num_agents)])
        
        self.lstm = nn.LSTM(self.obs_dim + self.act_dim, 128)
        self.fc1 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, self.act_dim)
        self.critic = nn.Linear(128, 1)

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
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        h, lstm_state = self.lstm(x, lstm_state)
        h = F.relu(self.fc1(h))
        value = self.critic(h)
        return value
    
    def evaluate(self, x, lstm_state, action):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        h, lstm_state = self.lstm(x, lstm_state)
        h = F.relu(self.fc1(h))
        logits = F.softmax(self.actor(h), dim=-1)
        values = self.critic(h)
        log_probs = torch.sum(torch.log(logits + 1e-20) * action, dim=-1)
        entropy = -torch.sum(logits * torch.log(logits + 1e-20), dim=-1).mean()
        return log_probs, entropy, values


def baseline():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)

    num_envs = 4
    num_agents = 9
    num_oppnts = num_agents - 1
    num_episodes = 10
    episode_length = 400
    env_fns = [make_env('1c3s5z', i) for i in range(num_envs)]

    if num_envs == 1:
        env = ShareDummyVecEnv(env_fns)
    else:
        env = ShareDummyVecEnv(env_fns)

    model = PPONet(310, 15, 64)
    params_dir = 'opponent_transformer/smac/pretrained_opponents/pretrained_parameters/1c3s5z'
    params_path = os.path.join(params_dir, 'rmappo_agent_0.pt')
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
    model.load_state_dict(modified_save_dict)
    model.eval()

    agent = Agent(env)

    step = 0

    device = 'cpu'
    obs, share_obs, available_actions = env.reset()

    oppnt_hidden_states = torch.zeros((num_envs, num_oppnts, 1, 64), dtype=torch.float32)
    oppnt_masks = torch.ones((num_envs, num_oppnts, 1))

    agent_hidden_states = torch.zeros(agent.lstm.num_layers, num_envs, agent.lstm.hidden_size).to(device)
    agent_cell_states = torch.zeros(agent.lstm.num_layers, num_envs, agent.lstm.hidden_size).to(device)

    last_action = torch.zeros(num_envs, agent.act_dim).to(device, dtype=torch.float32)

    last_battles_game = np.zeros(num_envs, dtype=np.float32)
    last_battles_won = np.zeros(num_envs, dtype=np.float32)

    tasks = [0] * num_envs

    episode = 0
    battles_won = 0

    episode_rewards = []
    one_episode_rewards = []

    while True:
        agent_obs = torch.from_numpy(obs[:, -1])
        oppnt_obs = torch.from_numpy(obs[:, :-1])
        agent_available_actions = torch.from_numpy(available_actions[:, -1])
        oppnt_available_actions = torch.from_numpy(available_actions[:, :-1])

        oppnt_actions = []
        for id in range(num_envs):
            oppnt_actions_id, oppnt_hidden_states[id] = get_opponent_actions(
                oppnt_obs[id], oppnt_hidden_states[id], oppnt_masks[id], oppnt_available_actions[id], tasks[id]
            )
            oppnt_actions.append(oppnt_actions_id.unsqueeze(0))
        oppnt_actions = torch.cat(oppnt_actions)

        x = torch.cat((agent_obs, last_action), dim=-1)
        agent_actions, agent_values, agent_lstm_states = agent.act(
            x,
            (agent_hidden_states, agent_cell_states),
            agent_available_actions
        )
        agent_hidden_states, agent_cell_states = agent_lstm_states

        agent_actions = agent_actions.unsqueeze(1)

        actions = torch.cat((oppnt_actions, agent_actions), dim=1)
        actions = actions.argmax(-1)

        obs, share_obs, rewards, dones, infos, available_actions = env.step(actions)
        one_episode_rewards.append(rewards)
        dones_env = np.all(dones, axis=1)

        agent_hidden_states[:, dones_env == True] = torch.zeros((agent.lstm.num_layers, (dones_env == True).sum(), agent.lstm.hidden_size), dtype=torch.float32).to(device)
        agent_cell_states[:, dones_env == True] = torch.zeros((agent.lstm.num_layers, (dones_env == True).sum(), agent.lstm.hidden_size), dtype=torch.float32).to(device)

        oppnt_hidden_states[dones_env == True] = torch.zeros(((dones_env == True).sum(), num_agents - 1, 1, 64), dtype=torch.float32).to(device)
        oppnt_masks = torch.ones((num_envs, num_oppnts, 1), dtype=torch.float32)
        oppnt_masks[dones_env == True] = torch.zeros(((dones_env == True).sum(), num_oppnts, 1), dtype=torch.float32)

        for i in range(num_envs):
            if dones_env[i]:
                episode += 1
                episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                one_episode_rewards = []
                if infos[i][0]['won']:
                    battles_won += 1

        if episode >= num_episodes:
            episode_rewards = np.array(episode_rewards)
            win_rate = battles_won / episode
            print("eval win rate is {}.".format(win_rate))
            print("eval average cumulative reward is {}.".format(episode_rewards.mean()))
            break

    # battles_won = []
    # battles_game = []
    # incre_battles_won = []
    # incre_battles_game = []

    # eval_battles_won = 0

    # for i, info in enumerate(infos):
    #     if 'battles_won' in info[0].keys():
    #         battles_won.append(info[0]['battles_won'])
    #         incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
    #     if 'battles_game' in info[0].keys():
    #         battles_game.append(info[0]['battles_game'])
    #         incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

    # incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
    # print("Incre Win Rate: ", incre_win_rate)


if __name__ == "__main__":
    baseline()
