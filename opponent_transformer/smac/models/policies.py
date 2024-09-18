import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import Categorical


# class NAM(nn.Module):
#     def __init__(self, envs):
#         super(NAM, self).__init__()

#         self.num_agents = len(envs.action_space)
#         self.num_opponents = self.num_agents - 1
#         self.act_dim = envs.action_space[-1].n
#         self.obs_dim = envs.observation_space[-1][0]
#         self.opp_act_dim = envs.action_space[0].n
#         self.opp_obs_dim = envs.observation_space[0][0]

#         self.lstm = nn.LSTM(self.obs_dim + self.act_dim, 128)
#         self.fc1 = nn.Linear(128, 128)
#         self.actor = nn.Linear(128, self.act_dim)
#         self.critic = nn.Linear(128, 1)

#     @torch.no_grad()
#     def act(self, x, lstm_state, available_actions):
#         x = x.unsqueeze(0)
#         available_actions = available_actions.unsqueeze(0)

#         h, lstm_state = self.lstm(x, lstm_state)
#         h = F.relu(self.fc1(h))

#         logits = self.actor(h)
#         values = self.critic(h)

#         logits[available_actions == 0] = -1e10

#         policy_probs = F.softmax(logits, dim=-1)
#         dist = OneHotCategorical(policy_probs)
#         actions = dist.sample()

#         actions = actions.squeeze(0)
#         values = values.squeeze(0)

#         return actions, values, lstm_state

#     def get_value(self, x, lstm_state):
#         x = x.unsqueeze(0)

#         h, lstm_state = self.lstm(x, lstm_state)
#         h = F.relu(self.fc1(h))
#         values = self.critic(h)
#         return values
    
#     def evaluate(self, x, lstm_state, action):
#         if len(x.size()) == 2:
#             x = x.unsqueeze(0)
#         h, lstm_state = self.lstm(x, lstm_state)
#         h = F.relu(self.fc1(h))

#         logits = self.actor(h)
#         values = self.critic(h)

#         policy_probs = F.softmax(logits, dim=-1)
#         log_probs = torch.sum(torch.log(policy_probs + 1e-20) * action, dim=-1)
#         entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-20), dim=-1).mean()
#         return log_probs, entropy, values


#TODO: implement rmappo training for single agent
class RPPO(nn.Module):
    def __init__(
        self,
        num_agents: int,
        num_opponents: int,
        act_dim: int,
        obs_dim: int,
        opp_act_dim: int,
        opp_obs_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = None,
        num_rnn_layers: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(RPPO, self).__init__()

        self.num_agents = num_agents
        self.num_opponents = num_opponents
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.opp_act_dim = opp_act_dim
        self.opp_obs_dim = opp_obs_dim

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.embedding_dim = self.obs_dim if embedding_dim is None else embedding_dim
    
        self.feature_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_rnn_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.actor = Categorical(hidden_dim, self.act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self.to(device)

    def forward(self, x, hxs, masks):
        # Base
        x = self.feature_norm(x)
        x = self.fc1(x)
        x = self.fc2(x)

        # RNN
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (hxs * masks.repeat(1, 1).unsqueeze(-1)).transpose(0, 1).contiguous()
            )
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

        # Output norm
        x = self.norm(x)

        return x, hxs

    @torch.no_grad()
    def act(self, x, hxs, masks, available_actions, deterministic=False):
        x = torch.from_numpy(x).to(**self.tpdv)
        hxs = torch.from_numpy(hxs).to(**self.tpdv)
        masks = torch.from_numpy(masks).to(**self.tpdv)
        available_actions = torch.from_numpy(available_actions).to(**self.tpdv)

        x, hxs = self(x, hxs, masks)

        action_logits = self.actor(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample() 
        action_log_probs = action_logits.log_probs(actions)

        values = self.critic(x)

        return actions, action_log_probs, values, hxs

    def get_values(self, x, hxs, masks):
        x = torch.from_numpy(x).to(**self.tpdv)
        hxs = torch.from_numpy(hxs).to(**self.tpdv)
        masks = torch.from_numpy(masks).to(**self.tpdv)

        x, hxs = self(x, hxs, masks)

        values = self.critic(x)

        return values

    def evaluate(self, x, hxs, masks, actions, available_actions, active_masks):
        x = torch.from_numpy(x).to(**self.tpdv)
        hxs = torch.from_numpy(hxs).to(**self.tpdv)
        masks = torch.from_numpy(masks).to(**self.tpdv)
        actions = torch.from_numpy(actions).to(**self.tpdv).argmax(-1)
        available_actions = torch.from_numpy(available_actions).to(**self.tpdv)
        active_masks = torch.from_numpy(active_masks).to(**self.tpdv)

        x, hxs = self(x, hxs, masks)

        action_logits = self.actor(x, available_actions)
        action_log_probs = action_logits.log_probs(actions)
        if active_masks is not None:
            dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()

        values = self.critic(x)

        return values, action_log_probs, dist_entropy
