import numpy as np
import torch

from .running_mean import RunningMeanStd
from .valuenorm import ValueNorm


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])


class A2CBuffer:
    def __init__(
        self,
        num_minibatches,
        batch_size,
        gamma,
        gae_lambda,
        num_envs,
        num_opponents,
        agent_obs_dim,
        agent_act_dim,
        oppnt_obs_dim,
        oppnt_act_dim,
        num_lstm_layers,
        hidden_size,
        device
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.agent_obs = torch.zeros((num_minibatches, batch_size, num_envs, agent_obs_dim)).to(device)
        self.last_agent_actions = torch.zeros((num_minibatches, batch_size, num_envs, agent_act_dim)).to(device)
        self.agent_actions = torch.zeros((num_minibatches, batch_size, num_envs, agent_act_dim)).to(device)
        self.oppnt_obs = torch.zeros((num_minibatches, batch_size, num_envs, num_opponents, oppnt_obs_dim)).to(device)
        self.oppnt_actions = torch.zeros((num_minibatches, batch_size, num_envs, num_opponents, oppnt_act_dim)).to(device)
        self.logprobs = torch.zeros((num_minibatches, batch_size, num_envs)).to(device)
        self.rewards = torch.zeros((num_minibatches, batch_size, num_envs)).to(device)
        self.dones = torch.zeros((num_minibatches, batch_size, num_envs)).to(device)
        self.values = torch.zeros((num_minibatches, batch_size + 1, num_envs)).to(device)
        self.returns = torch.zeros((num_minibatches, batch_size + 1, num_envs)).to(device)
        self.hidden_states = torch.zeros((num_minibatches, num_lstm_layers, num_envs, hidden_size)).to(device)
        self.cell_states = torch.zeros((num_minibatches, num_lstm_layers, num_envs, hidden_size)).to(device)

        self.running_mean = RunningMeanStd(shape=1, device=device)

    def insert(
        self,
        idx,
        step,
        agent_obs,
        last_agent_actions,
        agent_actions,
        oppnt_obs,
        oppnt_actions,
        agent_rewards,
        agent_values,
    ):
        self.agent_obs[idx, step] = agent_obs
        self.last_agent_actions[idx, step] = last_agent_actions
        self.agent_actions[idx, step] = agent_actions
        self.oppnt_obs[idx, step] = oppnt_obs
        self.oppnt_actions[idx, step] = oppnt_actions
        self.rewards[idx, step] = agent_rewards
        self.values[idx, step] = agent_values

    def compute_returns(self, idx, next_value):
        self.values[idx, -1] = next_value.detach()
        self.values[idx] = self.values[idx] * torch.sqrt(self.running_mean.var) + self.running_mean.mean

        lastgaelam = 0
        for t in reversed(range(self.batch_size)):
            delta = self.rewards[idx, t] + self.gamma * self.values[idx, t + 1] * (1.0 - self.dones[idx, t]) - self.values[idx, t]
            lastgaelam = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[idx, t]) * lastgaelam
            self.returns[idx, t] = lastgaelam + self.values[idx, t]
        
        self.running_mean.update(self.returns[idx, :-1].unsqueeze(-1))

        self.returns[idx] = (self.returns[idx] - self.running_mean.mean) / torch.sqrt(self.running_mean.var)

    def generate_batch(self, idx):
        mb_obs = self.agent_obs[idx].clone()
        mb_last_agent_actions = self.last_agent_actions[idx].clone()
        mb_agent_actions = self.agent_actions[idx].clone()
        mb_opp_obs = self.oppnt_obs[idx].clone()
        mb_opp_actions = self.oppnt_actions[idx].clone()
        mb_returns = self.returns[idx, :-1].clone()
        mb_hidden = self.hidden_states[idx].clone()
        mb_cell = self.cell_states[idx].clone()

        return {
            "agent_obs": mb_obs,
            "last_agent_actions": mb_last_agent_actions,
            "agent_actions": mb_agent_actions,
            "oppnt_obs": mb_opp_obs,
            "oppnt_actions": mb_opp_actions,
            "agent_returns": mb_returns,
            "agent_hidden_states": mb_hidden,
            "agent_cell_states": mb_cell
        }


class PPOBuffer:
    def __init__(
        self,
        args,
        num_agents,
        num_opponents,
        obs_dim,
        act_dim,
        opp_obs_dim,
        opp_act_dim,
        num_rnn_layers,
        hidden_dim,
        device
    ):
        self.episode_length = args.episode_length
        self.num_envs = args.num_envs
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.device = device

        self.num_agents = num_agents
        self.num_opponents = num_opponents
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.opp_act_dim = opp_act_dim
        self.opp_obs_dim = opp_obs_dim
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers

        self.obs = np.zeros((self.episode_length + 1, self.num_envs, self.obs_dim))
        self.actions = np.zeros((self.episode_length + 1, self.num_envs, self.act_dim))
        self.available_actions = np.zeros((self.episode_length + 1, self.num_envs, self.act_dim))
        self.oppnt_obs = np.zeros((self.episode_length + 1, self.num_envs, self.num_opponents, self.opp_obs_dim))
        self.oppnt_actions = np.zeros((self.episode_length + 1, self.num_envs, self.num_opponents, self.opp_act_dim))
        self.oppnt_available_actions = np.zeros((self.episode_length + 1, self.num_envs, self.num_opponents, self.opp_act_dim))
        self.logprobs = np.zeros((self.episode_length + 1, self.num_envs, 1))
        self.rewards = np.zeros((self.episode_length, self.num_envs, 1))
        self.values = np.zeros((self.episode_length + 1, self.num_envs, 1))
        self.returns = np.zeros((self.episode_length + 1, self.num_envs, 1))
        self.rnn_states = np.zeros((self.episode_length + 1, self.num_envs, self.num_rnn_layers, self.hidden_dim))
        self.masks = np.ones((self.episode_length + 1, self.num_envs, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.oppnt_rnn_states = np.zeros((self.episode_length + 1, self.num_envs, self.num_opponents, self.num_rnn_layers, 64))
        self.oppnt_masks = np.ones((self.episode_length + 1, self.num_envs, self.num_opponents, 1), dtype=np.float32)
        self.oppnt_bad_masks = np.ones_like(self.oppnt_masks)
        self.oppnt_active_masks = np.ones_like(self.oppnt_masks)

        self.value_normalizer = ValueNorm(1).to(self.device)
        self.step = 0

    def insert(
        self,
        obs,
        actions,
        logprobs,
        oppnt_obs,
        oppnt_actions,
        rewards,
        values,
        available_actions,
        oppnt_available_actions,
        rnn_states,
        masks,
        active_masks,
        bad_masks,
        oppnt_rnn_states,
        oppnt_masks,
        oppnt_active_masks,
        oppnt_bad_masks
    ):
        # Agent storage
        self.obs[self.step + 1] = obs.detach().cpu().numpy().copy()
        self.actions[self.step + 1] = actions.detach().cpu().numpy().copy()
        self.logprobs[self.step + 1] = logprobs.detach().cpu().numpy().copy()
        self.available_actions[self.step + 1] = available_actions.detach().cpu().numpy().copy()
        self.rewards[self.step] = rewards.detach().cpu().numpy().copy()
        self.values[self.step] = values.detach().cpu().numpy().copy()
        self.rnn_states[self.step + 1] = rnn_states.detach().cpu().numpy().copy()
        self.masks[self.step + 1] = masks.detach().cpu().numpy().copy()
        self.active_masks[self.step + 1] = active_masks.detach().cpu().numpy().copy()
        self.bad_masks[self.step + 1] = bad_masks.detach().cpu().numpy().copy()

        # Opponent storage
        self.oppnt_obs[self.step + 1] = oppnt_obs.detach().cpu().numpy().copy()
        self.oppnt_actions[self.step] = oppnt_actions.detach().cpu().numpy().copy()
        self.oppnt_available_actions[self.step + 1] = oppnt_available_actions.detach().cpu().numpy().copy()
        self.oppnt_rnn_states[self.step + 1] = oppnt_rnn_states.detach().cpu().numpy().copy()
        self.oppnt_masks[self.step + 1] = oppnt_masks.detach().cpu().numpy().copy()
        self.oppnt_active_masks[self.step + 1] = oppnt_active_masks.detach().cpu().numpy().copy()
        self.oppnt_bad_masks[self.step + 1] = oppnt_bad_masks.detach().cpu().numpy().copy()

        self.step = (self.step + 1) % self.episode_length

    def compute_returns(self, next_value):
        self.values[-1] = next_value.detach().cpu().numpy().copy()
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = self.rewards[step] + self.gamma * self.value_normalizer.denormalize(
                self.values[step + 1]) * self.masks[step + 1] \
                    - self.value_normalizer.denormalize(self.values[step])
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_normalizer.denormalize(self.values[step])
 
    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.available_actions[0] = self.available_actions[-1].copy()

        self.oppnt_obs[0] = self.oppnt_obs[-1].copy()
        self.oppnt_rnn_states[0] = self.oppnt_rnn_states[-1].copy()
        self.oppnt_masks[0] = self.oppnt_masks[-1].copy()
        self.oppnt_bad_masks[0] = self.oppnt_bad_masks[-1].copy()
        self.oppnt_active_masks[0] = self.oppnt_active_masks[-1].copy()
        self.oppnt_available_actions[0] = self.oppnt_available_actions[-1].copy()

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        batch_size = self.num_envs * self.episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        obs = _cast(self.obs[:-1])
        last_actions = _cast(self.actions[1:])
        actions = _cast(self.actions[:-1])
        logprobs = _cast(self.logprobs)
        advantages = _cast(advantages)
        values = _cast(self.values[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        # rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            obs_batch = []
            last_actions_batch = []
            actions_batch = []
            available_actions_batch = []
            values_batch = []
            return_batch = []
            rnn_states_batch = []
            masks_batch = []
            active_masks_batch = []
            old_logprobs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                last_actions_batch.append(last_actions[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                values_batch.append(values[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_logprobs_batch.append(logprobs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)   
            obs_batch = np.stack(obs_batch, axis=1)
            last_actions_batch = np.stack(last_actions_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            available_actions_batch = np.stack(available_actions_batch, axis=1)
            values_batch = np.stack(values_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_logprobs_batch = np.stack(old_logprobs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = _flatten(L, N, obs_batch)
            last_actions_batch = _flatten(L, N, last_actions_batch)
            actions_batch = _flatten(L, N, actions_batch)
            available_actions_batch = _flatten(L, N, available_actions_batch)
            values_batch = _flatten(L, N, values_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_logprobs_batch = _flatten(L, N, old_logprobs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield obs_batch, rnn_states_batch, actions_batch, last_actions_batch, \
                  values_batch, return_batch, masks_batch, active_masks_batch, old_logprobs_batch,\
                  adv_targ, available_actions_batch