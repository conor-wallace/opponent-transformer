import torch

from .running_mean import RunningMeanStd


class Buffer:
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
        self.hidden_states = torch.zeros((num_minibatches, 1, num_envs, hidden_size)).to(device)
        self.cell_states = torch.zeros((num_minibatches, 1, num_envs, hidden_size)).to(device)

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
        for t in reversed(range(args.batch_size)):
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