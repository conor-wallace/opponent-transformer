import torch
import numpy as np
from collections import defaultdict

from opponent_transformer.utils.util import check, get_shape_from_obs_space, get_shape_from_act_space, pad_sequence

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:])


class TransformerReplayBuffer(object):
    def __init__(self, args, num_agents, obs_space, act_space):
        self.episode_length = args.episode_length
        self.max_length = args.max_length
        self.n_rollout_threads = args.n_rollout_threads
        self.rnn_hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.act_dim = act_space.n

        obs_shape = get_shape_from_obs_space(obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        
        self.action_dim = act_space.n
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, self.action_dim), dtype=np.float32)
        else:
            self.available_actions = None

        self.act_dim = act_space.n
        self.act_shape = get_shape_from_act_space(act_space)
        self.actions = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, self.act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        attention_masks = []
        timesteps = []
        episode_masks_padded = np.concatenate((np.zeros((self.max_length-1)), np.ones(self.episode_length)))
        episode_timesteps_padded = np.concatenate((np.zeros((self.max_length-1)), np.arange(self.episode_length)))

        for i in range(self.episode_length):
            attention_masks.append(episode_masks_padded[i:i+self.max_length])
            timesteps.append(episode_timesteps_padded[i:i+self.max_length])

        attention_masks = np.array(attention_masks, dtype=np.long)
        timesteps = np.array(timesteps, dtype=np.long)
        self.attention_masks = np.repeat(attention_masks.reshape(self.episode_length, 1, self.max_length), self.n_rollout_threads, axis=1)
        self.timesteps = np.repeat(timesteps.reshape(self.episode_length, 1, self.max_length), self.n_rollout_threads, axis=1)

        self.step = 0

    def insert(self, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):

        self.obs[self.step + 1] = obs.copy()
        if rnn_states is not None:
            self.rnn_states[self.step + 1] = rnn_states.copy()
        if rnn_states_critic is not None:
            self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step + 1] = actions.copy()
        if action_log_probs is not None:
            self.action_log_probs[self.step] = action_log_probs.copy()
        if value_preds is not None:
            self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        if masks is not None:
            self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length
    
    def after_update(self):
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[
                            step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch))
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs[:-1, ind])
                obs_batch.append(self.obs[:-1, ind])
                rnn_states_batch.append(self.rnn_states[0:1, ind])
                rnn_states_critic_batch.append(self.rnn_states_critic[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(self.available_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, -1) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch, 1).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch, 1).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, oppnt_obs, oppnt_actions, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
        episode_start_indices = np.array([i * episode_length for i in range(n_rollout_threads)])

        if len(self.obs.shape) > 3:
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            obs = _cast(self.obs[:-1])
            oppnt_obs = _cast(oppnt_obs[:-1])

        prev_actions = _cast(self.actions[:-1])
        actions = _cast(self.actions[1:])
        oppnt_actions = _cast(oppnt_actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        rewards = np.concatenate((np.zeros((1, n_rollout_threads, 1), dtype=np.float32), self.rewards))
        rewards = _cast(rewards[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        attention_masks = _cast(self.attention_masks)
        timesteps = _cast(self.timesteps)
        rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states_critic.shape[2:])

        # print("Observations shape: ", obs.shape)
        # print("Opponent observations shape: ", oppnt_obs.shape)
        # print("Prev actions shape: ", prev_actions.shape)
        # print("Actions shape: ", actions.shape)
        # print("Opponent actions shape: ", oppnt_actions.shape)
        # print("Action logprobs shape: ", action_log_probs.shape)
        # print("Advantages shape: ", advantages.shape)
        # print("Values shape: ", value_preds.shape)
        # print("Returns shape: ", returns.shape)
        # print("Masks shape: ", masks.shape)
        # print("Active masks shape: ", active_masks.shape)
        # print("Attention masks shape: ", attention_masks.shape)
        # print("Timesteps shape: ", timesteps.shape)
        # print("RNN states shape: ", rnn_states.shape)
        # print("RNN states critic shape: ", rnn_states_critic.shape)

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            obs_batch = []
            oppnt_obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            prev_actions_batch = []
            actions_batch = []
            oppnt_actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            obs_seq_batch = []
            action_seq_batch = []
            reward_seq_batch = []
            timesteps_batch = []
            attention_masks_batch = []
            oppnt_obs_seq_batch = []
            oppnt_action_seq_batch = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                obs_batch.append(obs[ind:ind+data_chunk_length])
                oppnt_obs_batch.append(oppnt_obs[ind:ind+data_chunk_length])
                prev_actions_batch.append(prev_actions[ind:ind+data_chunk_length])
                actions_batch.append(actions[ind:ind+data_chunk_length])
                oppnt_actions_batch.append(oppnt_actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

                # Generate padded sequences
                obs_seq, action_seq, reward_seq, oppnt_obs_seq, oppnt_action_seq = [], [], [], [], []
                for ind_i in range(ind, ind+data_chunk_length):
                    episode_start_ind = episode_start_indices[np.argwhere(ind_i >= episode_start_indices)[-1]][0]
                    obs_seq_padded = pad_sequence(obs[episode_start_ind:ind_i], self.max_length)
                    action_seq_padded = pad_sequence(prev_actions[episode_start_ind:ind_i], self.max_length)
                    reward_seq_padded = pad_sequence(rewards[episode_start_ind:ind_i], self.max_length)
                    oppnt_obs_seq_padded = pad_sequence(oppnt_obs[episode_start_ind:ind_i], self.max_length)
                    oppnt_action_seq_padded = pad_sequence(oppnt_actions[episode_start_ind:ind_i], self.max_length)

                    obs_seq.append(obs_seq_padded)
                    action_seq.append(action_seq_padded)
                    reward_seq.append(reward_seq_padded)
                    oppnt_obs_seq.append(oppnt_obs_seq_padded)
                    oppnt_action_seq.append(oppnt_action_seq_padded)

                obs_seq = np.array(obs_seq)
                action_seq = np.array(action_seq)
                reward_seq = np.array(reward_seq)
                oppnt_obs_seq = np.array(oppnt_obs_seq)
                oppnt_action_seq = np.array(oppnt_action_seq)

                obs_seq_batch.append(obs_seq)
                action_seq_batch.append(action_seq)
                reward_seq_batch.append(reward_seq)
                timesteps_batch.append(timesteps[ind:ind+data_chunk_length])
                attention_masks_batch.append(attention_masks[ind:ind+data_chunk_length])
                oppnt_obs_seq_batch.append(oppnt_obs_seq)
                oppnt_action_seq_batch.append(oppnt_action_seq)

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            obs_batch = np.stack(obs_batch)
            oppnt_obs_batch = np.stack(oppnt_obs_batch)
            prev_actions_batch = np.stack(prev_actions_batch)
            actions_batch = np.stack(actions_batch)
            oppnt_actions_batch = np.stack(oppnt_actions_batch)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch)
            value_preds_batch = np.stack(value_preds_batch)
            return_batch = np.stack(return_batch)
            masks_batch = np.stack(masks_batch)
            active_masks_batch = np.stack(active_masks_batch)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch)
            adv_targ = np.stack(adv_targ)

            obs_seq_batch = np.stack(obs_seq_batch)
            action_seq_batch = np.stack(action_seq_batch)
            reward_seq_batch = np.stack(reward_seq_batch)
            timesteps_batch = np.stack(timesteps_batch)
            attention_masks_batch = np.stack(attention_masks_batch)
            oppnt_obs_seq_batch = np.stack(oppnt_obs_seq_batch)
            oppnt_action_seq_batch = np.stack(oppnt_action_seq_batch)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[2:])

            # print("Observations batch shape: ", obs_batch.shape)
            # print("Opponent observations batch shape: ", oppnt_obs_batch.shape)
            # print("Prev actions batch shape: ", prev_actions_batch.shape)
            # print("Actions batch shape: ", actions_batch.shape)
            # print("Opponent actions batch shape: ", oppnt_actions_batch.shape)
            # print("Action logprobs batch shape: ", old_action_log_probs_batch.shape)
            # print("Advantages batch shape: ", adv_targ.shape)
            # print("Values batch shape: ", value_preds_batch.shape)
            # print("Returns batch shape: ", return_batch.shape)
            # print("Masks batch shape: ", masks_batch.shape)
            # print("Active masks batch shape: ", active_masks_batch.shape)
            # print("RNN states batch shape: ", rnn_states_batch.shape)
            # print("RNN states critic batch shape: ", rnn_states_critic_batch.shape)
            # print("Obs seq batch shape: ", obs_seq_batch.shape)
            # print("Action seq batch shape: ", action_seq_batch.shape)
            # print("Reward seq batch shape: ", reward_seq_batch.shape)
            # print("Timesteps batch shape: ", timesteps_batch.shape)
            # print("Attention masks batch shape: ", attention_masks_batch.shape)

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = _flatten(L, N, obs_batch)
            oppnt_obs_batch = _flatten(L, N, oppnt_obs_batch)
            prev_actions_batch = _flatten(L, N, prev_actions_batch)
            actions_batch = _flatten(L, N, actions_batch)
            oppnt_actions_batch = _flatten(L, N, oppnt_actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            obs_seq_batch = _flatten(L, N, obs_seq_batch)
            action_seq_batch = _flatten(L, N, action_seq_batch)
            reward_seq_batch = _flatten(L, N, reward_seq_batch)
            timesteps_batch = _flatten(L, N, timesteps_batch)
            attention_masks_batch = _flatten(L, N, attention_masks_batch)
            oppnt_obs_seq_batch = _flatten(L, N, oppnt_obs_seq_batch)
            oppnt_action_seq_batch = _flatten(L, N, oppnt_action_seq_batch)

            action_seq_batch = torch.functional.F.one_hot(torch.from_numpy(action_seq_batch).long(), self.action_dim).numpy().squeeze(2)

            # print("Observations batch flattened shape: ", obs_batch.shape)
            # print("Opponent observations batch flattened shape: ", oppnt_obs_batch.shape)
            # print("Prev actions batch flattened shape: ", prev_actions_batch.shape)
            # print("Actions batch flattened shape: ", actions_batch.shape)
            # print("Opponent actions batch flattened shape: ", oppnt_actions_batch.shape)
            # print("Action logprobs batch flattened shape: ", old_action_log_probs_batch.shape)
            # print("Advantages batch flattened shape: ", adv_targ.shape)
            # print("Values batch flattened shape: ", value_preds_batch.shape)
            # print("Returns batch flattened shape: ", return_batch.shape)
            # print("Masks batch flattened shape: ", masks_batch.shape)
            # print("Active masks batch flattened shape: ", active_masks_batch.shape)
            # print("RNN states batch flattened shape: ", rnn_states_batch.shape)
            # print("RNN states critic batch flattened shape: ", rnn_states_critic_batch.shape)
            # print("Obs seq batch flattened shape: ", obs_seq_batch.shape)
            # print("Action seq batch flattened shape: ", action_seq_batch.shape)
            # print("Reward seq batch flattened shape: ", reward_seq_batch.shape)
            # print("Timesteps batch flattened shape: ", timesteps_batch.shape)
            # print("Attention masks batch flattened shape: ", attention_masks_batch.shape)
            # print("Opponent obs seq batch flattened shape: ", oppnt_obs_seq_batch.shape)
            # print("Opponent action seq batch flattened shape: ", oppnt_action_seq_batch.shape)

            yield (obs_batch, rnn_states_batch, 
            rnn_states_critic_batch, prev_actions_batch, actions_batch, 
            value_preds_batch, return_batch, masks_batch, active_masks_batch, 
            old_action_log_probs_batch, adv_targ, available_actions_batch,
            obs_seq_batch, action_seq_batch, reward_seq_batch, timesteps_batch, attention_masks_batch,
            oppnt_obs_seq_batch, oppnt_action_seq_batch)
