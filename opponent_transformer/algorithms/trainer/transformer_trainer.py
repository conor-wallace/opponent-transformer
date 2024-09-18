import numpy as np
import torch
import torch.nn as nn

from opponent_transformer.utils.util import cross_entropy_loss, get_grad_norm, huber_loss, mse_loss
from opponent_transformer.utils.valuenorm import ValueNorm
from opponent_transformer.utils.util import check, compute_input


class TransformerTrainer():
    """
    Trainer class for Opponent Transformer to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 opponent_policy,
                 opponent_model=None,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.opponent_policy = opponent_policy
        self.opponent_model = opponent_model

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            # print("Return batch update normalize shape: ", return_batch.shape)
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def cal_om_loss_and_accuracy(self, embeddings, oppnt_obs_targets, oppnt_action_targets, attention_masks):
        oppnt_obs_preds, oppnt_action_preds = self.opponent_model.predict_opponent(embeddings)

        oppnt_obs_preds = oppnt_obs_preds.reshape(-1, *oppnt_obs_preds.shape[2:])[attention_masks.reshape(-1) > 0]
        oppnt_obs_targets = oppnt_obs_targets.reshape(-1, *oppnt_obs_targets.shape[2:])[attention_masks.reshape(-1) > 0]
        oppnt_obs_loss = mse_loss(oppnt_obs_preds - oppnt_obs_targets).mean()

        num_oppnts, oppnt_act_dim = oppnt_action_preds.shape[2:]
        oppnt_action_targets = oppnt_action_targets.reshape(oppnt_action_preds.shape)
        oppnt_action_preds = oppnt_action_preds.reshape(-1, num_oppnts, oppnt_act_dim)[attention_masks.reshape(-1) > 0]
        oppnt_action_targets = oppnt_action_targets.reshape(-1, num_oppnts, oppnt_act_dim)[attention_masks.reshape(-1) > 0]
        oppnt_action_loss = cross_entropy_loss(oppnt_action_preds, oppnt_action_targets)

        num_samples = oppnt_action_preds.shape[0] * oppnt_action_preds.shape[1]
        oppnt_action_acc = torch.sum(oppnt_action_preds.argmax(dim=-1) == oppnt_action_targets.argmax(dim=-1)).cpu() / num_samples

        return oppnt_obs_loss + oppnt_action_loss, oppnt_obs_loss.item(), oppnt_action_acc

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        obs_batch, rnn_states_batch, rnn_states_critic_batch, prev_actions_batch, actions_batch,\
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, obs_seq_batch, action_seq_batch, reward_seq_batch, timesteps_batch, \
        attention_masks_batch, oppnt_obs_seq_batch, oppnt_action_seq_batch = sample

        # print("obs batch shape: ", obs_batch.shape)
        # print("actions batch shape: ", actions_batch.shape)
        # print("old action logprobs batch shape: ", old_action_log_probs_batch.shape)
        # print("values batch shape: ", value_preds_batch.shape)
        # print("returns batch shape: ", return_batch.shape)
        # print("advantages batch shape: ", adv_targ.shape)
        # print("rnn states batch shape: ", rnn_states_batch.shape)
        # print("rnn states critic batch shape: ", rnn_states_critic_batch.shape)

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        oppnt_obs_seq_batch = check(oppnt_obs_seq_batch).to(**self.tpdv)
        oppnt_action_seq_batch = check(oppnt_action_seq_batch).to(**self.tpdv)

        batch_dim = obs_batch.shape[0]

        embeddings = self.opponent_model(
            obs_seq_batch,
            action_seq_batch,
            reward_seq_batch,
            timesteps_batch,
            attention_masks_batch
        )

        om_loss, om_mse, om_acc = self.cal_om_loss_and_accuracy(embeddings, oppnt_obs_seq_batch, oppnt_action_seq_batch, attention_masks_batch)

        # might not want to update OM every epoch
        self.opponent_model.optimizer.zero_grad()

        om_loss.backward()

        if self._use_max_grad_norm:
            om_grad_norm = nn.utils.clip_grad_norm_(self.opponent_model.parameters(), self.max_grad_norm)
        else:
            om_grad_norm = get_grad_norm(self.opponent_model.parameters())

        self.opponent_model.optimizer.step()

        # Take the most recent embedding vector
        x_batch = embeddings[:, -1].detach()

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            x_batch.detach(), 
            rnn_states_batch, 
            rnn_states_critic_batch, 
            actions_batch, 
            masks_batch, 
            available_actions_batch,
            active_masks_batch
        )
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, om_loss, om_mse, om_acc, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, oppnt_buffers, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        oppnt_obs = np.concatenate([oppnt_buffer.obs for oppnt_buffer in oppnt_buffers], -1)
        oppnt_act = np.concatenate([oppnt_buffer.actions for oppnt_buffer in oppnt_buffers], -1)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['opponent_loss'] = 0
        train_info['opponent_obs_mse'] = 0
        train_info['opponent_action_acc'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, oppnt_obs, oppnt_act, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, om_loss, om_mse, om_acc, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['opponent_loss'] += om_loss.item()
                train_info['opponent_obs_mse'] += om_mse
                train_info['opponent_action_acc'] += om_acc
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
