import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from .buffer import PPOBuffer
from .utils import huber_loss
from ..models import RPPO
from ..pretrained_opponents.pretrained_mmm2_opponents import get_opponent_actions


class SMACTrainer:
    def __init__(
        self,
        args,
        envs,
        eval_envs,
    ):
        self.envs = envs
        self.eval_envs = eval_envs

        self.num_envs = args.num_envs
        self.num_eval_envs = args.num_eval_envs
        self.num_eval_episodes = args.num_eval_episodes
        self.episode_length = args.episode_length
        self.total_timesteps = args.total_timesteps
        self.num_epochs = args.num_epochs
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.max_grad_norm = args.max_grad_norm
        self.log_interval = args.log_interval
        self.eval_interval = args.eval_interval
        self.use_wandb = args.track

        self.clip_param = args.clip_coef
        self.huber_param = args.huber_coef
        self.value_param = args.value_coef

        self.num_agents = len(self.envs.action_space)
        self.num_opponents = self.num_agents - 1
        print("Num opponents: ", self.num_opponents)
        self.num_opponent_policies = args.num_opponent_policies
        self.act_dim = self.envs.action_space[-1].n
        self.obs_dim = self.envs.observation_space[-1][0]
        self.opp_act_dim = self.envs.action_space[0].n
        self.opp_obs_dim = self.envs.observation_space[0][0]
        self.hidden_dim = args.hidden_dim
        self.num_rnn_layers = args.num_rnn_layers

        self.model_type = args.model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        if self.model_type == "NAM":
            embedding_dim = self.obs_dim + self.act_dim
        elif self.model_type == "Oracle":
            embedding_dim = (
                self.obs_dim + self.act_dim +\
                self.opp_act_dim * self.num_opponents +\
                self.opp_obs_dim * self.num_opponents
            )
        else:
            embedding_dim = None

        self.agent = RPPO(
            num_agents=self.num_agents,
            num_opponents=self.num_opponents,
            act_dim=self.act_dim,
            obs_dim=self.obs_dim,
            opp_act_dim=self.opp_act_dim,
            opp_obs_dim=self.opp_obs_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=embedding_dim,
            num_rnn_layers=self.num_rnn_layers
        )

        self.buffer = PPOBuffer(
            args=args,
            num_agents=self.num_agents,
            num_opponents=self.num_opponents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            opp_obs_dim=self.opp_obs_dim,
            opp_act_dim=self.opp_act_dim,
            num_rnn_layers=self.num_rnn_layers,
            hidden_dim=args.hidden_dim,
            device=self.device
        )

        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)

        if args.env_id == '1c3s5z':
            from ..pretrained_opponents.pretrained_1c3s5z_opponents import get_opponent_actions
        elif args.env_id == 'MMM2':
            from ..pretrained_opponents.pretrained_mmm2_opponents import get_opponent_actions

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.total_timesteps) // self.episode_length // self.num_envs

        last_battles_game = np.zeros(self.num_envs, dtype=np.float32)
        last_battles_won = np.zeros(self.num_envs, dtype=np.float32)

        for episode in range(episodes):

            for step in range(self.episode_length):
                agent_actions, oppnt_actions, agent_rnn_states, oppnt_rnn_states, agent_action_log_probs, agent_values = self.collect(step)

                actions = torch.cat((oppnt_actions.argmax(-1).unsqueeze(-1), agent_actions.unsqueeze(1)), dim=1)

                obs, _, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, rewards, dones, infos, available_actions, \
                       agent_actions, oppnt_actions, agent_action_log_probs, agent_values, \
                       agent_rnn_states, oppnt_rnn_states

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            self.buffer.after_update()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.num_envs           
            # save model

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(episode,
                                episodes,
                                total_num_steps,
                                self.total_timesteps,
                                int(total_num_steps / (end - start))))

                battles_won = []
                battles_game = []
                incre_battles_won = []
                incre_battles_game = []                    

                for i, info in enumerate(infos):
                    if 'battles_won' in info[0].keys():
                        battles_won.append(info[0]['battles_won'])
                        incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                    if 'battles_game' in info[0].keys():
                        battles_game.append(info[0]['battles_game'])
                        incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])
                
                incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                print("incre win rate is {}.".format(incre_win_rate))
                if self.use_wandb:
                    wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)

                last_battles_game = battles_game
                last_battles_won = battles_won

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0:
                print("Evaluate")
                self.evaluate(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        agent_obs = obs[:, -1]
        oppnt_obs = obs[:, :-1]
        agent_available_actions = available_actions[:, -1]
        oppnt_available_actions = available_actions[:, :-1]

        # replay buffer
        self.buffer.obs[0] = agent_obs.copy()
        self.buffer.available_actions[0] = agent_available_actions.copy()
        self.buffer.oppnt_obs[0] = oppnt_obs.copy()
        self.buffer.oppnt_available_actions[0] = oppnt_available_actions.copy()

        # setup opponent policy tasks
        self.tasks = np.random.choice(range(self.num_opponent_policies), size=self.num_envs)

    def collect(self, step: int):
        oppnt_actions = []
        oppnt_rnn_states = []
        for id in range(self.num_envs):
            oppnt_actions_id, oppnt_rnn_states_id = get_opponent_actions(
                self.buffer.oppnt_obs[step, id], self.buffer.oppnt_rnn_states[step, id], self.buffer.oppnt_masks[step, id], self.buffer.oppnt_available_actions[step, id], self.tasks[id]
            )
            oppnt_actions.append(oppnt_actions_id.unsqueeze(0))
            oppnt_rnn_states.append(oppnt_rnn_states_id.unsqueeze(0))
        oppnt_actions = torch.cat(oppnt_actions)
        oppnt_rnn_states = torch.cat(oppnt_rnn_states)

        x = self.compute_embedding(step)
        agent_actions, agent_action_log_probs, agent_values, agent_rnn_states = self.agent.act(
            x,
            self.buffer.rnn_states[step],
            self.buffer.masks[step],
            self.buffer.available_actions[step]
        )
        agent_actions = agent_actions.cpu()
        agent_action_log_probs = agent_action_log_probs.cpu()
        agent_values = agent_values.cpu()
        agent_rnn_states = agent_rnn_states.cpu()

        return (
            agent_actions,
            oppnt_actions,
            agent_rnn_states,
            oppnt_rnn_states,
            agent_action_log_probs,
            agent_values
        )

    def compute_embedding(self, step: int):
        if self.model_type == 'NAM':
            x = np.concatenate((self.buffer.obs[step], self.buffer.actions[step]), axis=-1)
        else:
            x = self.buffer.obs[step]
        
        return x

    def insert(self, data):
        obs, rewards, dones, infos, available_actions, \
        agent_actions, oppnt_actions, agent_action_log_probs, agent_values, \
        agent_rnn_states, oppnt_rnn_states = data

        agent_obs = torch.from_numpy(obs[:, -1])
        oppnt_obs = torch.from_numpy(obs[:, :-1])
        agent_available_actions = torch.from_numpy(available_actions[:, -1])
        oppnt_available_actions = torch.from_numpy(available_actions[:, :-1])
        agent_rewards = torch.from_numpy(rewards[:, -1].reshape(-1, 1))
        agent_dones = dones[:, -1].reshape(-1, 1)
        oppnt_dones = dones[:, :-1]

        dones_env = np.all(dones, axis=1)
        agent_dones_env = np.all(agent_dones, axis=1)
        oppnt_dones_env = np.all(oppnt_dones, axis=1)

        # Add agent dim
        agent_rnn_states[dones_env == True] = torch.zeros(((dones_env == True).sum(), self.num_rnn_layers, self.hidden_dim), dtype=torch.float32)
        oppnt_rnn_states[dones_env == True] = torch.zeros(((dones_env == True).sum(), self.num_opponents, self.num_rnn_layers, 64), dtype=torch.float32)

        agent_masks = np.ones((self.num_envs, 1), dtype=np.float32)
        agent_masks[dones_env == True] = np.zeros(((dones_env == True).sum(), 1), dtype=np.float32)
        agent_masks = torch.from_numpy(agent_masks)

        oppnt_masks = np.ones((self.num_envs, self.num_opponents, 1), dtype=np.float32)
        oppnt_masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_opponents, 1), dtype=np.float32)
        oppnt_masks = torch.from_numpy(oppnt_masks)

        # Figure out how to fix this
        agent_active_masks = np.ones((self.num_envs, 1), dtype=np.float32)
        agent_active_masks[agent_dones == True] = np.zeros(((agent_dones == True).sum()), dtype=np.float32)
        agent_active_masks[agent_dones_env == True] = np.ones(((agent_dones_env == True).sum(), 1), dtype=np.float32)
        agent_active_masks = torch.from_numpy(agent_active_masks)

        oppnt_active_masks = np.ones((self.num_envs, self.num_opponents, 1), dtype=np.float32)
        oppnt_active_masks[oppnt_dones == True] = np.zeros(((oppnt_dones == True).sum(), 1), dtype=np.float32)
        oppnt_active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_opponents, 1), dtype=np.float32)
        oppnt_active_masks = torch.from_numpy(oppnt_active_masks)

        agent_bad_masks = torch.tensor([[[0.0] if info[-1]['bad_transition'] else [1.0]] for info in infos]).squeeze(1)
        oppnt_bad_masks = torch.tensor([[[0.0] if info[oppnt_id]['bad_transition'] else [1.0] for oppnt_id in range(self.num_opponents)] for info in infos])

        self.buffer.insert(
            obs=agent_obs,
            actions=agent_actions,
            logprobs=agent_action_log_probs,
            oppnt_obs=oppnt_obs,
            oppnt_actions=oppnt_actions,
            rewards=agent_rewards,
            values=agent_values,
            available_actions=agent_available_actions,
            oppnt_available_actions=oppnt_available_actions,
            rnn_states=agent_rnn_states,
            masks=agent_masks,
            active_masks=agent_active_masks,
            bad_masks=agent_bad_masks,
            oppnt_rnn_states=oppnt_rnn_states,
            oppnt_masks=oppnt_masks,
            oppnt_active_masks=oppnt_active_masks,
            oppnt_bad_masks=oppnt_bad_masks
        )

    def compute(self):
        self.agent.eval()
        self.agent.eval()

        x = self.compute_embedding(-1)
        next_values = self.agent.get_values(
            x,
            self.buffer.rnn_states[-1],
            self.buffer.masks[-1]
        )

        self.buffer.compute_returns(next_values)

    def train(self, update_actor=True):
        self.agent.train()

        advantages = self.buffer.returns[:-1] - self.buffer.value_normalizer.denormalize(self.buffer.values[:-1])

        advantages_copy = advantages.copy()
        advantages_copy[self.buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['loss'] = 0

        for _ in range(self.num_epochs):
            data_generator = self.buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)

            for sample in data_generator:

                loss = self.ppo_update(sample, update_actor)

                train_info['loss'] += loss.item()

        num_updates = self.num_epochs * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def ppo_update(self, sample, update_actor):
        obs_batch, rnn_states_batch, actions_batch, last_actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_logprobs_batch, \
        adv_targ, available_actions_batch = sample

        if self.model_type == 'NAM':
            x = np.concatenate((obs_batch, last_actions_batch), axis=-1)
        else:
            x = obs_batch

        values, logprobs, dist_entropy = self.agent.evaluate(
            x, 
            rnn_states_batch,
            masks_batch,
            actions_batch,
            available_actions_batch,
            active_masks_batch
        )

        old_logprobs_batch = torch.from_numpy(old_logprobs_batch).to(**self.tpdv)
        adv_targ = torch.from_numpy(adv_targ).to(**self.tpdv)
        value_preds_batch = torch.from_numpy(value_preds_batch).to(**self.tpdv)
        return_batch = torch.from_numpy(return_batch).to(**self.tpdv)
        active_masks_batch = torch.from_numpy(active_masks_batch).to(**self.tpdv)

        # actor update
        imp_weights = torch.exp(logprobs - old_logprobs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        policy_action_loss = (
            -torch.sum(
                torch.min(surr1, surr2),
                dim=-1,
                keepdim=True
            ) * active_masks_batch
        ).sum() / active_masks_batch.sum()

        policy_loss = policy_action_loss

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                self.clip_param)

        self.buffer.value_normalizer.update(return_batch)
        error_clipped = self.buffer.value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = self.buffer.value_normalizer.normalize(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.huber_param)
        value_loss_original = huber_loss(error_original, self.huber_param)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()

        loss = policy_loss + value_loss * self.value_param

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def evaluate(self, total_num_steps):
        self.agent.eval()

        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, _, eval_available_actions = self.eval_envs.reset()

        oppnt_rnn_states = np.zeros((self.num_eval_envs, self.num_opponents, 1, 64), dtype=np.float32)
        oppnt_masks = np.ones((self.num_eval_envs, self.num_opponents, 1))

        agent_rnn_states = np.zeros((self.num_eval_envs, 1, 128), dtype=np.float32)
        agent_masks = np.ones((self.num_eval_envs, 1))

        last_agent_actions = np.zeros((self.num_eval_envs, self.act_dim))

        eval_tasks = np.random.choice(range(self.num_opponent_policies), size=self.num_eval_envs)

        while True:
            agent_obs = eval_obs[:, -1]
            oppnt_obs = eval_obs[:, :-1]
            agent_available_actions = eval_available_actions[:, -1]
            oppnt_available_actions = eval_available_actions[:, :-1]

            oppnt_actions = []
            for id in range(self.num_eval_envs):
                oppnt_actions_id, oppnt_rnn_states[id] = get_opponent_actions(
                    oppnt_obs[id], oppnt_rnn_states[id], oppnt_masks[id], oppnt_available_actions[id], eval_tasks[id]
                )
                oppnt_actions.append(oppnt_actions_id.unsqueeze(0))
                # oppnt_rnn_states.append(oppnt_rnn_states_id.unsqueeze(0))
            oppnt_actions = torch.cat(oppnt_actions)
            # oppnt_rnn_states = torch.cat(oppnt_rnn_states)

            x = np.concatenate((agent_obs, last_agent_actions), axis=-1)
            agent_actions, _, _, agent_rnn_states = self.agent.act(
                x,
                agent_rnn_states,
                agent_masks,
                agent_available_actions
            )
            agent_actions = agent_actions.cpu()
            last_actions_actions = agent_actions.numpy()
            agent_rnn_states = agent_rnn_states.cpu().numpy()

            eval_actions = torch.cat((oppnt_actions.argmax(-1).unsqueeze(-1), agent_actions.unsqueeze(1)), dim=1)

            eval_obs, _, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            agent_dones = eval_dones[:, -1].reshape(-1, 1)
            oppnt_dones = eval_dones[:, :-1]

            eval_dones_env = np.all(eval_dones, axis=1)
            agent_dones_env = np.all(agent_dones, axis=1)
            oppnt_dones_env = np.all(oppnt_dones, axis=1)

            agent_rnn_states[eval_dones_env == True] = torch.zeros(((eval_dones_env == True).sum(), self.num_rnn_layers, self.hidden_dim), dtype=torch.float32)
            oppnt_rnn_states[eval_dones_env == True] = torch.zeros(((eval_dones_env == True).sum(), self.num_opponents, self.num_rnn_layers, 64), dtype=torch.float32)

            agent_masks = np.ones((self.num_eval_envs, 1), dtype=np.float32)
            agent_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), 1), dtype=np.float32)

            oppnt_masks = np.ones((self.num_eval_envs, self.num_opponents, 1), dtype=np.float32)
            oppnt_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_opponents, 1), dtype=np.float32)

            for eval_i in range(self.num_eval_envs):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.num_eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                break

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)