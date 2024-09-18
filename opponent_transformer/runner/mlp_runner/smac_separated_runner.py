import time

import imageio
import numpy as np
import torch
import wandb
from functools import reduce
from opponent_transformer.runner.mlp_runner.base_runner import Runner
from opponent_transformer.utils.util import compute_input


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACRunner(Runner):
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            episode_task_indices = np.random.choice(self.oppnt_policy[0].num_policies, self.n_rollout_threads)
            episode_tasks = self.oppnt_policy[0].joint_policy_indices[episode_task_indices]

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, oppnt_onehot_actions = self.collect(step, episode_tasks)

                # Obser reward and next obs
                obs, _, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic, oppnt_onehot_actions

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute(episode_tasks)
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.agent_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
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
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won
                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape))
                train_infos.update({"average_episode_rewards": np.mean(self.buffer.rewards) * self.episode_length})   
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, _, available_actions = self.envs.reset()

        self.buffer.obs[0] = np.array(list(obs[:, -1])).copy()
        self.buffer.available_actions[0] = np.array(list(available_actions[:, -1])).copy()

        for oppnt_id in range(self.num_agents - 1):
            self.oppnt_buffer[oppnt_id].obs[0] = np.array(list(obs[:, oppnt_id])).copy()
            self.oppnt_buffer[oppnt_id].available_actions[0] = np.array(list(available_actions[:, oppnt_id])).copy()

    @torch.no_grad()
    def collect(self, step, tasks):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        oppnt_onehot_actions = []
        for oppnt_id in range(self.num_agents - 1):
            action, rnn_state = self.oppnt_policy[oppnt_id].get_actions(
                self.oppnt_buffer[oppnt_id].obs[step],
                self.oppnt_buffer[oppnt_id].rnn_states[step],
                self.oppnt_buffer[oppnt_id].masks[step],
                self.oppnt_buffer[oppnt_id].available_actions[step],
                tasks=tasks[:, oppnt_id]
            )

            # [agents, envs, dim]
            action = _t2n(action)

            # rearrange action
            if self.envs.action_space[oppnt_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[oppnt_id].shape):
                    uc_onehot_action = np.eye(self.envs.action_space[oppnt_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        onehot_action = uc_onehot_action
                    else:
                        onehot_action = np.concatenate((onehot_action, uc_onehot_action), axis=1)
            elif self.envs.action_space[oppnt_id].__class__.__name__ == 'Discrete':
                onehot_action = np.squeeze(np.eye(self.envs.action_space[oppnt_id].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            oppnt_onehot_actions.append(onehot_action)
            rnn_states.append(_t2n(rnn_state))

        if self.algorithm_name == 'ppo':
            x = compute_input(
                agent_obs=self.buffer.obs[step],
                action_dim=self.envs.action_space[-1].n,
                batch_dim=self.n_rollout_threads
            )
        elif self.algorithm_name == 'nam':
            x = compute_input(
                agent_obs=self.buffer.obs[step],
                agent_actions=self.buffer.actions[step - 1],
                action_dim=self.envs.action_space[-1].n,
                batch_dim=self.n_rollout_threads
            )
        elif self.algorithm_name == 'oracle':
            oppnt_obs = np.concatenate([self.oppnt_buffer[i].obs[step] for i in range(self.num_agents - 1)], 1)
            oppnt_act = np.concatenate(oppnt_onehot_actions, 1)
            x = compute_input(
                agent_obs=self.buffer.obs[step],
                agent_actions=self.buffer.actions[step - 1],
                oppnt_obs=oppnt_obs,
                oppnt_actions=oppnt_act,
                action_dim=self.envs.action_space[-1].n,
                batch_dim=self.n_rollout_threads
            )

        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(
            x,
            self.buffer.rnn_states[step],
            self.buffer.rnn_states_critic[step],
            self.buffer.masks[step],
            self.buffer.available_actions[step]
        )

        values.append(_t2n(value))

        # [agents, envs, dim]
        action = _t2n(action)

        actions.append(action)
        action_log_probs.append(_t2n(action_log_prob))
        rnn_states.append(_t2n(rnn_state))
        rnn_states_critic.append( _t2n(rnn_state_critic))

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        oppnt_onehot_actions = np.array(oppnt_onehot_actions).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, oppnt_onehot_actions

    def insert(self, data):
        obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic, oppnt_onehot_actions = data
        agent_dones = dones[:, -1].reshape(-1, 1)
        dones_env = np.all(dones, axis=1)

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[agent_dones == True] = np.zeros(((agent_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        # Insert opponent data
        for oppnt_id in range(self.num_agents - 1):

            self.oppnt_buffer[oppnt_id].insert(
                np.array(list(obs[:, oppnt_id])),
                rnn_states[:, oppnt_id],
                oppnt_onehot_actions[:, oppnt_id],
                rewards[:, oppnt_id],
                masks[:, oppnt_id],
                available_actions=available_actions[:, oppnt_id]
            )

        # Insert agent data
        self.buffer.insert(
            np.array(list(obs[:, -1])),
            rnn_states[:, -1],
            rnn_states_critic[:, -1],
            actions[:, -1],
            action_log_probs[:, -1],
            values[:, -1],
            rewards[:, -1],
            masks[:, -1],
            bad_masks=bad_masks[:, -1],
            active_masks=active_masks[:, -1],
            available_actions=available_actions[:, -1]
        )

    def eval_insert(self, data):
        obs, rewards, dones, infos, actions = data

        self.buffer.insert(
            np.array(list(obs[:, -1])),
            None,
            None,
            actions[:, -1],
            None,
            None,
            rewards[:, -1],
            None
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_episode_task_indices = np.random.choice(self.oppnt_policy[0].num_policies, self.n_eval_rollout_threads)
        eval_episode_tasks = self.oppnt_policy[0].joint_policy_indices[eval_episode_task_indices]

        eval_obs, _, eval_available_actions = self.eval_envs.reset()
        agent_eval_actions = np.zeros((self.n_eval_rollout_threads, self.buffer.act_shape))
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        step = 0
        while True:
            eval_actions = []
            eval_oppnt_onehot_actions = []

            self.trainer.prep_rollout()

            # get opponent actions
            for oppnt_id in range(self.num_agents - 1):
                eval_action, eval_rnn_state = self.oppnt_policy[oppnt_id].act(
                    np.array(list(eval_obs[:, oppnt_id])),
                    eval_rnn_states[:, oppnt_id],
                    eval_masks[:, oppnt_id],
                    eval_available_actions[:, oppnt_id],
                    deterministic=True,
                    tasks=eval_episode_tasks[:, oppnt_id]
                )

                eval_action = eval_action.detach().cpu().numpy()

                # rearrange action
                if self.eval_envs.action_space[oppnt_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[oppnt_id].shape):
                        eval_uc_onehot_action = np.eye(self.eval_envs.action_space[oppnt_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_onehot_action = eval_uc_onehot_action
                        else:
                            eval_onehot_action = np.concatenate((eval_onehot_action, eval_uc_onehot_action), axis=1)
                elif self.eval_envs.action_space[oppnt_id].__class__.__name__ == 'Discrete':
                    eval_onehot_action = np.squeeze(np.eye(self.eval_envs.action_space[oppnt_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_actions.append(eval_action)
                eval_oppnt_onehot_actions.append(eval_onehot_action)
                eval_rnn_states[:, oppnt_id] = _t2n(eval_rnn_state)

            if self.algorithm_name == 'ppo':
                x = compute_input(
                    agent_obs=eval_obs[:, -1],
                    action_dim=self.envs.action_space[0].n,
                    batch_dim=self.n_eval_rollout_threads
                )
            elif self.algorithm_name == 'nam':
                x = compute_input(
                    agent_obs=eval_obs[:, -1],
                    agent_actions=agent_eval_actions,
                    action_dim=self.envs.action_space[0].n,
                    batch_dim=self.n_eval_rollout_threads
                )
            elif self.algorithm_name == 'oracle':
                oppnt_obs = np.concatenate([np.array(list(eval_obs[:, i])) for i in range(self.num_agents - 1)], 1)
                oppnt_act = np.concatenate(eval_oppnt_onehot_actions, 1)
                x = compute_input(
                    agent_obs=np.array(list(eval_obs[:, -1])),
                    agent_actions=agent_eval_actions,
                    oppnt_obs=oppnt_obs,
                    oppnt_actions=oppnt_act,
                    action_dim=self.envs.action_space[-1].n,
                    batch_dim=self.n_eval_rollout_threads
                )

            eval_action, eval_rnn_state = self.trainer.policy.act(
                x,
                eval_rnn_states[:, -1],
                eval_masks[:, -1],
                eval_available_actions[:, -1],
                deterministic=True,
            )

            eval_action = eval_action.detach().cpu().numpy()
            eval_actions.append(eval_action)
            eval_rnn_states[:, -1] = _t2n(eval_rnn_state)

            eval_actions = np.array(eval_actions).transpose(1, 0, 2)

            step += 1

            # Observe reward and next obs
            eval_obs, _, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.agent_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    eval_episode_task_indices = np.random.choice(self.oppnt_policy[0].num_policies, 1)
                    eval_episode_tasks[eval_i] = self.oppnt_policy[0].joint_policy_indices[eval_episode_task_indices]
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.agent_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
