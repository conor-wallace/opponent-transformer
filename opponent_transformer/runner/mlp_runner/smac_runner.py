import time
import wandb
import numpy as np
from functools import reduce
import torch
from opponent_transformer.runner.mlp_runner.base_shared_runner import Runner
from opponent_transformer.utils.util import compute_input


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        self.tasks = np.array([
            [9, 9, 9, 9, 16, 16, 16, 16], # rmappo high 0, rmappo medium 1
            [10, 10, 10, 10, 14, 14, 14, 14], # rmappo high 1, rmappo low 2
            [17, 17, 17, 17, 12, 12, 12, 12], # rmappo medium 2, rmappo low 0
            [0, 0, 0, 0, 7, 7, 7, 7], # rippo high 0, rippo medium 1 
            [1, 1, 1, 1, 5, 5, 5, 5], # rippo high 1, rippo low 2
            [8, 8, 8, 8, 3, 3, 3, 3], # rippo medium 2, rippo low 0
            [9, 9, 9, 9, 1, 1, 1, 1], # rmappo high 0, rippo high 1
            [16, 16, 16, 16, 8, 8, 8, 8], # rmappo medium 1, rippo medium 2
            [14, 14, 14, 14, 3, 3, 3, 3], # rmappo low 2, rippo low 0
            [0, 0, 0, 0, 10, 10, 10, 10], # rippo high 0, rmappo high 1
            [7, 7, 7, 7, 17, 17, 17, 17], # rippo medium 1, rmappo medium 2
            [5, 5, 5, 5, 12, 12, 12, 12], # rippo low 2, rmappo low 0
        ])

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            episode_tasks = np.random.choice(self.trainer.opponent_policy.num_policies, self.n_rollout_threads)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step, episode_tasks)

                # Obser reward and next obs
                obs, _, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 

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
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
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
                
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, _, available_actions = self.envs.reset()

        # replay buffer
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step, tasks):
        self.trainer.prep_rollout()

        # get opponent actions
        oppnt_action, oppnt_action_log_prob, oppnt_rnn_state = self.trainer.opponent_policy.get_actions(
            self.buffer.obs[step, :, :-1],
            self.buffer.rnn_states[step, :, :-1],
            self.buffer.masks[step, :, :-1],
            self.buffer.available_actions[step, :, :-1],
            tasks
        )

        oppnt_actions = np.array(np.split(_t2n(oppnt_action), self.n_rollout_threads))
        oppnt_action_log_probs = np.array(np.split(_t2n(oppnt_action_log_prob), self.n_rollout_threads))
        oppnt_rnn_states = np.array(np.split(_t2n(oppnt_rnn_state), self.n_rollout_threads))
        oppnt_rnn_states_critic = np.zeros_like(oppnt_rnn_states)

        # get agent actions

        if self.algorithm_name == 'ppo':
            x = compute_input(
                agent_obs=self.buffer.obs[step, :, -1],
                action_dim=self.envs.action_space[0].n,
                batch_dim=self.n_rollout_threads
            )
        elif self.algorithm_name == 'nam':
            x = compute_input(
                agent_obs=self.buffer.obs[step, :, -1],
                agent_actions=self.buffer.actions[step - 1, :, -1],
                action_dim=self.envs.action_space[0].n,
                batch_dim=self.n_rollout_threads
            )
        elif self.algorithm_name == 'oracle':
            x = compute_input(
                agent_obs=self.buffer.obs[step, :, -1],
                agent_actions=self.buffer.actions[step - 1, :, -1],
                oppnt_obs=self.buffer.obs[step, :, :-1],
                oppnt_actions=oppnt_actions,
                action_dim=self.envs.action_space[0].n,
                batch_dim=self.n_rollout_threads
            )

        agent_value, agent_action, agent_action_log_prob, agent_rnn_state, agent_rnn_state_critic \
            = self.trainer.policy.get_actions(
            x,
            np.concatenate(np.expand_dims(self.buffer.rnn_states[step, :, -1], 1)),
            np.concatenate(np.expand_dims(self.buffer.rnn_states_critic[step, :, -1], 1)),
            np.concatenate(np.expand_dims(self.buffer.masks[step, :, -1], 1)),
            np.concatenate(np.expand_dims(self.buffer.available_actions[step, :, -1], 1))
        )

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(agent_value), self.n_rollout_threads))
        agent_actions = np.array(np.split(_t2n(agent_action), self.n_rollout_threads))
        agent_action_log_probs = np.array(np.split(_t2n(agent_action_log_prob), self.n_rollout_threads))
        agent_rnn_states = np.array(np.split(_t2n(agent_rnn_state), self.n_rollout_threads))
        agent_rnn_states_critic = np.array(np.split(_t2n(agent_rnn_state_critic), self.n_rollout_threads))

        # combine agent and oppnt arrays
        actions = np.concatenate((oppnt_actions, agent_actions), axis=1)
        action_log_probs = np.concatenate((oppnt_action_log_probs, agent_action_log_probs), axis=1)
        rnn_states = np.concatenate((oppnt_rnn_states, agent_rnn_states), axis=1)
        rnn_states_critic = agent_rnn_states_critic

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        agent_rewards = np.expand_dims(rewards[:, -1], 1)
        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), 1, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        self.buffer.insert(
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            agent_rewards,
            masks,
            bad_masks,
            active_masks,
            available_actions
        )

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_tasks = np.random.choice(self.trainer.opponent_policy.num_policies, self.n_eval_rollout_threads)

        eval_obs, _, eval_available_actions = self.eval_envs.reset()
        agent_eval_actions = np.zeros((self.n_eval_rollout_threads, self.buffer.act_shape))
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        step = 0
        while True:
            self.trainer.prep_rollout()

            # get opponent actions
            oppnt_eval_actions, oppnt_eval_rnn_state = self.trainer.opponent_policy.act(
                eval_obs[:, :-1],
                eval_rnn_states[:, :-1],
                eval_masks[:, :-1],
                eval_available_actions[:, :-1],
                eval_episode_tasks,
                deterministic=True
            )

            oppnt_eval_actions = np.array(np.split(_t2n(oppnt_eval_actions), self.n_eval_rollout_threads))
            oppnt_eval_rnn_states = np.array(np.split(_t2n(oppnt_eval_rnn_state), self.n_eval_rollout_threads))

            # print("Eval obs shape: ", )

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
                x = compute_input(
                    agent_obs=eval_obs[:, -1],
                    agent_actions=agent_eval_actions,
                    oppnt_obs=eval_obs[:, :-1],
                    oppnt_actions=oppnt_eval_actions,
                    action_dim=self.envs.action_space[0].n,
                    batch_dim=self.n_eval_rollout_threads
                )

            agent_eval_actions, agent_eval_rnn_state = self.trainer.policy.act(
                x,
                np.concatenate(np.expand_dims(eval_rnn_states[:, -1], 1)),
                np.concatenate(np.expand_dims(eval_masks[:, -1], 1)),
                np.concatenate(np.expand_dims(eval_available_actions[:, -1], 1)),
                deterministic=True
            )

            agent_eval_actions = np.array(np.split(_t2n(agent_eval_actions), self.n_eval_rollout_threads))
            agent_eval_rnn_states = np.array(np.split(_t2n(agent_eval_rnn_state), self.n_eval_rollout_threads))

            eval_actions = np.concatenate((oppnt_eval_actions, agent_eval_actions), axis=1)
            eval_rnn_states = np.concatenate((oppnt_eval_rnn_states, agent_eval_rnn_states), axis=1)

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
                    eval_episode_tasks[eval_i] = np.random.choice(self.trainer.opponent_policy.num_policies, 1)
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
