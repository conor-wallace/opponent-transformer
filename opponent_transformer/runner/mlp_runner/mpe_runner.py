import time

import imageio
import numpy as np
import torch
import wandb
from opponent_transformer.runner.mlp_runner.base_runner import Runner
from opponent_transformer.utils.util import compute_input


def _t2n(x):
    return x.detach().cpu().numpy()


class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            episode_task_indices = np.random.choice(self.oppnt_policy[0].num_policies, self.n_rollout_threads)
            episode_tasks = self.oppnt_policy[0].joint_policy_indices[episode_task_indices]

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step, episode_tasks)

                # print("Len env actions = ", len(actions_env))
                # print("First env actions shape = ", len(actions_env[0]))
                # print("Env actions:")
                # pprint(actions_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

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
                        .format(self.agent_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    idv_rews = []
                    for info in infos:
                        for count, info in enumerate(infos):
                            if 'individual_reward' in infos[count][-1].keys():
                                idv_rews.append(infos[count][-1].get('individual_reward', 0))
                    train_infos.update({'individual_rewards': np.mean(idv_rews)})
                    train_infos.update({"average_episode_rewards": np.mean(self.buffer.rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        self.buffer.obs[0] = np.array(list(obs[:, -1])).copy()

        for oppnt_id in range(self.num_agents - 1):
            self.oppnt_buffer[oppnt_id].obs[0] = np.array(list(obs[:, oppnt_id])).copy()

    @torch.no_grad()
    def collect(self, step, tasks):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for oppnt_id in range(self.num_agents - 1):
            action, rnn_state = self.oppnt_policy[oppnt_id].get_actions(
                self.oppnt_buffer[oppnt_id].obs[step],
                self.oppnt_buffer[oppnt_id].rnn_states[step],
                self.oppnt_buffer[oppnt_id].masks[step],
                tasks=tasks[:, oppnt_id]
            )

            # [agents, envs, dim]
            action = _t2n(action)

            # rearrange action
            if self.envs.action_space[oppnt_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[oppnt_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[oppnt_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[oppnt_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[oppnt_id].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
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
            oppnt_act = np.concatenate(temp_actions_env, 1)
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

        # rearrange action
        if self.envs.action_space[oppnt_id].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[oppnt_id].shape):
                uc_action_env = np.eye(self.envs.action_space[oppnt_id].high[i]+1)[action[:, i]]
                if i == 0:
                    action_env = uc_action_env
                else:
                    action_env = np.concatenate((action_env, uc_action_env), axis=1)
        elif self.envs.action_space[oppnt_id].__class__.__name__ == 'Discrete':
            action_env = np.squeeze(np.eye(self.envs.action_space[oppnt_id].n)[action], 1)
        else:
            raise NotImplementedError

        actions.append(action)
        temp_actions_env.append(action_env)
        action_log_probs.append(_t2n(action_log_prob))
        rnn_states.append(_t2n(rnn_state))
        rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = data
        agent_dones = dones[:, -1].reshape(-1, 1)

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[agent_dones == True] = np.zeros(((agent_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # Insert opponent data
        for oppnt_id in range(self.num_agents - 1):
            oppnt_action = np.concatenate([action_env[oppnt_id].reshape(1, -1) for action_env in actions_env])

            self.oppnt_buffer[oppnt_id].insert(
                np.array(list(obs[:, oppnt_id])),
                rnn_states[:, oppnt_id],
                oppnt_action,
                rewards[:, oppnt_id],
                masks[:, oppnt_id]
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
            masks[:, -1]
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
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_episode_task_indices = np.random.choice(self.oppnt_policy[0].num_policies, self.n_eval_rollout_threads)
        eval_episode_tasks = self.oppnt_policy[0].joint_policy_indices[eval_episode_task_indices]
        agent_eval_actions = np.zeros((self.n_eval_rollout_threads, self.buffer.act_shape))
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):

            eval_actions = []
            eval_temp_actions_env = []
            rnn_states = []

            self.trainer.prep_rollout()

            for oppnt_id in range(self.num_agents - 1):
                eval_action, eval_rnn_state = self.oppnt_policy[oppnt_id].act(
                    np.array(list(eval_obs[:, oppnt_id])),
                    eval_rnn_states[:, oppnt_id],
                    eval_masks[:, oppnt_id],
                    deterministic=True,
                    tasks=eval_episode_tasks[:, oppnt_id]
                )

                eval_action = eval_action.detach().cpu().numpy()

                # rearrange action
                if self.eval_envs.action_space[oppnt_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[oppnt_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[oppnt_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[oppnt_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[oppnt_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_actions.append(eval_action)
                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, oppnt_id] = _t2n(eval_rnn_state)

            if self.algorithm_name == 'ppo':
                x = compute_input(
                    agent_obs=np.array(list(eval_obs[:, -1])),
                    action_dim=self.envs.action_space[-1].n,
                    batch_dim=self.n_eval_rollout_threads
                )
            elif self.algorithm_name == 'nam':
                x = compute_input(
                    agent_obs=np.array(list(eval_obs[:, -1])),
                    agent_actions=agent_eval_actions,
                    action_dim=self.envs.action_space[-1].n,
                    batch_dim=self.n_eval_rollout_threads
                )
            elif self.algorithm_name == 'oracle':
                oppnt_obs = np.concatenate([np.array(list(eval_obs[:, i])) for i in range(self.num_agents - 1)], 1)
                oppnt_act = np.concatenate(eval_temp_actions_env, 1)
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
                eval_rnn_states[:, oppnt_id],
                eval_masks[:, oppnt_id],
                deterministic=True,
            )

            eval_action = eval_action.detach().cpu().numpy()
            agent_eval_actions = eval_action
            # rearrange action
            if self.eval_envs.action_space[-1].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[-1].shape):
                    eval_uc_action_env = np.eye(self.eval_envs.action_space[-1].high[i]+1)[eval_action[:, i]]
                    if i == 0:
                        eval_action_env = eval_uc_action_env
                    else:
                        eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
            elif self.eval_envs.action_space[-1].__class__.__name__ == 'Discrete':
                eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[-1].n)[eval_action], 1)
            else:
                raise NotImplementedError

            eval_actions.append(eval_action)
            eval_temp_actions_env.append(eval_action_env)
            eval_rnn_states[:, -1] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_actions = np.array(eval_actions).transpose(1, 0, 2)
            data = eval_obs, eval_rewards, eval_dones, eval_infos, eval_actions
            self.eval_insert(data)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = {}
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.update({'eval_average_episode_rewards': eval_average_episode_rewards})
            # print("Average episode rewards: ", eval_average_episode_rewards)

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.agent_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.agent_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.agent_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.agent_args.ifi:
                        time.sleep(self.agent_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                # print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.agent_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.agent_args.ifi)
