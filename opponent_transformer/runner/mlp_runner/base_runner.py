import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from opponent_transformer.utils.separated_buffer import SeparatedReplayBuffer, SeparatedOpponentReplayBuffer
from opponent_transformer.utils.util import compute_input, get_shape_from_act_space, get_shape_from_obs_space
from opponent_transformer.algorithms.trainer import MLPTrainer as Trainer
from opponent_transformer.algorithms.opponent_transformer.algorithm.opponent_policy import OpponentPolicy
from opponent_transformer.algorithms.opponent_transformer.algorithm.transformer_policy import OpponentTransformerPolicy as Policy

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.agent_args = config['agent_args']
        self.oppnt_args = config['oppnt_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.num_agents = len(self.envs.action_space)
        self.num_opponents = self.num_agents - 1
        self.env_name = self.agent_args.env_name
        self.algorithm_name = self.agent_args.algorithm_name
        self.experiment_name = self.agent_args.experiment_name
        self.use_centralized_V = self.agent_args.use_centralized_V
        self.use_obs_instead_of_state = self.agent_args.use_obs_instead_of_state
        self.num_env_steps = self.agent_args.num_env_steps
        self.episode_length = self.agent_args.episode_length
        self.n_rollout_threads = self.agent_args.n_rollout_threads
        self.n_eval_rollout_threads = self.agent_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.agent_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.agent_args.use_linear_lr_decay
        self.hidden_size = self.agent_args.hidden_size
        self.use_wandb = self.agent_args.use_wandb
        self.use_render = self.agent_args.use_render
        self.recurrent_N = self.agent_args.recurrent_N

        # interval
        self.save_interval = self.agent_args.save_interval
        self.use_eval = self.agent_args.use_eval
        self.eval_interval = self.agent_args.eval_interval
        self.log_interval = self.agent_args.log_interval

        # dir
        self.model_dir = self.agent_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        if self.algorithm_name == 'ppo':
            embedding_size = None

            # No opponent model used
            self.oppnt_model = None
        elif self.algorithm_name == 'nam':
            obs_shape = get_shape_from_obs_space(self.envs.observation_space[0])[0]
            act_shape = self.envs.action_space[0].n
            embedding_size = obs_shape + act_shape

            # No opponent model used
            self.oppnt_model = None
        elif self.algorithm_name == 'oracle':
            obs_shape = get_shape_from_obs_space(self.envs.observation_space[0])[0]
            act_shape = self.envs.action_space[0].n
            embedding_size = obs_shape * self.num_agents + act_shape * self.num_agents

            # No opponent model used
            self.oppnt_model = None

        self.policy = Policy(
            self.agent_args,
            self.envs.observation_space[-1],
            self.envs.action_space[-1],
            device=self.device,
            embedding_size=embedding_size
        )

        # opponent policy network
        self.oppnt_policy = []
        for oppnt_id in range(self.num_agents - 1):
            # oppnt policy network
            po = OpponentPolicy(
                self.oppnt_args,
                self.envs.observation_space[oppnt_id],
                self.envs.action_space[oppnt_id],
                device=self.device
            )
            self.oppnt_policy.append(po)

        # algorithm
        self.trainer = Trainer(
            self.agent_args,
            self.policy,
            self.oppnt_policy,
            self.oppnt_model,
            device=self.device
        )

        # buffer
        self.buffer = SeparatedReplayBuffer(
            self.agent_args,
            self.num_agents,
            self.envs.observation_space[-1],
            self.envs.action_space[-1]
        )

        self.oppnt_buffer = []
        for oppnt_id in range(self.num_agents - 1):
            # buffer
            bu = SeparatedOpponentReplayBuffer(
                self.oppnt_args,
                self.envs.observation_space[oppnt_id],
                self.envs.action_space[oppnt_id]
            )
            self.oppnt_buffer.append(bu)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self, tasks):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()

        temp_actions_env = []
        for oppnt_id in range(self.num_agents - 1):
            action, rnn_state = self.oppnt_policy[oppnt_id].get_actions(
                self.oppnt_buffer[oppnt_id].obs[-1],
                self.oppnt_buffer[oppnt_id].rnn_states[-1],
                self.oppnt_buffer[oppnt_id].masks[-1],
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

            temp_actions_env.append(action_env)

        if self.algorithm_name == 'ppo':
            x = compute_input(
                agent_obs=self.buffer.obs[-1],
                action_dim=self.envs.action_space[-1].n,
                batch_dim=self.n_rollout_threads
            )
        elif self.algorithm_name == 'nam':
            x = compute_input(
                agent_obs=self.buffer.obs[-1],
                agent_actions=self.buffer.actions[-2],
                action_dim=self.envs.action_space[-1].n,
                batch_dim=self.n_rollout_threads
            )
        elif self.algorithm_name == 'oracle':
            oppnt_obs = np.concatenate([self.oppnt_buffer[i].obs[-1] for i in range(self.num_agents - 1)], 1)
            oppnt_act = np.concatenate(temp_actions_env, 1)
            x = compute_input(
                agent_obs=self.buffer.obs[-1],
                agent_actions=self.buffer.actions[-2],
                oppnt_obs=oppnt_obs,
                oppnt_actions=oppnt_act,
                action_dim=self.envs.action_space[-1].n,
                batch_dim=self.n_rollout_threads
            )

        next_value = self.trainer.policy.get_values(
            x,
            self.buffer.rnn_states_critic[-1],
            self.buffer.masks[-1]
        )
        next_value = _t2n(next_value)
        self.buffer.compute_returns(next_value, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, self.oppnt_buffer)
        for oppnt_buffer in self.oppnt_buffer:
            oppnt_buffer.after_update()
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.agent_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm.pt')
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)

    def restore_opponents(self):
        """Restore opponent policie's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.agent_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm.pt')
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

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
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
