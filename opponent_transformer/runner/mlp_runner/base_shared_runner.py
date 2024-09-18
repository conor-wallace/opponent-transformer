import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from opponent_transformer.utils.shared_buffer import SharedReplayBuffer
from opponent_transformer.utils.util import compute_input, get_shape_from_act_space, get_shape_from_obs_space
from opponent_transformer.algorithms.opponent_transformer.trainer import OpponentTransformerTrainer as Trainer
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
        elif self.algorithm_name == 'nam':
            obs_shape = get_shape_from_obs_space(self.envs.observation_space[0])[0]
            act_shape = self.envs.action_space[0].n
            embedding_size = obs_shape + act_shape
        elif self.algorithm_name == 'oracle':
            for idx, obs_space in enumerate(self.envs.observation_space):
                obs_shape = get_shape_from_obs_space(obs_space)[0]
                print(f"Agent {idx} obs shape: {obs_shape}")

            obs_shape = get_shape_from_obs_space(self.envs.observation_space[0])[0]
            act_shape = self.envs.action_space[0].n
            print("Obs shape: ", obs_shape)
            print("Act shape: ", act_shape)
            embedding_size = obs_shape * self.num_agents + act_shape * self.num_agents

        self.policy = Policy(
            self.agent_args,
            self.envs.observation_space[-1],
            self.envs.action_space[-1],
            device=self.device,
            embedding_size=embedding_size
        )

        # opponent policy network
        self.opponent_policy = OpponentPolicy(
            self.oppnt_args,
            self.envs.observation_space[0],
            self.envs.action_space[0],
            device=self.device
        )

        # algorithm
        self.trainer = Trainer(
            self.agent_args,
            self.policy,
            self.opponent_policy,
            device=self.device
        )

        # buffer
        self.buffer = SharedReplayBuffer(
            self.agent_args,
            self.num_agents,
            self.envs.observation_space[0],
            self.envs.action_space[0]
        )

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

        if self.algorithm_name == 'ppo':
            x = compute_input(
                agent_obs=self.buffer.obs[-1, :, -1],
                action_dim=self.envs.action_space[0].n,
                batch_dim=self.n_rollout_threads
            )
        elif self.algorithm_name == 'nam':
            x = compute_input(
                agent_obs=self.buffer.obs[-1, :, -1],
                agent_actions=self.buffer.actions[-2, :, -1],
                action_dim=self.envs.action_space[0].n,
                batch_dim=self.n_rollout_threads
            )
        elif self.algorithm_name == 'oracle':
            # get opponent actions
            oppnt_action, oppnt_action_log_prob, oppnt_rnn_state = self.trainer.opponent_policy.get_actions(
                self.buffer.obs[-1, :, :-1],
                self.buffer.rnn_states[-1, :, :-1],
                self.buffer.masks[-1, :, :-1],
                self.buffer.available_actions[-1, :, :-1],
                tasks
            )

            oppnt_actions = np.array(np.split(_t2n(oppnt_action), self.n_rollout_threads))

            x = compute_input(
                agent_obs=self.buffer.obs[-1, :, -1],
                agent_actions=self.buffer.actions[-2, :, -1],
                oppnt_obs=self.buffer.obs[-1, :, :-1],
                oppnt_actions=oppnt_actions,
                action_dim=self.envs.action_space[0].n,
                batch_dim=self.n_rollout_threads
            )

        next_values = self.trainer.policy.get_values(
            x,
            np.concatenate(np.expand_dims(self.buffer.rnn_states_critic[-1, :, -1], 1)),
            np.concatenate(np.expand_dims(self.buffer.masks[-1, :, -1], 1))
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
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
