import os
import sys
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../../")
from opponent_transformer.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from opponent_transformer.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from opponent_transformer.smac.training import SMACTrainer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "OM_Starcraft2"
    """the wandb's project name"""
    wandb_entity: str = "bhd445"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    env_id: str = "MMM2"
    """the id of the environment"""
    model_type: str = "NAM"
    """the agent policy type, can be one of [NAM, LIAM, OT, Oracle]"""
    hidden_dim: int = 128
    """the hidden dimension of the agent policy"""
    num_rnn_layers: int = 1
    """the number of GRU layers for the agent policy"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_opponent_policies: int = 1
    """the number of opponent policies to sample from"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_eval_envs: int = 1
    """the number of parallel game environments for evaluation"""
    num_eval_episodes: int = 10
    """the number of episodes to run for evaluation"""
    num_steps: int = 5
    """the number of steps to run in each environment per policy rollout"""
    episode_length: int = 100
    """the maximum length of an episode"""
    log_interval: int = 1
    """the number of episodes between logging"""
    eval_interval: int = 25
    """the number of episodes between evaluation"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_mini_batch: int = 1
    """the number of mini-batches"""
    num_epochs: int = 1
    """the K epochs to update the policy"""
    data_chunk_length: int = 10
    """chunk length for training recurrent policies"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    huber_coef: float = 10.0
    """coefficient for huber loss"""
    value_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, rank):
    def init_env():
        env = StarCraft2Env(
            map_name=env_id
        )
        env.seed(1000 + rank * 10000)
        return env

    return init_env


def make_eval_env(env_id, rank):
    def init_env():
        env = StarCraft2Env(
            map_name=env_id
        )
        env.seed(50000 + rank * 10000)
        return env

    return init_env


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.episode_length // args.num_mini_batch)
    args.num_iterations = args.total_timesteps // (args.episode_length * args.num_envs)
    print(f"Num iterations: {args.num_iterations}")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_num_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_fns = [make_env(args.env_id, i) for i in range(args.num_envs)]
    envs = ShareSubprocVecEnv(env_fns)

    eval_env_fns = [make_eval_env(args.env_id, i) for i in range(args.num_eval_envs)]
    eval_envs = ShareSubprocVecEnv(eval_env_fns)

    trainer = SMACTrainer(
        args=args,
        envs=envs,
        eval_envs=eval_envs,
    )

    trainer.run()
