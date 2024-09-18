#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
import torch
from pathlib import Path

from opponent_transformer.config import get_config
from opponent_transformer.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from opponent_transformer.envs.starcraft2.smac_maps import get_map_params
from opponent_transformer.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from opponent_transformer.runner.mlp_runner.smac_separated_runner import SMACRunner as Runner


"""Train script for SMAC."""
def make_train_env(agent_args):
    def get_env_fn(rank):
        def init_env():
            if agent_args.env_name == "StarCraft2":
                env = StarCraft2Env(agent_args)
            else:
                print("Can not support the " + agent_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(agent_args.seed + rank * 1000)
            return env

        return init_env

    if agent_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(agent_args.n_rollout_threads)])


def make_eval_env(agent_args):
    def get_env_fn(rank):
        def init_env():
            if agent_args.env_name == "StarCraft2":
                env = StarCraft2Env(agent_args)
            else:
                print("Can not support the " + agent_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(agent_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if agent_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(agent_args.n_eval_rollout_threads)])


def parse_agent_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)

    agent_args = parser.parse_known_args(args)[0]

    return agent_args


def parse_oppnt_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")

    oppnt_args = parser.parse_known_args(args)[0]

    oppnt_args.hidden_size = 64
    oppnt_args.use_recurrent_policy = False 
    oppnt_args.use_naive_recurrent_policy = False
    oppnt_args.use_centralized_V = False

    return oppnt_args


def main(args):
    agent_parser = get_config()
    agent_args = parse_agent_args(args, agent_parser)

    oppnt_parser = get_config()
    oppnt_args = parse_oppnt_args(args, oppnt_parser)

    if agent_args.algorithm_name == "ppo":
        agent_args.use_recurrent_policy = True
        agent_args.use_naive_recurrent_policy = False
    elif agent_args.algorithm_name == "nam":
        agent_args.use_recurrent_policy = True
        agent_args.use_naive_recurrent_policy = False
    elif agent_args.algorithm_name == "oracle":
        agent_args.use_recurrent_policy = True
        agent_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    # cuda
    if agent_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(agent_args.n_training_threads)
        if agent_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(agent_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / agent_args.env_name / agent_args.map_name / agent_args.algorithm_name / agent_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if agent_args.use_wandb:
        run = wandb.init(config=agent_args,
                         project=agent_args.env_name,
                         entity=agent_args.user_name,
                         notes=socket.gethostname(),
                         name=str(agent_args.algorithm_name) + "_" +
                              str(agent_args.experiment_name) +
                              "_seed" + str(agent_args.seed),
                         group=agent_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(agent_args.algorithm_name) + "-" + str(agent_args.env_name) + "-" + str(agent_args.experiment_name) + "@" + str(
            agent_args.user_name))

    # seed
    torch.manual_seed(agent_args.seed)
    torch.cuda.manual_seed_all(agent_args.seed)
    np.random.seed(agent_args.seed)

    # env
    envs = make_train_env(agent_args)
    eval_envs = make_eval_env(agent_args) if agent_args.use_eval else None
    num_agents = get_map_params(agent_args.map_name)["n_agents"]

    config = {
        "agent_args": agent_args,
        "oppnt_args": oppnt_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if agent_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if agent_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
