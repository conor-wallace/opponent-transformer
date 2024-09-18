import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from opponent_transformer.envs import ShareDummyVecEnv, StarCraft2Env
from opponent_transformer.smac.models.policies import NAM
from opponent_transformer.smac.training import Buffer
from opponent_transformer.smac.pretrained_opponents.pretrained_1c3s5z_opponents import get_opponent_actions


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "bhd445"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    env_id: str = "1c3s5z"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_opponent_policies: int = 1
    """the number of opponent policies to sample from"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_eval_envs: int = 10
    """the number of parallel game environments for evaluation"""
    num_eval_episodes: int = 10
    """the number of episodes to run for evaluation"""
    num_steps: int = 5
    """the number of steps to run in each environment per policy rollout"""
    episode_length: int = 400
    """the maximum length of an episode"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 10
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 1.0
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


@torch.no_grad()
def evaluate(agent, eval_envs, args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    eval_obs, _, eval_available_actions = eval_envs.reset()

    oppnt_hidden_states = torch.zeros((args.num_eval_envs, agent.num_opponents, 1, 64), dtype=torch.float32)
    oppnt_masks = torch.ones((args.num_eval_envs, agent.num_opponents, 1))

    agent_hidden_states = torch.zeros(agent.lstm.num_layers, args.num_eval_envs, agent.lstm.hidden_size).to(device)
    agent_cell_states = torch.zeros(agent.lstm.num_layers, args.num_eval_envs, agent.lstm.hidden_size).to(device)

    last_agent_actions = torch.zeros(args.num_eval_envs, agent.act_dim).to(device, dtype=torch.float32).to(device)

    tasks = np.random.choice(range(args.num_opponent_policies), size=args.num_eval_envs)

    episode = 0
    battles_won = 0

    episode_returns = []
    one_episode_returns = np.zeros((args.num_eval_envs, 1), dtype=np.float32)

    while True:
        agent_obs = torch.from_numpy(eval_obs[:, -1])
        oppnt_obs = torch.from_numpy(eval_obs[:, :-1])
        agent_available_actions = torch.from_numpy(eval_available_actions[:, -1])
        oppnt_available_actions = torch.from_numpy(eval_available_actions[:, :-1])

        oppnt_actions = []
        for id in range(args.num_eval_envs):
            oppnt_actions_id, oppnt_hidden_states[id] = get_opponent_actions(
                oppnt_obs[id], oppnt_hidden_states[id], oppnt_masks[id], oppnt_available_actions[id], tasks[id]
            )
            oppnt_actions.append(oppnt_actions_id.unsqueeze(0))
        oppnt_actions = torch.cat(oppnt_actions)

        x = torch.cat((agent_obs, last_agent_actions), dim=-1)
        agent_actions, agent_values, agent_lstm_states = agent.act(
            x,
            (agent_hidden_states, agent_cell_states),
            agent_available_actions
        )
        agent_values = agent_values.squeeze(-1)
        agent_hidden_states, agent_cell_states = agent_lstm_states

        actions = torch.cat((oppnt_actions, agent_actions.unsqueeze(1)), dim=1)
        actions = actions.argmax(-1)

        eval_obs, _, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(actions)
        eval_dones_env = np.all(eval_dones, axis=1)

        agent_hidden_states[:, eval_dones_env == True] = torch.zeros((agent.lstm.num_layers, (eval_dones_env == True).sum(), agent.lstm.hidden_size), dtype=torch.float32).to(device)
        agent_cell_states[:, eval_dones_env == True] = torch.zeros((agent.lstm.num_layers, (eval_dones_env == True).sum(), agent.lstm.hidden_size), dtype=torch.float32).to(device)

        oppnt_hidden_states[eval_dones_env == True] = torch.zeros(((eval_dones_env == True).sum(), agent.num_opponents, 1, 64), dtype=torch.float32).to(device)
        oppnt_masks = torch.ones((args.num_eval_envs, agent.num_opponents, 1), dtype=torch.float32)
        oppnt_masks[eval_dones_env == True] = torch.zeros(((eval_dones_env == True).sum(), agent.num_opponents, 1), dtype=torch.float32)

        for i in range(args.num_eval_envs):
            if eval_dones_env[i]:
                episode += 1
                episode_returns.append(one_episode_returns[i])
                one_episode_returns[i] = 0.0
                if eval_infos[i][0]['won']:
                    battles_won += 1
            else:
                one_episode_returns[i] += eval_rewards[i, -1]

        if episode >= args.num_eval_episodes:
            episode_returns = np.array(episode_returns)
            avg_return = episode_returns.mean()
            win_rate = battles_won / episode
            break

    return avg_return, win_rate


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.episode_length // args.num_minibatches)
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
    envs = ShareDummyVecEnv(env_fns)

    eval_env_fns = [make_eval_env(args.env_id, i) for i in range(args.num_eval_envs)]
    eval_envs = ShareDummyVecEnv(eval_env_fns)

    agent = NAM(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    buffer = Buffer(
        num_minibatches=args.num_minibatches,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_envs=args.num_envs,
        num_opponents=agent.num_opponents,
        agent_obs_dim=agent.obs_dim,
        agent_act_dim=agent.act_dim,
        oppnt_obs_dim=agent.opp_obs_dim,
        oppnt_act_dim=agent.opp_act_dim,
        num_lstm_layers=agent.lstm.num_layers,
        hidden_size=agent.lstm.hidden_size,
        device=device
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    counter = 0
    episodes_passed = -1
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        for idx in range(args.num_minibatches):
            if counter == 0:
                # reset environment and initialize data
                obs, _, available_actions = envs.reset()

                dones = torch.zeros(args.num_envs).to(device)
                last_agent_actions = torch.zeros(args.num_envs, agent.act_dim).to(device, dtype=torch.float32)
                agent_hidden_states = torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device)
                agent_cell_states = torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device)

                # shuffle the set of opponent policies
                tasks = np.random.choice(range(args.num_opponent_policies), size=args.num_envs)
                oppnt_hidden_states = torch.zeros((args.num_envs, agent.num_opponents, 1, 64), dtype=torch.float32)
                oppnt_masks = torch.ones((args.num_envs, agent.num_opponents, 1))

            # reset lstm state to the current state
            buffer.hidden_states[idx] = agent_hidden_states.clone()
            buffer.cell_states[idx] = agent_cell_states.clone()

            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.batch_size):
                # print(f"Iteration: {iteration}, Minibatch: {idx}, Step: {counter}")
                agent_obs = torch.from_numpy(obs[:, -1])
                oppnt_obs = torch.from_numpy(obs[:, :-1])
                agent_available_actions = torch.from_numpy(available_actions[:, -1])
                oppnt_available_actions = torch.from_numpy(available_actions[:, :-1])

                oppnt_actions = []
                for id in range(args.num_envs):
                    oppnt_actions_id, oppnt_hidden_states[id] = get_opponent_actions(
                        oppnt_obs[id], oppnt_hidden_states[id], oppnt_masks[id], oppnt_available_actions[id], tasks[id]
                    )
                    oppnt_actions.append(oppnt_actions_id.unsqueeze(0))
                oppnt_actions = torch.cat(oppnt_actions)

                x = torch.cat((agent_obs, last_agent_actions), dim=-1)
                agent_actions, agent_values, agent_lstm_states = agent.act(
                    x,
                    (agent_hidden_states, agent_cell_states),
                    agent_available_actions
                )
                agent_values = agent_values.squeeze(-1)
                agent_hidden_states, agent_cell_states = agent_lstm_states

                actions = torch.cat((oppnt_actions, agent_actions.unsqueeze(1)), dim=1)
                actions = actions.argmax(-1)

                obs, _, rewards, dones, infos, available_actions = envs.step(actions)
                agent_rewards = torch.from_numpy(rewards[:, -1]).squeeze(-1)
                dones_env = np.all(dones, axis=1)

                buffer.insert(
                    idx,
                    step,
                    agent_obs,
                    last_agent_actions,
                    agent_actions,
                    oppnt_obs,
                    oppnt_actions,
                    agent_rewards,
                    agent_values,
                )

                global_step += args.num_envs
                counter += 1
                last_agent_actions = agent_actions

                if counter == args.episode_length:
                    dones = torch.ones(args.num_envs).to(device)
                    episodes_passed += 1
                    counter = 0
                else:
                    dones = torch.zeros(args.num_envs).to(device)

            # bootstrap value if not done
            with torch.no_grad():
                if dones[0]:
                    next_value = torch.zeros(args.num_envs).to(device)
                else:
                    agent_obs = torch.from_numpy(obs[:, -1])

                    x = torch.cat(
                        (agent_obs, last_agent_actions)
                        , dim=-1
                    ).to(device)
                    next_value = agent.get_value(
                        x,
                        agent_lstm_states
                    ).reshape(1, -1)

                buffer.compute_returns(idx, next_value)

        # Optimizing the policy and value network
        for epoch in range(args.update_epochs):
            for idx in range(args.num_minibatches):
                batch = buffer.generate_batch(idx)

                mb_obs = batch.get("agent_obs")
                mb_last_actions = batch.get("last_agent_actions")
                mb_actions = batch.get("agent_actions")
                mb_opp_obs = batch.get("oppnt_obs")
                mb_opp_actions = batch.get("oppnt_actions")
                mb_returns = batch.get("agent_returns")
                mb_hidden = batch.get("agent_hidden_states")
                mb_cell = batch.get("agent_cell_states")

                x = torch.cat(
                    (mb_obs, mb_last_actions),
                    dim=-1
                )
                new_log_probs, entropy, new_values = agent.evaluate(
                    x,
                    (mb_hidden, mb_cell),
                    mb_actions
                )

                new_values = new_values.squeeze(-1)
                mb_advantages = mb_returns - new_values.detach()

                action_loss = -(mb_advantages.detach() * new_log_probs).mean()

                value_loss = (mb_returns - new_values).pow(2).mean()

                entropy_loss = entropy.mean()

                loss = action_loss - args.ent_coef * entropy_loss + value_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if (episodes_passed % 10 == 0) and (counter == 0):
            average_reward, win_rate = evaluate(agent, eval_envs, args)
            print(f"Episode={episodes_passed}, episodic_return={average_reward}, win_rate={win_rate}")
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], episodes_passed)
            writer.add_scalar("losses/value_loss", value_loss.item(), episodes_passed)
            writer.add_scalar("losses/policy_loss", action_loss.item(), episodes_passed)
            writer.add_scalar("losses/entropy", entropy_loss.item(), episodes_passed)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), episodes_passed)
            writer.add_scalar("charts/episodic_return", average_reward, episodes_passed)
            writer.add_scalar("charts/win_rate", win_rate, episodes_passed)

    envs.close()
    writer.close()
