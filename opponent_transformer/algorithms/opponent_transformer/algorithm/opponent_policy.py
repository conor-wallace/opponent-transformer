import glob
import os
import numpy as np
import torch
from opponent_transformer.algorithms.opponent_transformer.algorithm.actor_critic import Actor


SMAC_JOINT_POLICIES = {
    "3s_vs_5z": [
        [0, 0],
        [0, 2],
        [2, 2],
        # [0, 1],
        # [2, 1],
        [3, 3],
        [3, 5],
        [5, 5],
        # [3, 4],
        # [5, 4],
        [0, 3],
        [2, 5]
    ]
}

MPE_JOINT_POLICIES = {
    "simple_spread": [
        [0, 0],
        [0, 1],
        [0, 4],
        [1, 5],
        [0, 2],
        [1, 3],
        [4, 2],
        [5, 3],
        [6, 6],
        [6, 7],
        [6, 10],
        [7, 11],
        [6, 8],
        [7, 9],
        [10, 8],
        [11, 9],
    ]
}


class OpponentPolicy:
    """
    Opponent Transformer Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.algorithm_name = args.algorithm_name

        self.obs_space = obs_space
        self.act_space = act_space

        self.opponent_dir = args.opponent_dir
        policy_weights = sorted(glob.glob(os.path.join(self.opponent_dir, "*")))

        self.joint_policy_indices = np.array(MPE_JOINT_POLICIES[args.scenario_name])
        self.num_policies = len(self.joint_policy_indices)

        self.actors = []
        for policy_weight in policy_weights:
            policy_state_dict = torch.load(policy_weight)
            actor = Actor(args, self.obs_space, self.act_space, self.device)
            actor.load_state_dict(policy_state_dict)
            self.actors.append(actor)

    def get_actions(
            self,
            obs,
            rnn_states_actor,
            masks,
            available_actions=None,
            tasks=None,
            deterministic=False
        ):
        """
        Compute actions and value function predictions for the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        """
        actions, new_rnn_states_actor = [], []
        for i, task in enumerate(tasks):
            if available_actions is not None:
                available_actions_i = available_actions[i]
            else:
                available_actions_i = None

            task_actions, _, task_rnn_states_actor = self.actors[task](
                obs[i],
                rnn_states_actor[i],
                masks[i],
                available_actions=available_actions_i,
                deterministic=deterministic
            )
            actions.append(task_actions)
            new_rnn_states_actor.append(task_rnn_states_actor)

        actions = torch.cat(actions).unsqueeze(1)
        new_rnn_states_actor = torch.cat(new_rnn_states_actor).unsqueeze(1)

        return actions, new_rnn_states_actor

    def act(self, obs, rnn_states_actor, masks, available_actions=None, tasks=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, new_rnn_states_actor = [], []
        for i, task in enumerate(tasks):
            if available_actions is not None:
                available_actions_i = available_actions[i]
            else:
                available_actions_i = None

            task_actions, _, task_rnn_states_actor = self.actors[task](
                obs[i],
                rnn_states_actor[i],
                masks[i],
                available_actions=available_actions_i,
                deterministic=deterministic
            )
            actions.append(task_actions)
            new_rnn_states_actor.append(task_rnn_states_actor)

        actions = torch.cat(actions).unsqueeze(1)
        new_rnn_states_actor = torch.cat(new_rnn_states_actor).unsqueeze(1)

        return actions, new_rnn_states_actor
