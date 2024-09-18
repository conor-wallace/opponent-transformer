import copy
import numpy as np
import math
import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
        

def pad_sequence(x, seq_length):
    padding = np.zeros((seq_length, *x.shape[1:]))
    x_padded = np.concatenate((padding, x))
    return x_padded[-seq_length:]


def stack_inputs(embeddings):
    batch_size = embeddings[0].shape[0]
    seq_length = embeddings[0].shape[1]
    hidden_dim = embeddings[0].shape[2]
    num_modalities = len(embeddings)

    # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
    # which works nice in an autoregressive sense since states predict actions
    stacked_inputs = torch.stack(embeddings, dim=1).permute(0, 2, 1, 3).reshape(batch_size, num_modalities * seq_length, hidden_dim)

    return stacked_inputs


def stack_attention_mask(attention_mask, num_modalities, batch_size, seq_length):
    # to make the attention mask fit the stacked inputs, have to stack it as well
    stacked_attention_mask = torch.stack(
        (attention_mask,) * num_modalities, dim=1
    ).permute(0, 2, 1).reshape(batch_size, num_modalities * seq_length)

    return stacked_attention_mask


def compute_input(
    agent_obs,
    agent_actions=None,
    oppnt_obs=None,
    oppnt_actions=None,
    eval=False,
    batch_dim=None,
    action_dim=None,
):
    if agent_actions is None and oppnt_obs is None and oppnt_actions is None:
        x = agent_obs
    elif oppnt_obs is None and oppnt_actions is None:
        agent_actions_onehot = torch.functional.F.one_hot(torch.from_numpy(agent_actions).long(), action_dim).numpy()
        x = np.concatenate((agent_obs, agent_actions_onehot), -1)
    else:
        agent_actions_onehot = torch.functional.F.one_hot(torch.from_numpy(agent_actions).long(), action_dim).numpy()
        agent_actions_onehot = agent_actions_onehot.reshape(batch_dim, -1)

        x = np.concatenate((agent_obs, agent_actions_onehot, oppnt_obs, oppnt_actions), -1)

    return x


def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mse_loss(e):
    return e**2/2


def cross_entropy_loss(p, t):
    return torch.nn.functional.cross_entropy(p, t)


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c