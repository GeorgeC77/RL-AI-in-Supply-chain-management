import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def evaluate_policy(env, agent, turns):
    total_scores = 0
    for j in range(turns):
        done = False
        action = []
        s = env.reset()
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)

            # act = Action_adapter(a, env.mu, env.rho, env.sigma)
            act = Action_adapter(a, env.mu)
            s_next, r, dw = env.step(act)
            s = s_next
            done = dw

            total_scores += r
            action.append(act)
    return (total_scores / turns / (env.seq_length - 10))


def build_net(layer_shape, hidden_activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = hidden_activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
        super(Actor, self).__init__()
        layers = [state_dim] + list(hid_shape)

        self.a_net = build_net(layers, hidden_activation, output_activation)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic, with_logprob):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        # we learn log_std rather than std, so that exp(log_std) is always > 0
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = dist.rsample()

        '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        a = torch.tanh(u)
        if with_logprob:
            # Get probability density of logp_pi_a from probability density of u:
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a


class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2


# reward engineering for better training
def Reward_adapter_retailer(r):
    if r <= -1000: r = -1000
    return r


def Reward_adapter_manufacturer(r):
    if r <= -1000: r = -1000
    return r


# def Reward_adapter(r, EnvIdex):
#     # For Pendulum-v0
#     if EnvIdex == 0:
#         r = (r + 8) / 8
#
#     # For LunarLander
#     elif EnvIdex == 1:
#         if r <= -100: r = -10
#
#     # For BipedalWalker
#     elif EnvIdex == 4 or EnvIdex == 5:
#         if r <= -100: r = -1
#     return r


def Action_adapter(a, mu):
    # from [-1,1] to [0,max]
    max_action = 2 * mu
    return a * max_action / 2 + max_action / 2


def Action_adapter_reverse(act, mu):
    # from [0,max] to [-1,1]
    max_action = 2 * mu
    return (act - max_action / 2) / (max_action / 2)

# def Action_adapter(a, mu, rho, sigma):
#     sigma = 3 * sigma / (1 - rho ** 2) ** 0.5
#     # from [-1,1] to [-3sigma,3sigma]
#     return a * sigma + mu
#
#
# def Action_adapter_reverse(act, mu, rho, sigma):
#     # from [-3sigma,3sigma] to [-1,1]
#     sigma = 3 * sigma / (1 - rho ** 2) ** 0.5
#
#     return (act - mu) / sigma


def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
