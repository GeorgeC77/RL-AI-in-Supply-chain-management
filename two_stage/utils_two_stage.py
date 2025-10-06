import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import pandas as pd
import copy


def evaluate_policy(env, agent, env_2, agent_2, turns):
    total_scores = 0
    total_scores_2 = 0
    std_r = []
    std_m = []
    action = []
    demand_r = []
    ns_r = []
    order_r = []
    demand_m = []
    ns_m = []
    order_m = []
    bullwhip_r = []
    bullwhip_m = []
    nsamp_r = []
    nsamp_m = []
    for z in range(turns):
        done = False
        s = env.reset()
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            # act = Action_adapter(a, env.mu, env.rho, env.sigma)
            act = Action_adapter_r(a, env.mu)
            s_next, r, dw = env.step(act)
            s = s_next
            action.append(act)
            done = dw
        total_scores += np.sum(env.r[:env.seq_length - 10])

        order_r.append(copy.deepcopy(env.order[:env.seq_length - 10]))
        ns_r.append(copy.deepcopy(env.NS[:env.seq_length - 10]))
        demand_r.append(copy.deepcopy(env.demand[:env.seq_length - 10]))

        bullwhip_r.append(np.var(copy.deepcopy(env.order[:env.seq_length - 10])) / np.var(copy.deepcopy(env.demand[:env.seq_length - 10])))
        nsamp_r.append(np.var(copy.deepcopy(env.NS[:env.seq_length - 10])) / np.var(copy.deepcopy(env.demand[:env.seq_length - 10])))

        s_2 = env_2.reset(env)
        done_2 = False
        while not done_2:
            a_2 = agent_2.select_action(s_2, deterministic=True)
            # act_2 = Action_adapter(a_2, env_2.mu, env_2.rho, env_2.sigma)
            act_2 = Action_adapter_m(a_2, env_2.mu)
            s_next_2, r_2, dw_2 = env_2.step(act_2)
            s_2 = s_next_2
            done_2 = dw_2
        total_scores_2 += np.sum(env_2.r[6:env.seq_length - 10])

        std_r.append(copy.deepcopy(np.std(env.order[:-10])))
        std_m.append(copy.deepcopy(np.std(env_2.order[:-10])))
        order_m.append(copy.deepcopy(env_2.order[:env_2.seq_length - 10]))
        ns_m.append(copy.deepcopy(env_2.NS[:env_2.seq_length - 10]))
        demand_m.append(copy.deepcopy(env_2.demand[:env_2.seq_length - 10]))

        bullwhip_m.append(np.var(copy.deepcopy(env_2.order[:env_2.seq_length - 10])) / np.var(copy.deepcopy(env_2.demand[:env_2.seq_length - 10])))
        nsamp_m.append(np.var(copy.deepcopy(env_2.NS[:env_2.seq_length - 10])) / np.var(copy.deepcopy(env_2.demand[:env_2.seq_length - 10])))

    print(np.mean(std_r), np.mean(std_m))

    if env.is_train == False:
        pd.DataFrame(order_r).to_csv('data/order_r_' + str(env.rho) + '.csv', index=False)
        pd.DataFrame(ns_r).to_csv('data/NS_r_' + str(env.rho) + '.csv', index=False)
        pd.DataFrame(demand_r).to_csv('data/demand_r_' + str(env.rho) + '.csv', index=False)

        pd.DataFrame(order_m).to_csv('data/order_m_' + str(env_2.rho) + '.csv', index=False)
        pd.DataFrame(ns_m).to_csv('data/NS_m_' + str(env_2.rho) + '.csv', index=False)
        pd.DataFrame(demand_m).to_csv('data/demand_m_' + str(env_2.rho) + '.csv', index=False)
    if env.record_dynamics:

        return (total_scores / turns / (env.seq_length - 10), total_scores_2 / turns / (env_2.seq_length - 10 - 6), np.mean(bullwhip_r), np.mean(bullwhip_m), np.mean(nsamp_r), np.mean(nsamp_m))
    else:
        return (total_scores / turns / (env.seq_length - 10), total_scores_2 / turns / (env_2.seq_length - 10 - 6))


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
def Reward_adapter(r):
    if r <= -500: r = -500
    return r


def Action_adapter_r(a, mu):
    # from [-1,1] to [0,max]
    max_action = 2 * mu
    return a * max_action / 2 + max_action / 2


def Action_adapter_reverse_r(act, mu):
    # from [0,max] to [-1,1]
    max_action = 2 * mu
    return (act - max_action / 2) / (max_action / 2)


def Action_adapter_m(a, mu):
    # from [-1,1] to [mu-max, mu+max]
    max_action = 1 * mu
    return a * max_action + mu


def Action_adapter_reverse_m(act, mu):
    # from [mu-max, mu+max] to [-1,1]
    max_action = 1 * mu
    return (act - mu) / max_action


def Reward_adapter_retailer(r):
    if r <= -1000: r = -1000
    return r


def Reward_adapter_manufacturer(r):
    if r <= -1000: r = -1000
    return r


# def Action_adapter(a, mu, rho, sigma):
#     sigma = 3 * sigma/(1-rho**2)
#     # from [-1,1] to [-3sigma,3sigma]
#     return a * sigma + mu
#
#
# def Action_adapter_reverse(act, mu, rho, sigma):
#     # from [-3sigma,3sigma] to [-1,1]
#     sigma = 3 * sigma / (1 - rho ** 2)
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
