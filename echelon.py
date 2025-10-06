import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy
from utils import Reward_adapter_retailer, Reward_adapter_manufacturer, Action_adapter, Action_adapter_reverse


def generate_demand(mu, rho, sigma, length):
    d = np.zeros([length + 5])
    d[0] = mu
    for i in range(1, len(d)):
        d[i] = mu + rho * (d[i - 1] - mu) + (np.random.normal(0, sigma))
    return d


class echelon(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.seq_length = self.seq_len + 10  # to prevent bug in step func
        self.l = self.Lead_time
        self.a = np.zeros(self.seq_length)
        self.act = np.zeros(self.seq_length)
        self.action_space = None
        self.t = 0
        self.h = 1
        self.b = 9
        self.u = 4
        self.w = 6
        self.dw = False
        self.r = np.zeros(self.seq_length)

        # retailer
        self.NS = np.zeros(self.seq_length)
        if self.l > 1:
            self.WIP = np.zeros([self.seq_length, self.l])
        self.h_cost = np.zeros(self.seq_length)
        self.b_cost = np.zeros(self.seq_length)
        self.demand = np.zeros(self.seq_length + 5)
        self.order = np.zeros(self.seq_length)

    def reset(self):
        self.t = 0
        # self.action_space = [self.mu - 3 * self.sigma / (1 - self.rho ** 2) ** 0.5, self.mu + 3 * self.sigma / (1 - self.rho ** 2) ** 0.5]
        self.action_space = [0, self.mu * 2]
        self.r = np.zeros(self.seq_length)
        self.a = np.zeros(self.seq_length)
        self.act = np.zeros(self.seq_length)
        self.dw = False
        # retailer
        self.NS = np.zeros(self.seq_length)
        if self.l > 1:
            self.WIP = np.zeros([self.seq_length, self.l - 1])
        self.h_cost = np.zeros(self.seq_length)
        self.b_cost = np.zeros(self.seq_length)
        self.demand = generate_demand(self.mu, self.rho, self.sigma, self.seq_length)

        # initialization
        self.NS[self.t] = self.mu - self.demand[self.t + 5]
        if self.l > 1:
            for i in range(self.l - 1):
                self.WIP[self.t, i] = self.mu + (np.random.normal(0, self.sigma))

        self.h_cost[self.t] = max(0, self.NS[self.t]) * self.h
        self.b_cost[self.t] = max(0, -self.NS[self.t]) * self.b

        # check linearty
        if self.is_linear == 0:
            self.NS[self.t] = max(self.NS[self.t], 0)
            if self.l > 1:
                for i in range(self.l - 1):
                    self.WIP[self.t, i] = max(self.WIP[self.t, i], 0)

        if self.l > 1:
            return np.concatenate(
                (self.WIP[self.t, :], [self.NS[self.t]], self.demand[self.t: self.t + 6]))  # WIP, NS, Demand
        else:
            return np.concatenate(([self.NS[self.t]], self.demand[self.t: self.t + 6]))  # NS, Demand

    def step(self, action):
        # select the corresponding action
        self.act[self.t] = action
        # self.a[self.t] = Action_adapter_reverse(action, self.mu, self.rho, self.sigma)
        self.a[self.t] = Action_adapter_reverse(action, self.mu)

        self.order[self.t] = self.act[self.t]

        # check linearty
        if self.is_linear == 0:
            self.order[self.t] = max(self.order[self.t], 0)
        self.t += 1

        # retailer
        if self.l == 1:
            self.NS[self.t] = self.NS[self.t - 1] + self.order[self.t - 1] - self.demand[self.t + 5]
        else:
            self.NS[self.t] = self.NS[self.t - 1] + self.WIP[self.t - 1][0] - self.demand[self.t + 5]
            self.WIP[self.t, :] = np.concatenate((self.WIP[self.t - 1, 1:], [self.order[self.t - 1]]))

        self.h_cost[self.t] = max(0, self.NS[self.t]) * self.h
        self.b_cost[self.t] = max(0, -self.NS[self.t]) * self.b

        self.r[self.t - 1] = (-(self.b_cost[self.t - 1] + self.h_cost[self.t - 1]))

        # determine the reward according to evaluation metrics

        if self.metric in [2, 4]:
            self.r[self.t - 1] += -abs(self.order[self.t - 1] - self.mu) * 5.454
        if self.t == self.seq_length - 10:  # termination
            self.dw = True


        if self.is_train:
            self.r[self.t - 1] = Reward_adapter_retailer(self.r[self.t - 1])


        # check linearty
        if self.is_linear == 0:
            self.NS[self.t] = max(self.NS[self.t], 0)
            if self.l == 1:
                for i in range(self.l - 1):
                    self.WIP[self.t, i] = max(self.WIP[self.t, i], 0)

        if self.l > 1:
            return np.concatenate((self.WIP[self.t, :], [self.NS[self.t]], self.demand[self.t: self.t + 6])), self.r[
                self.t - 1], self.dw  # S=[WIP, NS, Demand], r, dw
        else:
            return np.concatenate(([self.NS[self.t]], self.demand[self.t: self.t + 6])), self.r[
                self.t - 1], self.dw  # S=[WIP, NS, Demand], r, dw