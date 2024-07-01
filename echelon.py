import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy


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
        self.WIP = np.zeros([self.seq_length, self.l])
        self.h_cost = np.zeros(self.seq_length)
        self.b_cost = np.zeros(self.seq_length)
        self.demand = np.zeros(self.seq_length + 5)

    def reset(self):
        self.t = 0
        self.action_space = np.arange(0, self.action_dim, 1)
        self.r = np.zeros(self.seq_length)
        self.a = np.zeros(self.seq_length)
        self.dw = False
        # retailer
        self.NS = np.zeros(self.seq_length)
        self.WIP = np.zeros([self.seq_length, self.l - 1])
        self.h_cost = np.zeros(self.seq_length)
        self.b_cost = np.zeros(self.seq_length)
        self.demand = generate_demand(self.mu, self.rho, self.sigma, self.seq_length)

        self.NS[self.t] = self.mu - self.demand[self.t + 5]
        for i in range(self.l - 1):
            self.WIP[self.t, i] = self.mu + (np.random.normal(0, self.sigma))
        self.h_cost[self.t] = max(0, self.NS[self.t]) * self.h
        self.b_cost[self.t] = max(0, -self.NS[self.t]) * self.b

        return np.concatenate(
            (self.WIP[self.t, :], [self.NS[self.t]], self.demand[self.t: self.t + 6]))  # WIP, NS, Demand

    def step(self, action):  # action: order quantity
        self.a[self.t] = action
        self.t += 1

        # retailer
        self.NS[self.t] = self.NS[self.t - 1] + self.WIP[self.t - 1][0] - self.demand[self.t + 5]
        self.h_cost[self.t] = max(0, self.NS[self.t]) * self.h
        self.b_cost[self.t] = max(0, -self.NS[self.t]) * self.b
        self.WIP[self.t, :] = np.concatenate((self.WIP[self.t - 1, 1:], [self.a[self.t - 1]]))


        # determine the reward according to evaluation metrics
        self.r[self.t - 1] = -(self.b_cost[self.t - 1] + self.h_cost[self.t - 1])
        if self.t == self.seq_length - 10 + 1: # one more time step a info is needed
            self.dw = True
            if self.metric in [2, 4]:
                self.r[self.t - 2] += -(np.std(self.a[:self.t - 1]) * 0.3636 * self.w) * (self.t - 1)

        return np.concatenate((self.WIP[self.t, :], [self.NS[self.t]], self.demand[self.t: self.t + 6])), self.r[
            self.t - 1], self.dw  # S=[WIP, NS, Demand], r, dw

# manufacturer
class echelon_m(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.seq_length = self.seq_len + 10  # to prevent bug in step func
        self.l = self.Lead_time
        self.a = np.zeros(self.seq_length)
        self.action_space = None
        self.t = 0
        self.h = 1
        self.b = 9
        self.u = 4
        self.w = 6
        self.dw = False
        self.r = np.zeros(self.seq_length)

        # manufacturer
        self.NS = np.zeros(self.seq_length)
        self.WIP = np.zeros([self.seq_length, self.l])
        self.h_cost = np.zeros(self.seq_length)
        self.b_cost = np.zeros(self.seq_length)
        self.demand = np.zeros(self.seq_length + 5)

    def reset(self, retailer):
        self.t = 0
        self.action_space = np.arange(0, self.action_dim, 1)
        self.r = np.zeros(self.seq_length)
        self.a = np.zeros(self.seq_length)
        self.dw = False
        # manufacturer
        self.NS = np.zeros(self.seq_length)
        self.WIP = np.zeros([self.seq_length, self.l - 1])
        self.h_cost = np.zeros(self.seq_length)
        self.b_cost = np.zeros(self.seq_length)
        # the first m-N(WIP) elements equal to mu
        self.demand[:3] = self.mu
        self.demand[3:5] = retailer.WIP[0,:]
        self.demand[5:] = retailer.a

        self.NS[self.t] = self.mu - self.demand[self.t + 5]
        for i in range(self.l - 1):
            self.WIP[self.t, i] = self.mu + (np.random.normal(0, self.sigma))
        self.h_cost[self.t] = max(0, self.NS[self.t]) * self.h
        self.b_cost[self.t] = max(0, -self.NS[self.t]) * self.b

        return np.concatenate(
            (self.WIP[self.t, :], [self.NS[self.t]], self.demand[self.t: self.t + 6]))  # WIP, NS, Demand

    def step(self, action):  # action: order quantity
        self.a[self.t] = action
        self.t += 1

        # manufacturer
        self.NS[self.t] = self.NS[self.t - 1] + self.WIP[self.t - 1][0] - self.demand[self.t + 5]
        self.h_cost[self.t] = max(0, self.NS[self.t]) * self.h
        self.b_cost[self.t] = max(0, -self.NS[self.t]) * self.b
        self.WIP[self.t, :] = np.concatenate((self.WIP[self.t - 1, 1:], [self.a[self.t - 1]]))

        # determine the reward according to evaluation metrics
        self.r[self.t - 1] = -(self.b_cost[self.t - 1] + self.h_cost[self.t - 1])
        if self.t == self.seq_length - 10:
            self.dw = True
            if self.metric in [2, 4]:
                self.r[self.t - 1] += -(np.std(self.a[:self.t]) * 0.3636 * self.w) * self.t

        return np.concatenate((self.WIP[self.t, :], [self.NS[self.t]], self.demand[self.t: self.t + 6])), self.r[
            self.t - 1], self.dw  # S=[WIP, NS, Demand], r, dw