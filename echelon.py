import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy

def generate_demand(mu, rho, sigma, length):
    d = np.zeros([length + 5])
    d[0] = mu
    for i in range(1, len(d)):
        d[i] = mu + rho * (d[i-1] - mu) + round(np.random.normal(0, sigma))
    return d




class echelon(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.seq_length = self.seq_len + 10 # to prevent bug in step func
        self.l = self.Lead_time
        self.NS = np.zeros(self.seq_length)
        self.WIP = np.zeros([self.seq_length, self.l-1])
        self.h_cost = np.zeros(self.seq_length)
        self.b_cost = np.zeros(self.seq_length)
        self.demand = np.zeros(self.seq_length + 5)
        self.a = np.zeros(self.seq_length)
        self.action_space = None
        self.t = 0
        self.h = 1
        self.b = 9
        self.r = np.zeros(self.seq_length)
        self.dw = False

    def reset(self):
        self.NS = np.zeros(self.seq_length)
        self.WIP = np.zeros([self.seq_length, self.l - 1])
        self.h_cost = np.zeros(self.seq_length)
        self.b_cost = np.zeros(self.seq_length)
        self.r = np.zeros(self.seq_length)
        self.a = np.zeros(self.seq_length)
        self.demand = generate_demand(self.mu, self.rho, self.sigma, self.seq_length)
        self.action_space = np.arange(0, self.action_dim, 1)
        self.t = 0

        self.NS[self.t] = self.mu - self.demand[self.t + 5]
        self.WIP[self.t, :] = np.ones(self.l - 1) * self.mu
        self.h_cost[self.t] = max(0, self.NS[self.t]) * self.h
        self.b_cost[self.t] = max(0, -self.NS[self.t]) * self.b
        self.r[self.t] = -(self.b_cost[self.t] + self.h_cost[self.t])
        self.a[self.t] = self.mu
        self.dw = False
        return np.concatenate((self.WIP[self.t,:], [self.NS[self.t]], self.demand[self.t: self.t+6])) # WIP, NS, Demand

    def step(self, action): # action: order quantity
        self.t += 1
        if self.t == self.seq_length - 5:
            self.dw = True

        self.a[self.t] = action
        self.NS[self.t] = self.NS[self.t - 1] + self.WIP[self.t - 1][0] - self.demand[self.t + 5]
        self.h_cost[self.t] = max(0, self.NS[self.t]) * self.h
        self.b_cost[self.t] = max(0, -self.NS[self.t]) * self.b
        self.WIP[self.t,:] = np.concatenate((self.WIP[self.t-1, 1:], [self.a[self.t - 1]]))  # CG May 1st, bug detected
        self.r[self.t] = -(self.b_cost[self.t] + self.h_cost[self.t])
        return np.concatenate((self.WIP[self.t,:], [self.NS[self.t]], self.demand[self.t: self.t+6])), self.dw  # WIP, NS, Demand