import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy

def generate_demand(mu, rho, sigma, length):
    d = np.zeros([length])
    d[0] = mu
    for i in range(1, len(d)):
        d[i] = mu + rho * (d[i-1] - mu) + round(np.random.normal(0, sigma))
    return d




class echelon(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.seq_length = self.seq_len
        self.l = self.Lead_time
        self.NS = None
        self.WIP = None
        self.h_cost = None
        self.b_cost = None
        self.demand = None
        self.action_space = None
        self.t = 0
        self.h = 1
        self.b = 9
        self.r = None
        self.dw = False

    def reset(self):
        self.NS = round(np.random.normal(0, self.sigma))
        self.WIP = np.zeros([self.l - 1])
        for i in range(len(self.WIP)):
            self.WIP[i] = self.mu + round(np.random.normal(0, self.sigma))
        self.h_cost = max(0, self.NS) * self.h
        self.b_cost = max(0, -self.NS) * self.b
        self.demand = self.mu
        self.action_space = np.arange(0, self.action_dim, 1)
        self.t = 0
        self.r = -(self.b_cost + self.h_cost)
        return np.concatenate((self.WIP, [self.NS, self.demand])) # WIP, NS, Demand, holding cost, backlog cost

    def step(self, action): # action: order quantity
        self.t += 1
        if self.t == self.seq_length:
            self.NS = round(np.random.normal(0, self.sigma))
            self.WIP = np.zeros([self.l - 1])
            for i in range(len(self.WIP)):
                self.WIP[i] = self.mu + round(np.random.normal(0, self.sigma))
            self.h_cost = max(0, self.NS) * self.h
            self.b_cost = max(0, -self.NS) * self.b
            self.demand = self.mu
            self.t = 0
            self.r = -(self.b_cost + self.h_cost)
            self.dw = True
        else:
            self.dw = False
            self.r = -(self.h_cost + self.b_cost)

            self.NS = self.NS + self.WIP[0] - self.demand
            self.h_cost = max(0, self.NS) * self.h
            self.b_cost = max(0, -self.NS) * self.b

            # self.WIP = np.concatenate((self.WIP[1:], [action-self.mu])) # the quantity of demand should be centred around mu, e.g., from 1 to 20 if mu=10

            self.WIP = np.concatenate((self.WIP[1:], [action]))  # CG May 1st, bug detected

            self.demand = self.mu + self.rho * (self.demand - self.mu) + round(np.random.normal(0, self.sigma))
        return np.concatenate((self.WIP, [self.NS, self.demand])), self.r, self.dw  # WIP, NS, Demand, holding cost, backlog cost









