from utils import Actor, Double_Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy


class SAC_countinuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005

        self.actor = Actor(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e5), dvc=self.dvc)

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=torch.float32, requires_grad=True, device=self.dvc)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32, requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

    def select_action(self, state, deterministic):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)
            a, _ = self.actor(state, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def train(self, ):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
            target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (~dw) * self.gamma * (
                    target_Q - self.alpha * log_pi_a_next)  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.q_critic.parameters(): params.requires_grad = False

        a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters(): params.requires_grad = True

        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # def save(self,EnvName, timestep):
    # 	torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
    # 	torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

    def save(self, algo, EnvName, steps, para_rho, para_metric, idx):
        torch.save(self.actor.state_dict(),
                   "./model/{}_actor_{}_{}_{}_{}_{}.pth".format(algo, EnvName, steps, para_rho, para_metric, idx))
        torch.save(self.q_critic.state_dict(),
                   "./model/{}_q_critic_{}_{}_{}_{}_{}.pth".format(algo, EnvName, steps, para_rho, para_metric, idx))

    def load(self, algo, EnvName, steps, para_rho, para_metric, stage, collaboration_flag):
        path = ''
        if para_metric in [3, 4]:
            if collaboration_flag == 1:
                path += 'col/'
            else:
                path += 'sel/'

        if stage == 1:
            path += 'r/'
            eche = 'r'
        else:
            path += 'm/'
            eche = 'm'
        self.actor.load_state_dict(torch.load(
            "./model/metric_" + str(para_metric) + "/" + path + "{}_actor_{}_{}_{}_{}_{}.pth".format(algo, EnvName,
                                                                                                     steps,
                                                                                                     para_rho,
                                                                                                     para_metric,
                                                                                                     eche)))
        self.q_critic.load_state_dict(torch.load(
            "./model/metric_" + str(para_metric) + "/" + path + "{}_q_critic_{}_{}_{}_{}_{}.pth".format(algo, EnvName,
                                                                                                        steps,
                                                                                                        para_rho,
                                                                                                        para_metric,
                                                                                                        eche)))


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, dvc, max_size=int(1e5)):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        # consider the lead time, s_next, reward

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add(self, env, step):
        for i in range(step):
            if env.l > 1:
                self.s[i + step * self.ptr] = torch.from_numpy(
                    np.concatenate((env.WIP[i, :], [env.NS[i]], env.demand[i: i + 6])))
                self.s_next[i + step * self.ptr] = torch.from_numpy(
                    np.concatenate((env.WIP[i + 1, :], [env.NS[i + 1]], env.demand[i + 1: i + 1 + 6])))
            else:
                self.s[i + step * self.ptr] = torch.from_numpy(
                    np.concatenate(([env.NS[i]], env.demand[i: i + 6])))
                self.s_next[i + step * self.ptr] = torch.from_numpy(
                    np.concatenate(([env.NS[i + 1]], env.demand[i + 1: i + 1 + 6])))

            self.a[i + step * self.ptr] = env.a[i]
            # self.r[i + step * self.ptr] = env.r[i + env.l] # reward delayed
            self.r[i + step * self.ptr] = env.r[i]  # reward not delayed; almost no impact
            self.size = min(self.size + 1, self.max_size)
        self.ptr = (self.ptr + 1) % (self.max_size // step)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
    def clear(self):
        """Clear all contents in the buffer."""
        self.ptr = 0
        self.size = 0
        self.s.zero_()
        self.a.zero_()
        self.r.zero_()
        self.s_next.zero_()
        self.dw.zero_()
