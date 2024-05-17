import math

from utils import evaluate_policy, str2bool
from datetime import datetime
from DQN import DQN_agent
import gymnasium as gym
import os, shutil
import argparse
import torch
import echelon
import numpy as np


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='Linear, FR, MOQ, CC')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=1000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(50e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=400, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')

parser.add_argument('--Lead_time', type=int, default=3, help='Lead time')
parser.add_argument('--rho', type=float, default=0, help='AR coefficient')
parser.add_argument('--mu', type=float, default=20, help='AR average')
parser.add_argument('--sigma', type=float, default=5, help='AR std')
parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
parser.add_argument('--action_dim', type=int, default=40, help='Action dimension, num of admissible order quantity')

opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def main():
    idx = -math.inf
    EnvName = ['linear', 'FR', 'MOQ', 'CC']



    opt.state_dim = opt.Lead_time + 1 + 5  # WIP_dim: l-1, NS_dim: 1, h_cost_dim: 1


    env = echelon.echelon(**vars(opt))
    eval_env = echelon.echelon(**vars(opt))


    #Algorithm Setting
    if opt.Duel: algo_name = 'Duel'
    else: algo_name = ''
    if opt.Double: algo_name += 'DDQN'
    else: algo_name += 'DQN'

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:',algo_name,'  Env:',EnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
          '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.seq_len, '\n')

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name,EnvName[opt.EnvIdex],opt.seed) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    #Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    if opt.Loadmodel: agent.load(algo_name,EnvName[opt.EnvIdex],"Opt",opt.rho) # load the Opt model
    # if opt.Loadmodel: agent.load(algo_name, EnvName[opt.EnvIdex], opt.ModelIdex, opt.rho) # load model by steps index

    score = evaluate_policy(eval_env, agent, turns=10)
    print('EnvName:', EnvName[opt.EnvIdex], 'score:', score)


if __name__ == '__main__':
    main()