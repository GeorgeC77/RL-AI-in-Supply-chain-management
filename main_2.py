import math

from utils_2 import evaluate_policy, str2bool
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
parser.add_argument('--random_steps', type=int, default=int(5e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')

parser.add_argument('--Lead_time', type=int, default=3, help='Lead time')
parser.add_argument('--rho', type=float, default=0.9, help='AR coefficient')
parser.add_argument('--mu', type=float, default=20, help='AR average')
parser.add_argument('--sigma', type=float, default=2, help='AR std')
parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
parser.add_argument('--action_dim', type=int, default=40, help='Action dimension, num of admissible order quantity')
parser.add_argument('--metric', type=int, default=4, help='idx of evaluation metric')
parser.add_argument('--m', type=int, default=5, help='length of past demand vector')
parser.add_argument('--col', type=int, default=1, help='whether collaboration exists. 0: No; 1: Yes')

opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def main():
    idx = -math.inf
    EnvName = ['linear', 'FR', 'MOQ', 'CC']



    opt.state_dim = opt.Lead_time - 1 + 1 + 6  # WIP_dim: l-1, NS_dim: 1, demand_dim: 6

    # retailer
    env = echelon.echelon(**vars(opt))
    eval_env = echelon.echelon(**vars(opt))

    # manufacturer
    env_2 = echelon.echelon_m(**vars(opt))
    eval_env_2 = echelon.echelon_m(**vars(opt))


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
    agent_2 = DQN_agent(**vars(opt))
    if opt.Loadmodel: agent.load(algo_name, EnvName[opt.EnvIdex], "Opt", opt.rho, opt.metric, 1)  # load the Opt model for the retailer

    if opt.col == 1: # load the Opt model for the manufacturer in collaboration scenario
        agent_2.load(algo_name, EnvName[opt.EnvIdex], "Opt", opt.rho, opt.metric, 2)

    total_steps = 0
    while total_steps < opt.Max_train_steps:
        s = env.reset()

        env_seed += 1
        done = False

        '''Interact & trian'''
        while not done:
            a = agent.select_action(s, deterministic=False) # no need for random serach
            s_next, r, dw = env.step(a)
            done = dw
            s = s_next
            if done:
                # the manufacturer's action
                s_2 = env_2.reset(env)
                done_2 = False
                while not done_2:
                    # e-greedy exploration only exists in the non-collaboration scenario
                    if total_steps < opt.random_steps and opt.col == 0:
                        a_2 = np.random.choice(env_2.action_space)
                    else:
                        a_2 = agent_2.select_action(s_2, deterministic=False)
                    s_next_2, r_2, dw_2 = env_2.step(a_2)
                    done_2 = dw_2
                    s_2 = s_next_2
                if opt.col == 1: # in the collaboration scenario, the retailer also consider the manufacturer's cost
                    env.r += env_2.r
                agent.replay_buffer.add(env, env.seq_length - 10)  # record the training data when the simulation is finished
                agent_2.replay_buffer.add(env_2, env_2.seq_length - 10)

            '''Update'''
            # train 50 times every 50 steps rather than 1 training per step. Better!
            if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                for j in range(opt.update_every):
                    if opt.col: # train the retailer's model only in collaboration scenario
                        agent.train()
                    agent_2.train()

            '''Noise decay & Record & Log'''
            if total_steps % 1000 == 0:
                agent.exp_noise *= opt.noise_decay
                agent_2.exp_noise *= opt.noise_decay
            if total_steps % opt.eval_interval == 0:
                score, score_2 = evaluate_policy(eval_env, agent, eval_env_2, agent_2, turns=100)
                if opt.write:
                    writer.add_scalar('ep_r', score, global_step=total_steps)
                    writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                print('EnvName:', EnvName[opt.EnvIdex], 'seed:', opt.seed,
                      'steps: {}k'.format(int(total_steps / 1000)), 'scores:', (score), (score_2))
                '''save the best model'''
                if opt.col == 1: # in collaboration scenario, consider both the echelon's costs
                    Score = score + score_2
                else:
                    Score = score_2
                if Score >= idx:
                    idx = Score
                    if opt.col == 1:
                        agent.save(algo_name, EnvName[opt.EnvIdex], "Opt", opt.rho, opt.metric, 'r') # save the retailer's model in the collaboration scenario
                    agent_2.save(algo_name, EnvName[opt.EnvIdex], "Opt", opt.rho, opt.metric, 'm')
                    print("Opt: "+str(int(total_steps / 1000))+" "+str((Score)))
            total_steps += 1

            '''save model'''
            # if total_steps % opt.save_interval == 0:
            #     agent.save(algo_name, EnvName[opt.EnvIdex], int(total_steps / 1000), opt.rho, opt.metric, 1)
            #     agent_2.save(algo_name, EnvName[opt.EnvIdex], int(total_steps / 1000), opt.rho, opt.metric, 2)

if __name__ == '__main__':
    main()