import math

from utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse
from datetime import datetime
from SAC import SAC_countinuous
import os, shutil
import argparse
import torch
import echelon
import numpy as np
import pandas as pd

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='Linear, FR, MOQ, CC')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=1000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(5e5), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(50e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=int(5e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')

parser.add_argument('--algo_idex', type=int, default=0, help='SAC, PPO')
parser.add_argument('--Lead_time', type=int, default=4, help='Lead time')  # Taka lead time, equals Steve lead time+1
parser.add_argument('--rho', type=float, default=0.9, help='AR coefficient')  # auto correlation parameter
parser.add_argument('--mu', type=float, default=20, help='AR average')  # mean
parser.add_argument('--sigma', type=float, default=2, help='AR std')  # std of the white noise
parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')  # sequence length for a single simulation run
parser.add_argument('--action_dim', type=int, default=1, help='Action dimension')
parser.add_argument('--metric', type=int, default=4, help='idx of evaluation metric')  # 1: only inventory cost; 2: both inventory and capacity cost
parser.add_argument('--m', type=int, default=5, help='length of past demand vector')  # information provided to the algorithm at each time step
parser.add_argument('--col', type=int, default=0, help='whether collaboration exists. 0: No; 1: Yes')  # fixed to 0 for one stage supply chain
parser.add_argument('--is_linear', type=int, default=1, help='if linearity assumption exists. 0: No; 1: Yes')
parser.add_argument('--is_train', type=str2bool, default=True, help='train or test')


opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)  # from str to torch.device
print(opt)


def main():
    idx = -math.inf
    EnvName = ['linear', 'FR', 'MOQ', 'CC']
    Scores = []

    opt.state_dim = opt.Lead_time - 1 + 1 + 6  # WIP_dim: l-1, NS_dim: 1, demand_dim: 6

    env = echelon.echelon(**vars(opt))
    eval_env = echelon.echelon(**vars(opt))

    # Algorithm Setting
    algo_names = ['SAC', 'PPO']
    algo_name = algo_names[opt.algo_idex]

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:', algo_name, '  Env:', EnvName[opt.EnvIdex], '  state_dim:', opt.state_dim, '  Random Seed:',
          opt.seed, '  max_e_steps:', opt.seq_len, '\n')

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name, EnvName[opt.EnvIdex], opt.seed) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    agent = SAC_countinuous(**vars(opt))
    if opt.Loadmodel: agent.load(algo_name, EnvName[opt.EnvIdex], "Opt", opt.rho)  # load the Opt model
    # if opt.Loadmodel: agent.load(algo_name, EnvName[opt.EnvIdex], opt.ModelIdex, opt.rho) # load model by steps index

    total_steps = 0
    while total_steps < opt.Max_train_steps:
        s = env.reset()
        env_seed += 1
        done = False

        '''Interact & trian'''
        while not done:
            # e-greedy exploration
            if total_steps < opt.random_steps:
                act = np.random.uniform(env.action_space[0], env.action_space[1], opt.action_dim)
            else:
                a = agent.select_action(s, deterministic=False)  # a∈[-1,1]
                # act = Action_adapter(a, opt.mu, opt.rho, opt.sigma)
                act = Action_adapter(a, opt.mu)

            s_next, r, dw = env.step(act)
            done = dw
            if done:
                agent.replay_buffer.add(env, env.seq_length - 10)  # record the training data when the simulation is finished
            s = s_next

            '''Update'''
            # train 50 times every 50 steps rather than 1 training per step. Better!
            if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                for j in range(opt.update_every):
                    agent.train()

            '''Record & Log'''
            if total_steps % opt.eval_interval == 0:
                score = evaluate_policy(eval_env, agent, turns=100)
                Scores.append(score)
                if opt.write:
                    writer.add_scalar('ep_r', score, global_step=total_steps)
                print('EnvName:', EnvName[opt.EnvIdex], 'seed:', opt.seed,
                      'steps: {}k'.format(int(total_steps / 1000)), 'score:', (score))
                '''save the best model'''
                if score >= idx:
                    idx = score
                    agent.save(algo_name, EnvName[opt.EnvIdex], "Opt", opt.rho, opt.metric, 'r')
                    print("Opt: " + str(int(total_steps / 1000)) + " " + str((score)))
            total_steps += 1

            '''save model'''
            # if total_steps % opt.save_interval == 0:
            #     agent.save(algo_name, EnvName[opt.EnvIdex], int(total_steps / 1000), opt.rho, opt.metric, 1)

    df = pd.DataFrame(Scores)
    df.to_csv('data/Scores_' + str(opt.rho) + '.csv', index=False)


if __name__ == '__main__':
    main()
