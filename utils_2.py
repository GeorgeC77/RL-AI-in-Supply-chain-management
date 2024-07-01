import pandas as pd
import numpy as np

def evaluate_policy(env, agent, env_2, agent_2, turns):
    total_scores = 0
    total_scores_2 = 0
    action = []
    for z in range(turns):
        done = False
        s = env.reset()
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw = env.step(a)
            s = s_next
            action.append(a)
            done = dw
        total_scores += np.sum(env.r[:env.seq_length - 10])
        s_2 = env_2.reset(env)
        done_2 = False
        while not done_2:
            a_2 = agent_2.select_action(s_2, deterministic=True)
            s_next_2, r_2, dw_2 = env_2.step(a_2)
            s_2 = s_next_2
            done_2 = dw_2
        total_scores_2 += np.sum(env_2.r[6:env.seq_length - 10])
        # if z == 0:
        #     df = pd.DataFrame(action)
        #     df.to_csv('data/order.csv', index=False)

    return (total_scores/turns/(env.seq_length - 10), total_scores_2/turns/(env_2.seq_length - 10 - 6))


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise