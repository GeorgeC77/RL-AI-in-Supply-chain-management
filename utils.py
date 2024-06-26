import pandas as pd
import numpy as np

def evaluate_policy(env, agent, turns):
    total_scores = 0

    for j in range(turns):
        action = []
        s = env.reset()
        for i in range(env.seq_length - 10):
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw = env.step(a)
            s = s_next
            total_scores += r

            action.append(a)
        # print(np.std(action))
        # if j == 0:
        #     df = pd.DataFrame(action)
        #     df.to_csv('data/order.csv', index=False)

    return (total_scores/turns/(env.seq_length - 10))


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