def evaluate_policy(env, agent, turns):
    total_scores = 0
    for j in range(turns):
        s = env.reset()
        for i in range(env.seq_length - 5):
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, dw = env.step(a)
            s = s_next
        total_scores += sum(env.r[:-5])

    return int(total_scores/turns/(env.seq_length - 5))


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