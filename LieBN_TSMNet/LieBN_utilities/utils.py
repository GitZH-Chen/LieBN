import datetime
import fcntl
import random

import numpy as np
import torch as th

def get_model_name(args):
    bias = 'bias' if args.learn_mean else 'non_bias'

    if args.model_type=='TSMNet+SPDDSMBN' or args.model_type=='TSMNet':
        name = f'{args.seed}-{args.lr}-wd_{args.weight_decay}-{args.model_type}-{args.optimizer}-{args.architecture}-{bias}-{datetime.datetime.now().strftime("%H_%M")}'
    elif args.model_type == 'TSMNet+LieBN':
        if args.metric == 'AIM' or args.metric == 'LEM':
            name = f'{args.seed}-{args.lr}-wd_{args.weight_decay}-{args.model_type}-{args.optimizer}-{args.architecture}-{bias}-{args.metric}-({args.theta},{args.alpha},{args.beta})-{datetime.datetime.now().strftime("%H_%M")}'
        elif args.metric== 'LCM':
            name = f'{args.seed}-{args.lr}-wd_{args.weight_decay}-{args.model_type}-{args.optimizer}-{args.architecture}-{bias}-{args.metric}-({args.theta})-{datetime.datetime.now().strftime("%H_%M")}'
    else:
        raise Exception('unknown metric {} or model'.format(args.metric,args.model_type))
    return name

def write_final_results(file_path,message):
    # Create a file lock
    with open(file_path, "a") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Acquire an exclusive lock
        # Write the message to the file
        file.write(message + "\n")
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Release the lock

def set_seed_thread(seed,threadnum):
    th.set_num_threads(threadnum)
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)