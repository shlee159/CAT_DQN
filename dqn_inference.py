import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import math 
import random 
import pandas as pd
import datetime

from CATENV import CATEnv
from memory_dqn import ReplayMemory
from models_dqn import *

import argparse

train_path = "train_data"
PATH = "inference_data"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('iterations', type=int)
    parser.add_argument('--action', default='dqn', type=str, required=False)
    parser.add_argument('--period', default='200', type=str, required=False, help="perf sampling period")

    args = parser.parse_args()
    return args

# DQN_INFERENCE
def select_action_dqn(state):
    global policy_net

    with torch.no_grad():
        state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).float()
        # print(policy_net(state_cat,state_core).max(1)[1].view(1,1))
        policy_net.eval()
        return policy_net(state).max(1)[1].view(1,1)

# Random policy
def select_action_random(state):
    return torch.tensor([[random.randrange(5)]], device=device, dtype=torch.long)


def inference(env, num_iterations):
    global steps_done

    # observation = np.zeros(6*8)
    observation = env.reset()

    while steps_done < num_iterations:

        action = select_action(observation)
        new_observation, reward = env.step(action.item())
        print("step : ", steps_done, " action : ", action, " reward : ", reward)

        reward_list.append(reward)
        action_list.append(action.to('cpu'))
        # sum_reward += reward
        # reward = torch.tensor([reward], device=device)
        
        observation = new_observation

        steps_done += 1

    env.finish()



##
def pmu_callback(pmu_dicts):
    for cpuIdx in range(8):
        pmu_callback.times[cpuIdx].append(pmu_dicts[cpuIdx]["time"])
        pmu_callback.instructions[cpuIdx].append(pmu_dicts[cpuIdx]["r0C0"])
        pmu_callback.cycles[cpuIdx].append(pmu_dicts[cpuIdx]["r03C"])
        # 
        # pmu_callback.cycles_l2_miss[cpuIdx].append(pmu_dicts[cpuIdx]["r01A3"])
        # pmu_callback.llc_references[cpuIdx].append(pmu_dicts[cpuIdx]["cache-references"])
        # pmu_callback.llc_misses[cpuIdx].append(pmu_dicts[cpuIdx]["cache-misses"])
pmu_callback.times = []        
pmu_callback.instructions = []
pmu_callback.cycles = []
for coreIdx in range(8):
    pmu_callback.times.append([])
    pmu_callback.instructions.append([])
    pmu_callback.cycles.append([])
##


# Main
if __name__=='__main__':
    args = parse_args()

    # set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    if args.action == "dqn":
        select_action = select_action_dqn
    elif args.action == "random":
        select_action = select_action_random
    #
    reward_list=[]
    action_list=[]

    # create networks
    policy_net = DQN(n_actions=5).to(device)
    # policy_net = DQN(n_actions=9).to(device)
    policy_net.load_state_dict(torch.load(train_path+"/dqn_model_state.pt"))

    start_time = datetime.datetime.now()
    print(start_time)

    # create environment
    env = CATEnv(args.period)
    env.run(pmu_callback)

    steps_done = 0
    inference(env, args.iterations) #

    end_time = datetime.datetime.now()
    print(end_time)
    print("elapsed time : ", end_time - start_time)

    import os
    os.makedirs(PATH, exist_ok=True)

    results = {'reward': reward_list, 'action': action_list}
    df = pd.DataFrame(results)
    df.to_csv(PATH+"/pandas_data.csv")
    #
    np.savetxt(PATH+"/reward.csv", reward_list, delimiter=",")
    np.savetxt(PATH+"/action.csv", action_list, delimiter=",")
    for i in [0,1,2,3,4,5,6,7]:
        np.savetxt(PATH+"/core"+str(i)+"_times.txt", pmu_callback.times[i], delimiter=',')
        np.savetxt(PATH+"/core"+str(i)+"_instructions.txt", pmu_callback.instructions[i], delimiter=',')
        np.savetxt(PATH+"/core"+str(i)+"_cycles.txt", pmu_callback.cycles[i], delimiter=',')