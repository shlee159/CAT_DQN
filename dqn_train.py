import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import math
import random
import pandas
import datetime

from CATENVtrain import CATEnv
from memory_dqn import ReplayMemory
from models_dqn import *

PATH = "train_data"

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)

def select_action(state):
    global steps_done
    
    slope = ##
    eps_threshold = slope * steps_done +1
    # print("epsilon : ", eps_threshold)
    epsilons.append(eps_threshold)  # save data

    state = np.expand_dims(state, axis=0)
    state = torch.from_numpy(state).float()

    with torch.no_grad():
        policy_net.eval()
        qvalues = policy_net(state)
        policy_net.train()

        if random.random() > eps_threshold:
            print("maximum qvalued action is selected")
            maxv = qvalues.max(1)
            
            reward_predict_list.append(maxv[0].view(1,1).item())    # save data
            qactions.append(maxv[1].view(1,1).item())   # save data
            qa_or_ra.append(1)   # save data

            return maxv[1].view(1,1)
        
        else:
            qa = torch.tensor([[random.randrange(5)]], device=device, dtype=torch.long)
            
            reward_predict_list.append(qvalues[0][qa.item()].item())    # save data
            qa_or_ra.append(0)  # save data

            return qa


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)

    """
    zip(*transitions) unzips the transitions into 
    Trnasition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)
    """
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))) 

    state_batch = torch.from_numpy(np.array(batch.state)).float()
    next_state_batch = torch.from_numpy(np.array(batch.next_state)).float()
    reward_batch = torch.cat(rewards)
    action_batch = torch.cat(actions)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()

    reward_batch=reward_batch.to(torch.float32)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    losses.append(loss.item())  # save data
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def train(env, num_iterations):
    global steps_done

    # observation = np.zeros(6*8)
    observation = env.reset()

    while steps_done < num_iterations:
        action = select_action(observation)
        new_observation, reward = env.step(action.item())
        print("step : ", steps_done, " action : ", action, " reward : ", reward)
        reward_list.append(reward)  # save data
        action_list.append(action.to('cpu'))    # save data

        reward = torch.tensor([reward], device=device)
        memory.push(observation, action.to('cpu'), new_observation, reward.to('cpu'))
        observation = new_observation

        if steps_done > INITIAL_MEMORY:
            optimize_model()

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        steps_done += 1
        if steps_done % 10000==0:
            torch.save(policy_net.state_dict(), PATH+"/model/dqn_model_state"+str(steps_done//10000)+".pt")
            torch.save(policy_net, PATH+"/model/dqn_model"+str(steps_done//10000))
    
    env.finish()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    lr = 1e-4
    #
    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY = 1000000
    #
    TARGET_UPDATE=1000
    INITIAL_MEMORY=1000
    MEMORY_SIZE=10*INITIAL_MEMORY

    # saving data
    reward_list = []
    action_list = []
    reward_predict_list = []
    epsilons = []
    losses = []
    qactions = []
    qa_or_ra = []


    policy_net = DQN(n_actions=5).to(device)
    target_net = DQN(n_actions=5).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    memory = ReplayMemory(MEMORY_SIZE)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)


    start_time = datetime.datetime.now()
    print(start_time)

    env = CATEnv()
    steps_done = 0
    train(env, ###)

    end_time = datetime.datetime.now()
    print(end_time)
    print("time : ", end_time - start_time)


    # save model
    torch.save(policy_net.state_dict(), PATH+"/dqn_model_state.pt")
    torch.save(policy_net, PATH+"/dqn_model")

     # save data
    dict1 = {'reward': reward_list, 'action': action_list, 'reward_predict':reward_predict_list}
    df = pd.DataFrame(dict1)
    df.to_csv(PATH+"/pandas_data.csv")
    #
    np.savetxt(PATH+"/reward.csv", reward_list, delimiter=",")
    np.savetxt(PATH+"/action.csv", action_list, delimiter=",")
    np.savetxt(PATH+"/reward_predict.csv", reward_predict_list, delimiter=",")
    np.savetxt(PATH+"/epsilon.csv", epsilons, delimiter=",")
    np.savetxt(PATH+"/losses.csv", losses, delimiter=",")
    np.savetxt(PATH+"/qactions.csv", qactions, delimiter=",")
    np.savetxt(PATH+"/qa_or_ra.csv", qa_or_ra, delimiter=",")