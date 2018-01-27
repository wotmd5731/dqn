# -*- coding: utf-8 -*-

import gym
import math
import copy
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#import torchvision.transforms as T



#cuda setting
#use_cuda = torch.cuda.is_available()
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor




class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, args):
        if len(self.memory) >self.capacity:
            #overflow mem cap pop the first element
            self.memory.pop(0)
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
        



"""
recode
deep + replay mem + seperate net (+cuda setting# 더 느림.)
3-layer fc 4->128->2 
target_interval =20 
batch 32
dis 0.9
epsil 0.2
lr=0.00001
mem 10000
episode:  0 Evaluation Average Reward: 53.9 delta time: 0.00601649284362793
episode:  100 Evaluation Average Reward: 9.2 delta time: 10.993052959442139
episode:  200 Evaluation Average Reward: 10.2 delta time: 3.4960572719573975
episode:  300 Evaluation Average Reward: 8.9 delta time: 3.5772788524627686
episode:  400 Evaluation Average Reward: 9.0 delta time: 3.407315731048584
episode:  500 Evaluation Average Reward: 9.3 delta time: 3.4298768043518066
episode:  600 Evaluation Average Reward: 9.7 delta time: 3.500068426132202
episode:  700 Evaluation Average Reward: 10.1 delta time: 3.719191312789917
episode:  800 Evaluation Average Reward: 10.1 delta time: 3.738720655441284
episode:  900 Evaluation Average Reward: 10.0 delta time: 3.789877414703369
episode:  1000 Evaluation Average Reward: 11.3 delta time: 3.8560426235198975
episode:  1100 Evaluation Average Reward: 12.8 delta time: 4.188951730728149
episode:  1200 Evaluation Average Reward: 9.6 delta time: 4.098704814910889
episode:  1300 Evaluation Average Reward: 9.4 delta time: 3.484025001525879
episode:  1400 Evaluation Average Reward: 14.1 delta time: 3.783843994140625
episode:  1500 Evaluation Average Reward: 14.0 delta time: 4.96256685256958
episode:  1600 Evaluation Average Reward: 15.9 delta time: 5.094928026199341
episode:  1700 Evaluation Average Reward: 15.2 delta time: 5.72114109992981
episode:  1800 Evaluation Average Reward: 19.6 delta time: 6.446122169494629
episode:  1900 Evaluation Average Reward: 25.1 delta time: 7.774754762649536
episode:  2000 Evaluation Average Reward: 25.9 delta time: 8.826680660247803
episode:  2100 Evaluation Average Reward: 29.5 delta time: 9.147507667541504
episode:  2200 Evaluation Average Reward: 73.2 delta time: 11.46989130973816
episode:  2300 Evaluation Average Reward: 18.5 delta time: 18.698139905929565
episode:  2400 Evaluation Average Reward: 86.7 delta time: 16.985936164855957
episode:  2500 Evaluation Average Reward: 126.4 delta time: 38.88582158088684
episode:  2600 Evaluation Average Reward: 114.7 delta time: 43.80441355705261
episode:  2700 Evaluation Average Reward: 123.6 delta time: 40.79264807701111
episode:  2800 Evaluation Average Reward: 199.4 delta time: 49.800164222717285

"""

#initilaize replay memory D to cap N
mem = ReplayMemory(10000)
main_dqn= DQN()
main_param=list(main_dqn.parameters())
if use_cuda:
    main_dqn.cuda()

target_dqn = DQN()
target_param=list(target_dqn.parameters())
#print("target update done ",main_param[0][0] , target_param[0][0])
if use_cuda:
    target_dqn.cuda()

target_dqn.load_state_dict(main_dqn.state_dict())
#target_param=list(target_dqn.parameters())
#print("target update done ",main_param[0][0] , target_param[0][0])


target_update_interval = 20

batch_size = 32
discount = 0.9

#e-greedy 
epsilon=0.2
env = gym.make('CartPole-v0')
#optimizer = optim.RMSprop(main_dqn.parameters())
optimizer = optim.Adam(main_dqn.parameters(), lr=0.00001)
criterion = nn.MSELoss() 


current_time = time.time()


def preprocess(state):
    return Variable(torch.from_numpy(state)).type(Tensor)


action = 0
target_update_count = 0

for episode in range(10000):
    
    state = env.reset()
    
#    state = preprocess(state).unsqueeze(0)
    
    
    for t in range(500):
        target_update_count +=1
        if random.random() <= epsilon :
            action = random.randrange(0,2)
        else :
            action = main_dqn(Variable(FloatTensor(state.reshape(1,4)),volatile=True)).max(1)[1].data[0] #return max index call [1] 
        next_state , reward , done, _ = env.step(action)
#        next_state = preprocess(next_state).unsqueeze(0)
#        reward = Tensor([reward])
#        action = Tensor([action])
#        done = Tensor([done])
        mem.push([state, action, reward, next_state, done])
        state = next_state
        
        if len(mem) > batch_size :
            batch = mem.sample(batch_size)
            [states, actions, rewards, next_states, dones] = zip(*batch)
            state_batch = Variable(Tensor(states))
            action_batch = Variable(LongTensor(actions))
            reward_batch = Variable(Tensor(rewards))
            next_states_batch = Variable(Tensor(next_states))
            
            state_action_values = main_dqn(state_batch).gather(1, action_batch.view(-1,1)).view(-1)
            # 32x1 return max action probability
            
            next_states_batch.volatile = True
            next_state_values = target_dqn(next_states_batch).max(1)[0]
            for i in range(batch_size):
                if dones[i]:
                    next_state_values.data[i]=0
            
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * discount) + reward_batch
            expected_state_action_values.volatile = True
            
            loss = criterion(state_action_values, expected_state_action_values)    
        
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in main_dqn.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            
#            for i in range(batch_size):
#                if done[i] == 1 :
#                    y = reward[i]
#                else :
#                    y = reward[i] + discount * main_dqn(next_state[i].unsqueeze(0)).max(1)[0].data[0]
##                loss = (y - main_dqn(state[i].unsqueeze(0)).max(1)[0].data[0])**2
#                loss = loss = F.smooth_l1_loss(y , main_dqn(state[i].unsqueeze(0)).max(1)[0])
#                optimizer.zero_grad()
#                loss.backward()
#                for param in main_dqn.parameters():
#                    param.grad.data.clamp_(-1, 1)
#                optimizer.step()

    
        if done :
            break
        if (target_update_count % target_update_interval) ==0 :
#            print("updata ",target_update_count)
            target_dqn.load_state_dict(main_dqn.state_dict())
    #            print("target update done ",main_param[0][0] , target_param[0][0])

        
    if episode % 100 == 0:
        global current_time
        prev_time = current_time
        current_time = time.time()
        
        test_cnt = 10
        total_reward = 0
        for i in range(test_cnt):
            test_state = env.reset()
            for j in count():
                env.render()
                test_action = main_dqn(Variable(FloatTensor(test_state.reshape(1,4)),volatile=True)).max(1)[1].data[0]
                test_state, test_reward, test_done, _ = env.step(test_action)
                total_reward += test_reward
                if test_done:
                    break
        ave_reward = total_reward/test_cnt
        print('episode: ',episode,'Evaluation Average Reward:',ave_reward, 'delta time:',current_time-prev_time)
#            if ave_reward >= 300:
#                break
    
    
    

