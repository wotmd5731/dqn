# -*- coding: utf-8 -*-

import argparse
import random
import torch
from torch.autograd import Variable
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


import gym
env = gym.make('CartPole-v0')


class DQN(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(4,32)
    self.fc2 = nn.Linear(32,2)
    
    
  def forward(self, x):
#    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return x



state = Variable(torch.zeros(1,4))

net = DQN()
optimizer = optim.SGD(net.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()  # LogSoftmax + ClassNLL Loss

for i_episode in range(20):
    state = env.reset()
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = Variable(state)
    
    
    prev_reward = 0
    for t in range(1000):
#        env.render()
        if random.random()<0.1 :
#            action = random.randrange(0,2)
            action = env.action_space.sample()
        else :
            action = net(state).data.max(0)[1][0]
        
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = Variable(state)
        
#        state = torch.from_numpy(state).type(torch.FloatTensor)
        
        
        
        if reward > prev_reward :
            target = action 
        else :
            target = not action
        
            
        prev_reward = reward
        

        optimizer.zero_grad()
        
        loss = loss(Variable(action), Variable(target))
        loss.backward()
        optimizer.step()    # Does the update
        
        
        if done:
            print("Episode finished after ",t+1,"timesteps ", i_episode, "episode")
            break
        
    
    
    