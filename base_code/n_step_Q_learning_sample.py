# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:33:00 2018

@author: JAE
"""


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



#initialize thread step counter  t <- 1
#initialize target network parameters  theta^- <- theta
#initialize thread specific parameters  theta' = theta
#initialize network gradients d theta <- 0

optimizer = optim.Adam(main_net.parameters(), lr=0.0001)

main_net
target_net


#repeat
while(T<T_MAX):
    #clear gradients dtheta <- 0
    optimizer.zero_grad()
    #synchronize thread specific parameters theta' = theta
#    thread_net = main_net
    target_net = main_net
    t_start = t
    state = env.reset()


    while terminal(state) or t-t_start == t_max:
        #Take action at according to the e-greedy policy based on Q(st, a; theta')
        if random.random() <= epsilon :
            action = random.randrange(0,env_action_size)
        else :
            action = main_net(Variable(FloatTensor(state.reshape(1,env_state_size)),volatile=True)).max(1)[1].data[0] #return max index call [1]
        #receive reward r_t and new state s_t+1
        next_state , reward , done, _ = env.step(action)
        t = t+1
        T = T+1

    #R if terminal st then 0, if non-terminal  st then max_a Q(st,a;theta^-)
    if terminal(state):
        R = 0
    else:
        main_net(state).max(1)
    for i in range(t-1,t_start):
        R =  reward[i]+gamma*R
        #accumlate gradients wrt theta'
        #d theta = d theata + partial_D (R-Q(si,ai;theta'))^2 / partial_D ( theta')

    #perfom aynchoronous update of theta using d_theta

    if T % INTERVAL :
        #theta sync

