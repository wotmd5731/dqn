# -*- coding: utf-8 -*-


"""
프로그램은 도는데  
스코어가 안늘어남....

"""

import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#import torchvision.transforms as T


import argparse
from argument import get_args
args = get_args('DQN_LSTM')
args.game = 'CartPole-v1'
args.max_step = 500
args.action_space =2 
args.state_space = 4

from env import Env
env = Env(args)

args.memory_capacity = 1000000
args.learn_start = 1000000
#args.render= True
args.lr = 0.001
from memory import ReplayMemory , episodic_experience_buffer
#memory = LSTM_ReplayMemory(args)
memory = episodic_experience_buffer(args.memory_capacity)


from agent import Agent
agent = Agent(args,dqn_lstm=True)



"""
define test function
"""
from plot import _plot_line
current_time = time.time()
Ts, Trewards, Qs = [], [], []
def test(main_episode):
    global current_time
    prev_time = current_time
    current_time = time.time()
    T_rewards, T_Qs = [], []
    Ts.append(main_episode)
    total_reward = 0
    
    episode = 0
    
    
    while episode < args.evaluation_episodes:
        episode += 1
        T=0
        reward_sum=0
        state = env.reset()
        cx = Variable(torch.zeros(1,1, args.hidden_size))
        hx = Variable(torch.zeros(1,1, args.hidden_size))


        while T < args.max_step:
            T += 1
            if args.render:
                env.render()
                
            action,hx,cx = agent.get_action(state,hx,cx,evaluate=True)
            next_state , reward , done, _ = env.step(action)
            state = next_state
    
            total_reward += reward
            reward_sum += reward
            if done:
                
                break
        T_rewards.append(reward_sum)
    ave_reward = total_reward/args.evaluation_episodes
    # Append to results
    Trewards.append(T_rewards)
#        Qs.append(T_Qs)
    
    # Plot
    _plot_line(Ts, Trewards, 'rewards_'+args.name+args.game, path='results')
#        _plot_line(Ts, Qs, 'Q', path='results')
    
    # Save model weights
#        main_dqn.save('results')
    print('episode: ',main_episode,'Evaluation Average Reward:',ave_reward, 'delta time:',current_time-prev_time)
#            if ave_reward >= 300:
#                break
    


"""
randomize state push in memory
before main loop start
"""
global_count = 0
episode = 0
while True:
    
    episode += 1
    T=0
    ss = env.reset()
    cx = Variable(torch.zeros(1,1, args.hidden_size))
    hx = Variable(torch.zeros(1,1, args.hidden_size))
    episodeBuffer = []
    
    while T < args.max_step:
        a = random.randrange(0,args.action_space)
        ns , r , d, _ = env.step(a)
#        episodeBuffer.append(np.reshape(np.array([s,a,r,ns,d]),[5]))
        dd = ss.tolist()
        dd.append(a)
        dd.append(r)
        [dd.append(nn) for nn in ns.tolist() ]
        dd.append(int(d))
        
        episodeBuffer.append(np.array(dd))
        
#        memory.push([state, action, reward, next_state, done])
        ss = ns
        T += 1
        global_count += 1
        if d :
            memory.add(episodeBuffer)
            break
    print("\r push : %d/%d  "%(global_count,args.learn_start),end='\r',flush=True)
    if global_count > args.learn_start:
        break

print('')

"""
main loop
"""
global_count = 0
episode = 0
while episode < args.max_episode_length:
    episode += 1
    T=0
    s = env.reset()
    cx = Variable(torch.zeros(1,1, args.hidden_size))
    hx = Variable(torch.zeros(1,1, args.hidden_size))
    episodeBuffer = []
    
    while T < args.max_step:
        T += 1
        global_count += 1
        a,hx,cx = agent.get_action(s,hx,cx)
       
        ns , r , d, _ = env.step(a)
        if args.reward_clip > 0:
            r = max(min(r, args.reward_clip), -args.reward_clip)  # Clip rewards
        
        dd = ss.tolist()
        dd.append(a)
        dd.append(r)
        [dd.append(nn) for nn in ns.tolist() ]
        dd.append(int(d))
        
        episodeBuffer.append(np.array(dd))
#        memory.push([state, action, reward, next_state, done])
        s = ns
        
        if global_count % args.replay_interval == 0 :
            agent.lstm_learn(memory)
        if global_count % args.target_update_interval == 0 :
            agent.target_dqn_update()
            
            
        if d :
            memory.add(episodeBuffer)
            break
    if episode % args.evaluation_interval == 0 :
        test(episode)
