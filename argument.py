# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable



def get_args(name):
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--name', type=str, default=name, help='stored name')
    
    parser.add_argument('--epsilon', type=float, default=0.33, help='random action select probability')
    
    
#    parser.add_argument('--render', type=bool, default=True, help='enable rendering')
    parser.add_argument('--render', type=bool, default=False, help='enable rendering')
    
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--game', type=str, default='CartPole-v1', help='gym game')
#    parser.add_argument('--game', type=str, default='Acrobot-v1', help='gym game')
#    parser.add_argument('--game', type=str, default='MountainCar-v0', help='gym game')
    parser.add_argument('--max-step', type=int, default=500, metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--action-space', type=int, default=2 ,help='game action space')
    parser.add_argument('--state-space', type=int, default=4 ,help='game action space')
    

    parser.add_argument('--max-episode-length', type=int, default=10000, metavar='LENGTH', help='Max episode length (0 to disable)')
    
    #parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
#    parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Network hidden size')
    
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
    
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    
    #parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=1000000, metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--learn-start', type=int, default=1000000 , metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--replay-interval', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
    
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent')
    parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    
    parser.add_argument('--target-update-interval', type=int, default=5, metavar='τ', help='Number of steps after which to update target network')
    
    parser.add_argument('--reward-clip', type=int, default=10, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')

    parser.add_argument('--max-gradient-norm', type=float, default=10, metavar='VALUE', help='Max value of gradient L2 norm for gradient clipping')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=10, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
    
    parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between logging status')
    
    # Setup
    args = parser.parse_args()
    " disable cuda "
    args.disable_cuda = True
#    args.disable_cuda = False
        
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
      print(' ' * 26 + k + ': ' + str(v))
    args.cuda = torch.cuda.is_available() and not args.disable_cuda
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if args.cuda:
        torch.cuda.manual_seed(random.randint(1, 10000))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else :
        torch.set_default_tensor_type('torch.FloatTensor')
      
    
    return args

    
