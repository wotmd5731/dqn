# -*- coding: utf-8 -*-
import torch
import gym

class Env():
  def __init__(self, args):
    super().__init__()
    self.actions = [ 0, 1 ]
    self.env = gym.make(args.game)
    self.env = self.env.unwrapped
    
  def reset(self):
    return self.env.reset()

  def step(self, action):
    return self.env.step(action)
  def render(self):
    self.env.render()
      


  def action_space(self):
    return len(self.actions)


class Env_CNN():
  def __init__(self, args):
    super().__init__()
    self.actions = [ 0, 1 ]
    self.env = gym.make(args.game)
    self.seq_size = 50
    self.mem = []
    for i in range(self.seq_size):
        self.mem.append([0,0,0,0])
        
  def reset(self):
      state = self.env.reset()
      self.mem.pop(0)
      self.mem.append(state)
      return self.mem

  def step(self, action):
      ss_, rr,  dd, _ = self.env.step(action)
      self.mem.pop(0)
      self.mem.append(ss_)
      return self.mem, rr,dd ,_
  
  def render(self):
    self.env.render()
      


  def action_space(self):
    return len(self.actions)
