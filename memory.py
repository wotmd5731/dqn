# -*- coding: utf-8 -*-
import random
from collections import namedtuple
import torch
from torch.autograd import Variable

import numpy as np


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PER_Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def push(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch_idx = []
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( data) 
            batch_idx.append(idx)

        return batch , batch_idx

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)




class ReplayMemory(object):
    def __init__(self, args):
        self.capacity = args.memory_capacity
        self.memory = []

    def push(self, args):
        if len(self.memory) > self.capacity:
            #overflow mem cap pop the first element
            self.memory.pop(0)
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class episodic_experience_buffer():
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    # 더할 때 버퍼사이즈를 넘으면, 앞에서부터 지우고 다시 넣는다.
    def add(self,experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
            
    def sample(self,batch_size,trace_length):
        sampled_episodes = random.sample(self.buffer,batch_size)
        sampledTraces = []
        # 이전과 다른 부분, 샘플로 뽑힌 에피소드에서 지정된 크기만큼의 걸음(프레임)을 붙여서 가져온다.
        for episode in sampled_episodes:
            point = np.random.randint(0,len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces,[trace_length,batch_size,11])
    



#class LSTM_ReplayMemory(object):
#    def __init__(self, args):
#        self.capacity = args.memory_capacity
#        self.memory = []
#
#    def push(self, args):
#        if len(self.memory) > self.capacity:
#            #overflow mem cap pop the first element
#            self.memory.pop(0)
#        self.memory.append(args)
#
#    def sample(self,batch,trace): 
#        
##        for episode in sampled_episodes:
##            point = np.random.randint(0,len(episode)+1-trace_length)
##            sampledTraces.append(episode[point:point+trace_length])
##        sampledTraces = np.array(sampledTraces)
##        return np.reshape(sampledTraces,[batch_size*trace_length,5])
#        
#        sss=[]
#        aaa=[]
#        rrr=[]
#        sss_=[]
#        ddd=[]
#            
#        for bb in range(batch):
#            # seq_len(timestamp), batch, input_size
#            num = random.randint(0,len(self.memory)-1-trace)
#            ss=[]
#            aa=[]
#            rr=[]
#            ss_=[]
#            dd=[]
#            t = 0
#            while t<trace: # 
#                ss.append(self.memory[num+t][0].reshape(1,1,4))
#                aa.append(numpy.array(self.memory[num+t][1]).reshape(1,1,1))
#                rr.append(numpy.array(self.memory[num+t][2]).reshape(1,1,1))
#                ss_.append(self.memory[num+t][3].reshape(1,1,4))
#                dd.append(numpy.array(self.memory[num+t][4]).reshape(1,1,1).astype('int'))
#                t += 1
#            sss.append(numpy.vstack(ss))
#            aaa.append(numpy.vstack(aa))
#            rrr.append(numpy.vstack(rr))
#            sss_.append(numpy.vstack(ss_))
#            ddd.append(numpy.vstack(dd))
#      
#        
#
#        sss = numpy.hstack(sss)
#        aaa = numpy.hstack(aaa)
#        rrr = numpy.hstack(rrr)
#        sss_ = numpy.hstack(sss_)
#        ddd = numpy.hstack(ddd)
#        
#        return [sss,aaa,rrr,sss_,ddd]
#        
#        
#
#
#    def __len__(self):
#        return len(self.memory)




