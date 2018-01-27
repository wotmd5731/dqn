# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:36:23 2018

@author: JAE
"""
import numpy as np
tt=[]

kk = [[5,5,5,5],[1],[2],[6,6,6,6]]

tt.append([[5,5,5,5],[1],[2],[6,6,6,6]]) 
tt.append([[5,5,5,5],[1],[2],[6,6,6,6]]) 
tt.append([[5,5,5,5],[1],[2],[6,6,6,6]]) 
tt.append([[5,5,5,5],[1],[2],[6,6,6,6]]) 
tt.append([[5,5,5,5],[1],[2],[6,6,6,6]]) 

#print(tt)
#        for episode in sampled_episodes:
#            point = np.random.randint(0,len(episode)+1-trace_length)
#            sampledTraces.append(episode[point:point+trace_length])
#        sampledTraces = np.array(sampledTraces)
#        return np.reshape(sampledTraces,[batch_size*trace_length,5])


aa=np.array(tt)
print(aa)