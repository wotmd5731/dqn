# DQN
Created by Pytorch



basic_DQN 
- The basic DQN was created by looking at nature 2015 paper.

Double DQN
- to Prevent Q-learning algorithm overestimation
- Paper : Deep Reinforcement Learning with Double Q-learning

Dueling DQN
- dueling network represents two separate estimators: 
  one for the state value function and one for the state-dependent action advantage function.
- Paper : Dueling Network Architectures for Deep Reinforcement Learning

Distributional DQN
- ....

Noisy DQN
- ....

Priority Experience Replay (PER) DQN
- proportional PER, using TD error 

1-Dimension CNN DQN
- Feature Extraction of 1-D Time Series Data Using conv1d CNN.

Deep Recurrent Q Network (DRQN) using LSTM
- lstm is learned using last state , that returned the sequectial experience memory.
- but not conrrectly work. It works but do not learn.

Multi-step DQN w/o Replay memory 
- n step learning , w/o memory , 
- There is no experience memory, so can not efficiently learning.

