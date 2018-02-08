# DQN
Created by Pytorch

DQN Learning using numerical values returned from Cartpole 
I have completed applying the basic classic system ( CartPole-v1 , Acrobot-v1, MountainCar-v0 ).


the default setting is CartPole system.
x : position of cart on the track
θ : angle of the pole with the vertical
dx/dt : cart velocity
dθ/dt : rate of change of the angle


basic_DQN 
- The basic DQN was created by looking at nature 2015 paper.
- Paper : 

          Playing Atari with Deep Reinforcement Learning
          human level control through deep reinforcement learning
          
Double DQN
- To prevent overestimation of Q-learning algorithm 
- Paper : 
          
          Deep Reinforcement Learning with Double Q-learning

Dueling DQN
- dueling network represents two separate estimators: 
  one for the state value function and one for the state-dependent action advantage function.
- Paper : 

          Dueling Network Architectures for Deep Reinforcement Learning

Distributional DQN
- ....
- Paper : 

          A Distributional Perspective on Reinforcement Learning


Noisy DQN
- ....

Priority Experience Replay (PER) DQN
- proportional PER, using TD error 
- Paper : 

          PRIORITIZED EXPERIENCE REPLAY

1-Dimension CNN DQN
- Feature Extraction of 1-D Time Series Data Using conv1d CNN.
- Paper : 

Deep Recurrent Q Network (DRQN) using LSTM
- lstm is learned using last state , that returned the sequectial experience memory.
- but not conrrectly work. It works but do not learn.
- Paper : 

          DRQN_Playing FPS Games with Deep Reinforcement Learning 
          Asynchronous Methods for Deep Reinforcement Learning 
          DRQN with Prioritized Experience Replay, Double-Q
          

Multi-step DQN w/o Replay memory 
- n step learning , w/o memory , Using TD Error , Propotional PER 
- There is no experience memory, so can not efficiently learning.
- Paper : 

          Asynchronous Methods for Deep Reinforcement Learning 
          Multi-step Reinforcement Learning: A Unifying Algorithm
          
