# -*- coding: utf-8 -*-
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
    
from model import DQN,Dueling_DQN,Noisy_Distributional_Dueling_DQN , DQN_CNN,DQN_LSTM


use_cuda = torch.cuda.is_available() and False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
glob_count=0    

class Agent(nn.Module):
    
    def __init__(self,args,dueling=False,noisy_distributional=False,dqn_cnn=False,dqn_lstm=False):
        super().__init__()
        
        self.batch_size = args.batch_size 
        self.discount = args.discount
        self.max_gradient_norm = args.max_gradient_norm
        self.epsilon = args.epsilon
        self.action_space = args.action_space
        self.hidden_size = args.hidden_size
        self.dqn_lstm = False
        self.noisy_distributional=False
        self.dueling=False
        self.dqn_cnn=False
        self.state_space = args.state_space
        
        
        if noisy_distributional:
            self.main_dqn= Noisy_Distributional_Dueling_DQN(args)
            self.target_dqn = Noisy_Distributional_Dueling_DQN(args)
            self.atoms = args.atoms
            self.v_min = args.V_min
            self.v_max = args.V_max
            self.support = torch.linspace(args.V_min, args.V_max, args.atoms)  # Support (range) of z
            self.delta_z = (args.V_max - args.V_min) / (args.atoms - 1)
            self.m = torch.zeros(args.batch_size, self.atoms).type(torch.FloatTensor)
            self.noisy_distributional = True
            
        elif dueling:
            self.dueling=True
            self.main_dqn= Dueling_DQN(args)
            self.target_dqn = Dueling_DQN(args)
        elif dqn_cnn:
            self.dqn_cnn=True
            self.main_dqn= DQN_CNN(args)
            self.target_dqn = DQN_CNN(args)
        elif dqn_lstm:
            self.dqn_lstm=True
            self.main_dqn= DQN_LSTM(args)
            self.target_dqn = DQN_LSTM(args)
        
        else:
            self.main_dqn= DQN(args)
            self.target_dqn = DQN(args)
        
        
        if args.cuda:
            self.main_dqn.cuda()
            self.target_dqn.cuda()
        
        self.target_dqn_update()
        #target_dqn.load_state_dict(main_dqn.state_dict())
        #target_param=list(target_dqn.parameters())
        #print("target update done ",main_param[0][0] , target_param[0][0])
        self.optimizer = optim.Adam(self.main_dqn.parameters(), lr=args.lr)

#        plt.show()
        
    def target_dqn_update(self):
        self.target_dqn.parameter_update(self.main_dqn)
        
    def get_action(self,state,hx=0,cx=0,evaluate = False):
        action=0
        global glob_count
        if self.dqn_lstm:
            if random.random() <= self.epsilon and not evaluate:
                action ,hx,cx = self.main_dqn(Variable(FloatTensor(state.reshape(1,1,self.state_space)),volatile=True),hx,cx) #return max index call [1] 
                action = random.randrange(0,self.action_space)
            else:
                action ,hx,cx = self.main_dqn(Variable(FloatTensor(state.reshape(1,1,self.state_space)),volatile=True),hx,cx) #return max index call [1] 
                action = action.max(2)[1].data[0][0]
#                print(action)
            return action,hx,cx
            
        
        
        if random.random() <= self.epsilon and not evaluate:
            action = random.randrange(0,self.action_space)
        elif self.noisy_distributional :
#            ps = self.main_dqn(Variable(FloatTensor(state.reshape(1,4)),volatile=True)).data
    
#            plt.clf()
#            plt.plot(self.support.numpy(),ps.numpy()[0][0],\
#                     self.support.numpy(),ps.numpy()[0][1])
##            plt
#            plt.draw()
#            plt.pause(0.0001)
            action = (self.main_dqn(Variable(FloatTensor(state.reshape(1,self.state_space)),volatile=True)).data * self.support).sum(2).max(1)[1][0]
        elif self.dqn_cnn:
            action = self.main_dqn(Variable(state,volatile=True)).max(1)[1].data[0]
        
        
        else :
            ret = self.main_dqn(Variable(FloatTensor(state.reshape(1,self.state_space)),volatile=True))
#            glob_count += 1
#            if glob_count%100 ==0 :
#                print(ret.data)
            action = ret.max(1)[1].data[0] #return max index call [1] 
        return action
        
    
    def basic_learn(self,memory):
        batch = memory.sample(self.batch_size)
        [states, actions, rewards, next_states, dones] = zip(*batch)
        state_batch = Variable(Tensor(states))
        action_batch = Variable(LongTensor(actions))
        reward_batch = Variable(Tensor(rewards))
        next_states_batch = Variable(Tensor(next_states))
        
        state_action_values = self.main_dqn(state_batch).gather(1, action_batch.view(-1,1)).view(-1)
        next_states_batch.volatile = True
        next_state_values = self.target_dqn(next_states_batch).max(1)[0]
        for i in range(self.batch_size):
            if dones[i]:
                next_state_values.data[i]=0
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount) + reward_batch
        expected_state_action_values.volatile = False
        loss = F.mse_loss(state_action_values, expected_state_action_values)        
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
            
    def lstm_learn(self,memory):
        seq_len = 8
        batch_size =32
        data = memory.sample(batch_size,seq_len)
        ss = data[:,:,0:4]
        aa = data[:,:,4]
        rr = data[:,:,5]
        ss_ = data[:,:,6:10]
        dd = data[:,:,10]
        # seq_len(timestamp), batch, input_size
        ss = Variable(Tensor(ss),volatile=True).view(batch_size,seq_len,self.state_space)
        aa = Variable(Tensor(aa).type(LongTensor).unsqueeze(2),volatile=True).view(batch_size,seq_len,1)
        rr = Variable(Tensor(rr).unsqueeze(2),volatile=True).view(batch_size,seq_len,1)
        ss_ = Variable(Tensor(ss_),volatile=True).view(batch_size,seq_len,self.state_space)
        dd = Variable(Tensor(dd).unsqueeze(2),volatile=True).view(batch_size,seq_len,1)
        
        cx = Variable(torch.zeros(1,1, self.hidden_size))
        hx = Variable(torch.zeros(1,1, self.hidden_size))
        Q,hx,cx = self.main_dqn(ss[:,:4,:],hx,cx)
        for ii in range(4,8):
            idd = 1 - dd[:,ii,:]
            irr = rr[:,ii,:]
            iaa = aa[:,ii,:]
            Q1, _, _ = self.target_dqn(ss_[:,ii,:].unsqueeze(1),hx,cx)
            target_act_val = (irr + self.discount * Q1.max(2)[0]*idd).view(-1)
            Q2,hx,cx = self.main_dqn(ss[:,ii,:].unsqueeze(1),hx,cx)
            main_act_val = Q2.gather(2,iaa.unsqueeze(1)).view(-1)
            main_act_val.requires_grad=True
            main_act_val.volatile = False
#            loss = nn.MSELoss(Q1_act_val , Q2_act_val).mean()
            loss = F.mse_loss(main_act_val, target_act_val)
#            loss =(( Q1_act_val - Q2_act_val)**2).mean()
            loss.requires_grad=True
            loss.volatile = False
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
            self.optimizer.step()
        
        return
        
        hx_tau_1 = hx
        cx_tau_1 = cx
        Q,hx,cx = self.main_dqn(ss[:,7,:].unsqueeze(1),hx,cx)
        
#        q = np.max(self.target_model.get_q(s2, state_in), axis=1)
#        targets = r + self.gamma * (1 - d) * q
#        self.model.learn(inputs, targets, state_in, a)

        
                
        Q1, _, _ = self.target_dqn(ss_[:,-1,:].unsqueeze(1),hx,cx)
        Q1_act_val = (rr + self.discount * Q1.max(2)[0]*dd).view(-1)
        
        Q2,_,_ = self.main_dqn(ss[:,-1,:].unsqueeze(1),hx_tau_1,cx_tau_1)
        Q2_act_val = Q2.gather(2,aa.unsqueeze(1)).view(-1)
        
        
        Q2.requires_grad=True
        Q2_act_val.requires_grad=True
        loss =(( Q1_act_val - Q2_act_val)**2).mean()
        
        """여기까지 라스트 loss Back propagation 앞내용은 전부 hxcx 생성용."""
        
#        Q1, hx,cx = self.main_dqn(ss_[0:7],hx,cx)
#        Q2, _, _ = self.target_dqn(ss_[0:4],hx,cx)
        
#        Q1, hx,cx = self.main_dqn(ss_[4:7],hx,cx)
#        Q2, _, _ = self.target_dqn(ss_[7],hx,cx)
        
        '''            
        0~4번까진 이전 상태의 히스토리를 초기화 하는 용도.
        4~7까지 학습 하는 용도.  backprop 적용.
        7의 경우 target net action state 구하는 용도
        
        '''

        
        ''' 
        음.... 뭐 loss 구해서 back prop 하면 끝인데....  dd 가 종료 케이스 에 대한 처리를 어찌 해야될지 모르겟음.. 
        일단은 done = 1 이후 존재하는 시퀀스에 대해서 done =1 로 연속 변경 은 햇는데...
        그래서 state 가 다르기 때문에 이걸 학습하기 때문에 이렇게 하면 안될것같은데 찾아봐야됨..
        '''
        loss =(( Q1_act_val - Q2_act_val)**2).mean()
        self.main_dqn.zero_grad()
        self.target_dqn.zero_grad()
#            if next_states_batch.grad_fn == None : next_states_batch.volatile = True
        loss.requires_grad=True
        loss.volatile = False
#        loss = loss/self.batch_size
        
#        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
        
        
    def CNN_learn(self,memory):
        batch = memory.sample(self.batch_size)
        [states, actions, rewards, next_states, dones] = zip(*batch)
        
        state_batch = Variable(torch.stack(states).squeeze(1))
        next_states_batch = Variable(torch.stack(next_states).squeeze(1))
        
        
        action_batch = Variable(LongTensor(actions))
        reward_batch = Variable(Tensor(rewards))
        
        state_action_values = self.main_dqn(state_batch).gather(1, action_batch.view(-1,1)).view(-1)
        next_states_batch.volatile = True
        next_state_values = self.target_dqn(next_states_batch).max(1)[0]
        for i in range(self.batch_size):
            if dones[i]:
                next_state_values.data[i]=0
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount) + reward_batch
        expected_state_action_values.volatile = False
        loss = F.mse_loss(state_action_values, expected_state_action_values)        
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
        
        
    def get_td_error(self,re,st,ac,st_):
        re = Variable(Tensor([re])).view(1,-1)
        st = Variable(Tensor([st])).view(1,-1)
        st_ = Variable(Tensor([st_])).view(1,-1)
        ac = Variable(LongTensor([ac])).view(1,-1)
        
        td_error = re + self.discount * self.target_dqn(st_).max(1)[0] - self.main_dqn(st).gather(1,ac)
        return abs(td_error.data[0][0])
    
        
        
    def PER_learn(self,memory):
        batch, batch_idx  = memory.sample(self.batch_size)
        
        [states, actions, rewards, next_states, dones] = zip(*batch)
        state_batch = Variable(Tensor(states))
        action_batch = Variable(LongTensor(actions))
        reward_batch = Variable(Tensor(rewards))
        next_states_batch = Variable(Tensor(next_states))
        
        state_action_values = self.main_dqn(state_batch).gather(1, action_batch.view(-1,1)).view(-1)
        next_states_batch.volatile = True
        next_state_values = self.target_dqn(next_states_batch).max(1)[0]
        for i in range(self.batch_size):
            if dones[i]:
                next_state_values.data[i]=0
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount) + reward_batch
        expected_state_action_values.volatile = False
        loss = F.mse_loss(state_action_values, expected_state_action_values)        
        
        td_error = expected_state_action_values - state_action_values
        for i in range(self.batch_size):
            val = abs(td_error[i].data[0])
            memory.update(batch_idx[i],val)
            
        
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
        
    def double_dqn_learn(self,memory):
        batch = memory.sample(self.batch_size)
        [states, actions, rewards, next_states, dones] = zip(*batch)
        state_batch = Variable(Tensor(states))
        action_batch = Variable(LongTensor(actions))
        reward_batch = Variable(Tensor(rewards))
        next_states_batch = Variable(Tensor(next_states))
        
        state_action_values = self.main_dqn(state_batch).gather(1, action_batch.view(-1,1)).view(-1)
        next_states_batch.volatile = True
        argmax_Q = self.main_dqn(next_states_batch).max(1)[1].unsqueeze(1)
        next_state_values = self.target_dqn(next_states_batch).gather(1,argmax_Q).squeeze(1)
        for i in range(self.batch_size):
            if dones[i]:
                next_state_values.data[i]=0
        expected_state_action_values = (next_state_values *  self.discount) + reward_batch
        expected_state_action_values.volatile = False
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
    
    def reset_noise(self):
        self.main_dqn.reset_noise()
        
        
        
    
    def _get_categorical(self, next_states, rewards, mask):
        batch_sz = next_states.size(0)
        gamma = self.discount

        # Compute probabilities p(x, a)
        probs = self.target_dqn(next_states).data
        qs = torch.mul(probs, self.support.expand_as(probs))
        argmax_a = qs.sum(2).max(1)[1].unsqueeze(1).unsqueeze(1)
        action_mask = argmax_a.expand(batch_sz, 1, self.atoms)
        qa_probs = probs.gather(1, action_mask).squeeze()

        # Mask gamma and reshape it torgether with rewards to fit p(x,a).
        rewards = rewards.expand_as(qa_probs)
        gamma = (mask.float() * gamma).expand_as(qa_probs)

        # Compute projection of the application of the Bellman operator.
#        bellman_op = rewards + gamma * self.support.unsqueeze(0).expand_as(rewards)
        bellman_op = rewards + gamma * self.support.unsqueeze(0).expand_as(rewards)
        bellman_op = torch.clamp(bellman_op, self.v_min, self.v_max)

        # Compute categorical indices for distributing the probability
        m = self.m.fill_(0)
        b = (bellman_op - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Distribute probability
        """
        for i in range(batch_sz):
            for j in range(self.atoms):
                uidx = u[i][j]
                lidx = l[i][j]
                m[i][lidx] = m[i][lidx] + qa_probs[i][j] * (uidx - b[i][j])
                m[i][uidx] = m[i][uidx] + qa_probs[i][j] * (b[i][j] - lidx)
        for i in range(batch_sz):
            m[i].index_add_(0, l[i], qa_probs[i] * (u[i].float() - b[i]))
            m[i].index_add_(0, u[i], qa_probs[i] * (b[i] - l[i].float()))

        """
        # Optimized by https://github.com/tudor-berariu
        offset = torch.linspace(0, ((batch_sz - 1) * self.atoms), batch_sz)\
            .type(torch.LongTensor)\
            .unsqueeze(1).expand(batch_sz, self.atoms)

        m.view(-1).index_add_(0, (l + offset).view(-1),
                              (qa_probs * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1),
                              (qa_probs * (b - l.float())).view(-1))
        return Variable(m)



    def noisy_distributional_dddqn_train(self,memory):
        batch = memory.sample(self.batch_size)
        [states, actions, rewards, next_states, dones] = zip(*batch)
        
        dones = list(dones)
        mask = [not dones[i] for i in range(len(dones))]
        mask = LongTensor(mask).view(-1,1)
        rewards = Tensor(rewards).view(-1,1)
        
        
        
        
        states = Variable(Tensor(states))
        actions = Variable(LongTensor(actions))
        rewards = (Tensor(rewards))
        next_states = Variable(Tensor(next_states))
        
        
        # Compute probabilities of Q(s,a*)
        q_probs = self.main_dqn(states)
        actions = actions.view(self.batch_size, 1, 1)
        action_mask = actions.expand(self.batch_size, 1, self.atoms)
        qa_probs = q_probs.gather(1, action_mask).squeeze()

        # Compute distribution of Q(s_,a)
        target_qa_probs = self._get_categorical(next_states, rewards, mask)

        # Compute the cross-entropy of phi(TZ(x_,a)) || Z(x,a)
        qa_probs.data.clamp_(0.01, 0.99)  # Tudor's trick for avoiding nans
        loss = - torch.sum(target_qa_probs * torch.log(qa_probs))
        
        
        # Accumulate gradients
        self.main_dqn.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.main_dqn.parameters(), self.max_gradient_norm)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
        
        