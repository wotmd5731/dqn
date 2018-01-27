# -*- coding: utf-8 -*-
"""
without  replay memory

TODO : apply  replay memory 


"""
from JAE_IMPORT import *

reward_save_name='reward_DQN_double_dueling_multistep'
#
#class ReplayMemory(object):
#    def __init__(self, capacity):
#        self.capacity = capacity
#        self.memory = []
#
#    def push(self, args):
#        if len(self.memory) >self.capacity:
#            #overflow mem cap pop the first element
#            self.memory.pop(0)
#        self.memory.append(args)
#
#    def sample(self, batch_size):
#        return random.sample(self.memory, batch_size)
#
#    def __len__(self):
#        return len(self.memory)
#
#

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.count = 0

    def push(self, args):
        self.count+=1
        if self.count == mem_interval:
            self.count = 0
            return 
            
        if len(self.memory) >self.capacity:
            #overflow mem cap pop the first element
            self.memory.pop(0)
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
#        self.fc1_a = nn.Linear(4,128)
        
        self.fc1 = nn.Linear(env_state_size,128)
        self.action_space = 2
#        self.fc2_v = nn.Linear(128,128)
#        self.fc2_a = nn.Linear(128,128)
        
#        self.fc3_v = nn.Linear(128,1)
#        self.fc3_a = nn.Linear(128,env_action_size)
        
        self.fc_h = nn.Linear(128, 128)
        self.fc_z_v = nn.Linear(128, 1)
        self.fc_z_a = nn.Linear(128, self.action_space )
        
        
    def forward(self, x):
#        x_v = F.relu(self.fc1_v(x))
#        x_a = F.relu(self.fc1_a(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc_h(x))
        
        
        v, a = self.fc_z_v(x), self.fc_z_a(x)  # Calculate value and advantage streams
        a_mean = torch.stack(a.chunk(self.action_space, 1), 1).mean(1)
        x = v.repeat(1, self.action_space) + a - a_mean.repeat(1, self.action_space)  # Combine streams
            
        
        
#        x_v = F.relu(self.fc2_v(x))
#        x_a = F.relu(self.fc2_a(x))
#        
#        x_v = F.relu(self.fc3_v(x_v)).expand(x.size(0), env_action_size)
#        x_a = F.relu(self.fc3_a(x_a))
##        print(x_v)
#        
#        x = x_v + x_a - x_a.mean(1).unsqueeze(1).expand(x.size(0), env_action_size)

        return x

# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)'

  ys = torch.Tensor(ys_population)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='transparent'), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color='transparent'), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)




#initilaize replay memory D to cap N


mem = ReplayMemory(mem_capacity)
main_dqn= DQN()
main_param=list(main_dqn.parameters())
if use_cuda:
    main_dqn.cuda()

target_dqn = DQN()
target_param=list(target_dqn.parameters())
#print("target update done ",main_param[0][0] , target_param[0][0])
if use_cuda:
    target_dqn.cuda()

target_dqn.load_state_dict(main_dqn.state_dict())
#target_param=list(target_dqn.parameters())
#print("target update done ",main_param[0][0] , target_param[0][0])





#optimizer = optim.RMSprop(main_dqn.parameters())
optimizer = optim.Adam(main_dqn.parameters(), lr=learning_rate)
#criterion = nn.MSELoss() 


while len(mem) < learning_start:
    state = env.reset()
    for t in range(500):
        action = random.randrange(0,2)
        next_state , reward , done, _ = env.step(action)
        mem.push([state, action, reward, next_state, done])
        state = next_state
        if done :
            break

print('learning start size  : ',len(mem))

for episode in range(10000):
    
    state = env.reset()
    t=0
    done = 0
    action_stack = []
    next_state_stack = []
    reward_stack = []
    done_stack = []
    
#    state = preprocess(state).unsqueeze(0)
    
    while t < env_max_count:
#        print(t)
        t_start = t    
        
        while not (done or t-t_start == t_max):
            if random.random() <= epsilon :
                action = random.randrange(0,env_action_size)
            else :
                action = main_dqn(Variable(FloatTensor(state.reshape(1,env_state_size)),volatile=True)).max(1)[1].data[0] #return max index call [1] 
            
            next_state , reward , done, _ = env.step(action)
            action_stack.append(action)
            next_state_stack.append(next_state)
            reward_stack.append(reward)
            done_stack.append(done)
            
            state = next_state
            t = t + 1
            
            
#        R = Variable(FloatTensor([0]))
#        R.requires_grad = True
#        R.volatile = False
        if done:
            R = Variable(FloatTensor([0]),volatile=True)
        else :
            R = target_dqn(Variable(FloatTensor(state.reshape(1,env_state_size)),volatile=True)).max(1)[0]
        
#        for i in range(t-1, t_start):
        loss = Variable(FloatTensor([0]))
        for i in range(t_start,t):
            r_i = Variable(Tensor([reward_stack.pop()]))
            s_i = Variable(Tensor(next_state_stack.pop().reshape(1,-1)))
            a_i = Variable(LongTensor([action_stack.pop()]))
#            s_i.volatile = False
            
            R = r_i + step_gamma*R
            loss = loss + ( R - main_dqn(s_i).gather(1, a_i.view(-1,1)).view(-1) )**2
        
        loss.requires_grad=True
        loss.volatile = False
        
        optimizer.zero_grad()
        loss.backward()
        if grad_clamp:
            for param in main_dqn.parameters():
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        
        if (target_update_count % target_update_interval) ==0 :
#            print("updata ",target_update_count)
            target_dqn.load_state_dict(main_dqn.state_dict())
    #            print("target update done ",main_param[0][0] , target_param[0][0])
        if done:
            break
    
    if episode % episode_interval == 0:
        global current_time
        prev_time = current_time
        current_time = time.time()
        T_rewards, T_Qs = [], []
        test_cnt = 10
        Ts.append(episode)
        total_reward = 0
        
        for i in range(test_cnt):
            total_sub_reward = 0
            test_state = env.reset()
            for j in count():
                if env_render:
                    env.render()
                test_action = main_dqn(Variable(FloatTensor(test_state.reshape(1,4)),volatile=True)).max(1)[1].data[0]
                test_state, test_reward, test_done, _ = env.step(test_action)
                total_reward += test_reward
                total_sub_reward += test_reward
                if test_done:
                    T_rewards.append(total_sub_reward)
                    break
        ave_reward = total_reward/test_cnt
        # Append to results
        Trewards.append(T_rewards)
#        Qs.append(T_Qs)
        
        # Plot
        _plot_line(Ts, Trewards, reward_save_name, path='results')
#        _plot_line(Ts, Qs, 'Q', path='results')
        
        # Save model weights
#        main_dqn.save('results')
        print('episode: ',episode,'Evaluation Average Reward:',ave_reward, 'delta time:',current_time-prev_time)
#            if ave_reward >= 300:
#                break
    
    
    
    
