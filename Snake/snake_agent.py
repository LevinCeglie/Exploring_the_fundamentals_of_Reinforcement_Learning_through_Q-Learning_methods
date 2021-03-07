##############################################################
# This code contains the Python object for the DQL agent.
# the agent is made up of a DQN and the implemented algorithm
##############################################################

#partly inspired by:
#https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_torch_2020.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import time
import os


class DQN(nn.Module):
    def __init__(self, obs_space, act_space, nr_hiddenlayers, nr_neurons):
        super(DQN, self).__init__()
        #modular deep Q-network
        #amount and size of linear hidden layers are freely adjustable
        self.nr_hiddenlayers = nr_hiddenlayers
        nr_neurons.insert(0,obs_space)
        nr_neurons.append(act_space)
        self.nr_neurons = nr_neurons

        temp = []
        for i in range(self.nr_hiddenlayers+1):
            temp.append(nn.Linear(self.nr_neurons[i],self.nr_neurons[i+1]))

        self.layers = nn.ModuleList(temp)


    def forward(self, state):
        #forward pass thorugh the network
        x = state
        for i in range(self.nr_hiddenlayers):
            x = F.relu(self.layers[i](x))

        return self.layers[-1](x)


class Agent:
    def __init__(self, gamma, lr, epsilon, epsilon_decay, epsilon_min, 
                obs_space, act_space, device, updates_per_step, 
                lr_decay = False, lr_milestones = None, lr_gamma = None, 
                nr_hiddenlayers=1, nr_neurons=[128], max_mem_length=1000):
        #deep q-learning agent
        #initializing variables and constants
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.device = device
        self.act_space = act_space
        self.obs_space = obs_space
        self.updates_per_step = updates_per_step
        self.lr_decay = lr_decay
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma
        self.max_mem_length = max_mem_length

        #initializing agent's memory
        self.memory = deque(maxlen=self.max_mem_length)

        #initializing agent's deep-Q-network
        self.dqn = DQN(self.obs_space, self.act_space, 
                        nr_hiddenlayers, nr_neurons).to(self.device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        #initializing learning rate scheduler, if lr decay is wanted
        if self.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                        self.optimizer, 
                                        milestones=self.lr_milestones, 
                                        gamma=self.lr_gamma)
        
    
    def savedata(self, state, action, reward, 
                                new_state, gameovertemp):
        #used to append a transition to the agent's memory
        self.memory.append((state, action, 
                            reward, new_state, gameovertemp))

    def act(self, state):
        #choosing action with epsilon-greedy strategy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.act_space)
        else:
            qvalues = self.dqn.forward(state)
            return torch.argmax(qvalues).item()
            
    def act_greedy(self, state):
        #choosing action completely greedy
        qvalues = self.dqn.forward(state)
        return torch.argmax(qvalues).item()

    def train(self):
        #sampling a specific number of datapoints from memory
        trainsample = random.sample(self.memory, self.updates_per_step)

        #iterates through every datapoint (transition) and 
        # performs a weight update on the Q-network
        for state, action, reward, new_state, done in trainsample:
            if done:
                target = reward
            else:
                target = reward + (self.gamma 
                        * torch.max(self.dqn.forward(new_state)).item())

            prediction = self.dqn.forward(state)[action].to(self.device)
            target = torch.tensor(target).to(self.device)
            self.optimizer.zero_grad()
            loss = F.mse_loss(target, prediction)
            loss.backward()
            self.optimizer.step()
            if self.lr_decay:
                self.scheduler.step()

        #adjustment of the exploration rate epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def get_lr(self):
        #for development used to check current learning rate
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    

