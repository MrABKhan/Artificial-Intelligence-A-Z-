# AI for Self Driving Car Intro via Comments :D by Muhammad Hamza, M.A Abu Bakar and M.Faseeh!

# Importing the libraries inorder to use for ann, we used torch here as it's format is easy, nn is the module responsible for making neural networks
# thus importing that and optimizer and auto-gradient for backpropagation.!

import numpy as np #numpy import
#import nesscary python mods
import random
import os

#torch and Neural Network Module
import torch
import torch.nn as nn
#Neural Network layer module for making functional layers plus optimizer 
import torch.nn.functional as F
import torch.optim as optim

#
import torch.autograd as autograd
from torch.autograd import Variable

#So our first part is creating the architecture of the Neural Network, so we will make a class Network for that.

class Network(nn.Module):
            # This is the constructor fuction so-to-say in python for classes    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__() #initialized the NN
        self.input_size = input_size 
        self.nb_action = nb_action
        #two fully connected layers with 30 nodes, one takes input and other throws output...so to speak 
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state):
        # easy and simple forward passs, takes itself as an object and passes the the input aka state of the game through the 
    # fully connected layer 1 and 2.
        x = F.relu(self.fc1(state))
        # as you can see we get the predicted Q values as the output from the state!
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay as it drastically changes performance in this!!! without it we are almost like start a-new everytime.

class ReplayMemory(object):
    
    def __init__(self, capacity):
        # initialize the buffer for saving the experiences.
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        # simple buffer implementattion of saving any event to the buffer aka the memory
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        # important step, this returns random samples from the replay memory.
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning,Most importatnt step!

class Dqn():
    #initializer or aka constructor for this
    def __init__(self, input_size, nb_action, gamma):
        # we give it a gamma that it takes and it creates a ANN ( as we made previously ) and it also makes a 10000 size buffer memory( 10000 experiences aka saved frames)
        
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        # we used the Adam optimizer here since it's pretty good and Learning rate is set at a pretty standard setting or 0.001, havent messed around much witht that
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # last but not lease, settting action and reward.
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        # So this takes the state of the screen, makes it go through our model aka the ANN, then does a softmax on it.
        probs = F.softmax(self.model(Variable(state, volatile = True))) 
        # the action with the highest probability aka the Q value will be returned! voila we did it. for a single action that is.
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #This will now find the rewards for the batch of actions that we take and then decide on them whether they were good or not.
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        
        #This updates the Buffer memory!!!
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        
        # we will save the brain= aka last know weights of the ANN for the Q ftn as a file
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
