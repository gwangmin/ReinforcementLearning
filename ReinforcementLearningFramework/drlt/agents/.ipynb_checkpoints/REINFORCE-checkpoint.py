'''
This file defines REINFORCE Algorithm
'''

import numpy as np
import keras.backend as t
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from .PolicyBased import PolicyBased


class REINFORCE_Agent(PolicyBased):
    '''
    Monte-Carlo Policy Gradient(REINFORCE Algorithm) Agent.
        - PolicyBased(BaseAgent)
    '''
    def __init__(self, state_size, action_size, gamma=0.99, weights_path=None, optimizer=Adam(lr=0.001)):
        '''
        Initializer
        
        state_size: Observation space size.
        action_size: Action space size.
        gamma: (optional) Discount factor. Default 0.99
        weights_path (optional) Load weights from this path. If None, do not load. Default None.
        optimizers: (optional) Optimizer. Default Adam(lr=0.001).
        '''
        super().__init__(state_size, action_size, gamma, weights_path, optimizers)
        
        # model prepare
        self.model = self.build_policy_net()
        self.optimizer = self.build_policy_optimizer(self.model, optimizer)
        
        # for train()
        self.states = []
        self.actions = []
        self.rewards = []
    
    def build_update(self,optimizer):
        '''
        Build policy net optimizer
        
        net: Policy net.
        optimizer: Optimizer.
        
        Return - Update op.
        '''
        action = t.placeholder(shape=[None,self.action_size])
        discounted_rewards = t.placeholder(shape=[None,])
        
        prob = t.sum(self.model.output * action,axis=1)
        j = t.log(prob) * discounted_rewards
        loss = -t.sum(j)
        
        updates = optimizer.get_updates(self.model.trainable_weights,[],loss)
        return t.function([self.model.input,action,discounted_rewards],[],updates=updates)
    
    def get_discounted_rewards(self):
        '''
        Return discounted rewards
        
        Return - Discounted rewards list.
        '''
        discounted_rewards = np.zeros_like(self.rewards)
        ret = 0
        for t in reversed(range(len(self.rewards))):
            ret = self.rewards[t] + (self.gamma * ret)
            discounted_rewards[t] = ret
        return discounted_rewards
    
    def train(self):
        '''
        Training
        '''
        # regulization
        discounted_rewards = np.float32(self.get_discounted_rewards())
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        
        actions_for_train = []
        for action in self.actions:
            act = np.zeros(self.action_size)
            act[action] = 1
            actions_for_train.append(act)
        
        self.optimizer([self.states, actions_for_train, discounted_rewards])
        
        self.states = []
        self.actions = []
        self.rewards = []
    
    def append(self, state, action, reward):
        '''
        Append sample(s,a,r)
        
        state: Current state.
        action: Selected action.
        reward: Current reward.
        '''
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


