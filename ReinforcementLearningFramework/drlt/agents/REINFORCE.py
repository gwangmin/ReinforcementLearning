'''
This file implements REINFORCE Algorithm
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
        optimizer: (optional) Optimizer. Default Adam(lr=0.001).
        '''
        super().__init__(state_size, action_size, gamma, weights_path, (optimizer,'adam'))

        # model prepare
        self.model = self.build_policy_net()
        self.optimizer = self.build_policy_optimizer(self.model, optimizer)

        # for train()
        self.states = []
        self.actions = []
        self.rewards = []

    def get_discounted_rewards(self):
        '''
        Return discounted rewards

        Return - Discounted rewards list [G_1, G_2, G_3, ...]
        '''
        discounted_rewards = np.zeros_like(self.rewards)
        g = 0
        for t in reversed(range(len(self.rewards))):
            g = self.rewards[t] + (self.gamma * g)
            discounted_rewards[t] = g
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

        # update
        self.optimizer([self.states, actions_for_train, discounted_rewards])

        # flush
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


