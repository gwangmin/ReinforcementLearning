'''
This file defines Policy Based RL Agent's Base Class
'''

import numpy as np
import keras.backend as t
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from .BaseAgent import BaseAgent


class PolicyBased(BaseAgent):
    '''
    Policy Based RL Agent's Base Class
        - BaseAgent
    '''
    def __init__(self, state_size, action_size, gamma=0.99, weights_path=None, optimizers=(Adam(lr=0.001),'adam')):
        '''
        Initializer

        state_size: Observation space size.
        action_size: Action space size.
        gamma: (optional) Discount factor. Default 0.99
        weights_path (optional) Load weights from this path. If None, do not load. Default None.
        optimizers: (optional) Tuple. Optimizers. 1st element is policy net's optimizer, 2nd element is value net's optimizer. Default (Adam(lr=0.001),'adam').
        '''
        super().__init__(state_size, action_size, gamma, weights_path)

        # models
        self.policy_net = self.build_policy_net()
        self.policy_optimizer = self.build_policy_optimizer(self.policy_net, optimizers[0])
        self.value_net = self.build_value_net(optimizers[1])

    def build_policy_net(self):
        '''
        Build policy net
        If you want to change this model, override this method.
        Defualt Structure:
            approximate policy
            fc(50) - fc(50) - fc(action_size)
            loss func - undefined

        Return - Policy net.
        '''
        model = Sequential()
        model.add(Dense(50,input_dim=self.state_size,kernel_initializer='he_uniform',activation='relu'))
        model.add(Dense(50,kernel_initializer='he_uniform',activation='relu'))
        model.add(Dense(self.action_size,kernel_initializer='he_uniform',activation='softmax'))
        return model

    def build_policy_optimizer(self, net, optimizer):
        '''
        Build policy net optimizer

        net: Policy net.
        optimizer: Optimizer.

        Return - Update op.
        '''
        action = t.placeholder(shape=[None,self.action_size])
        discounted_rewards = t.placeholder(shape=[None,])

        prob = t.sum(net.output * action, axis=1)
        j = t.log(prob) * discounted_rewards
        loss = -t.sum(j)

        updates = optimizer.get_updates(net.trainable_weights,[],loss)
        return t.function([net.input, action, discounted_rewards],[],updates=updates)

    def build_value_net(self, optimizer):
        '''
        Build value net
        If you want to change this model, override this method.
        Defualt Structure:
            approximate value function
            fc(50) - fc(50) - fc(action_size)
            loss func - mse

        optimizer: Optimizer.

        Return - value net.
        '''
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(50, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(self.action_size, kernel_initializer='he_uniform', activation='linear'))
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def get_action(self, state):
        '''
        Return selected action

        state: current state

        Return - selected action
        '''
        if state.shape[0] != 1: state=np.reshape(state, [1,-1])
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size,1,p=policy)[0]


