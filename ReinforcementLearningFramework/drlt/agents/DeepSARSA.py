'''
This file defines Deep SARSA Agent
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from .ValueBased import ValueBased


class DeepSARSA_Agent(ValueBased):
    '''
    Deep SARSA Agent
        - Value based(BaseAgent)
    '''
    def __init__(self, state_size, action_size, gamma=0.99, weights_path=None, epsilon=(1.0,0.999,0.1), optimizer='adam'):
        '''
        Initializer
        
        state_size: Observation space size.
        action_size: Action space size.
        gamma: (optional) Discount factor. Default 0.99
        weights_path (optional) Load weights from this path. If None, do not load. Default None.
        epsilon: (optional) Tuple(epsilon, decay_rate, epsilon_min) or float(constant epsilon value) or None. Default (1.0, 0.999, 0.1).
        optimizer: (optional) Optimizer. Default 'adam'.
        '''
        super().__init__(state_size, action_size, gamma, weights_path, epsilon, optimizer)
    
    def train(self, state, action, reward, next_state, next_action, done):
        '''
        Training
        
        state, action, reward, next_state, next_action, done: Samples
        '''
        if state.shape[0] != 1: state = np.reshape(state, [1,-1])
        if next_state.shape[0] != 1: next_state = np.reshape(next_state, [1,-1])
        
        target = self.model.predict(state)[0]
        
        if done:
            target[action] = reward
        else:
            target[action] = reward + self.gamma * self.model.predict(next_state)[0][next_action]
        
        target = np.reshape(target, [1,-1])
        
        self.model.fit(state, target, epochs=1, verbose=0)


