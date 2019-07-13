'''
This file defines Value Based RL Agent's Base Class
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from .BaseAgent import BaseAgent


class ValueBased(BaseAgent):
    '''
    Value Based RL Agent's Base Class
        - BaseAgent
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
        super().__init__(state_size, action_size, gamma, weights_path)
        
        # for epsilon greedy policy
        if isinstance(epsilon, tuple) == True:
            self.decay = True
            self.epsilon = epsilon[0]
            self.decay_rate = epsilon[1]
            self.epsilon_min = epsilon[2]
        else:
            self.decay = False
            self.epsilon = epsilon
        
        # model
        self.model = self.build_model(optimizer)
    
    def build_model(self, optimizer):
        '''
        Build_model
        If you want to change this model, override this method.
        Defualt Structure:
            approximate value function
            fc(50) - fc(50) - fc(action_size)
            loss func - mse
        
        optimizer: Optimizer.
        
        Return - network
        '''
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(50, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(self.action_size, kernel_initializer='he_uniform', activation='linear'))
        model.compile(loss='mse',optimizer=optimizer)
        return model
    
    def get_action(self, state, epsilon=None):
        '''
        Return selected action
        
        state: current state
        epsilon: float or None. If float, use this epsilon value(once). If None,
                    use value which set in initializer.
        
        Return - selected action
        '''
        # determine epsilon
        if epsilon == None:
            x = False
            if self.epsilon == None:
                epsilon = -1
            else:
                epsilon = self.epsilon
        else:
            x = True
        
        # select action
        if np.random.rand() <= epsilon:
            a = np.random.choice(self.action_size,1)[0]
        else:
            if state.shape[0] != 1: state = np.reshape(state, [1,-1])
            v = self.model.predict(state)[0]
            a = np.argmax(v)
        
        # decay epsilon. if needed
        if x == False and self.decay == True:
            if self.epsilon > self.epsilon_min:
                tmp = self.epsilon * self.decay_rate
                if tmp >= self.epsilon_min: self.epsilon = tmp
        
        # return
        return a


