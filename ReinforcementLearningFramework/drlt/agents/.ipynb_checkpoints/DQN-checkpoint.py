'''
This file defines DQN Agent
'''

from random import sample
from collections import deque
import numpy as np
from keras.models import Sequential
from .ValueBased import ValueBased


class DQN_Agent(ValueBased):
    '''
    DQN(Deep Q Network) Agent
    agent.train() raise exception. For more detailed information, please show the train() method.
        - ValueBased(BaseAgent)
    '''
    def __init__(self, state_size, action_size, gamma=0.99, weights_path=None, epsilon=(1.0,0.999,0.1), optimizer='adam', replay_size=1000000, batch_size=32):
        '''
        Initializer
        
        state_size: Observation space size.
        action_size: Action space size.
        gamma: (optional) Discount factor. Default 0.99
        weights_path (optional) Load weights from this path. If None, do not load. Default None.
        epsilon: (optional) Tuple(epsilon, decay_rate, epsilon_min) or float(constant epsilon value) or None. Default (1.0, 0.999, 0.1).
        optimizer: (optional) Optimizer. Default 'adam'.
        replay_size: (optional) Maximum length for replay memory. Default 1000000.
        batch_size: (optional) Batch size for one training. Default 32.
        '''
        super().__init__(state_size, action_size, gamma, None, epsilon, optimizer)# weights_path=None. Because of sync.
        
        self.batch_size = batch_size
        
        # model for seperated networks
        self.target_model = self.build_model(optimizer)
        self.sync_networks()
        
        # replay
        self.memory = deque(maxlen=replay_size)
        
        # resume for sync
        if weights_path != None:
            self.load(weights_path)
            self.sync_networks()
    
    def sync_networks(self):
        '''
        Sync networks
        '''
        self.target_model.set_weights(self.model.get_weights())
    
    def append(self, state, action, reward, next_state, done):
        '''
        Append sample to replay memory
        
        state, action, reward, next_state, done: Samples to appended.
        '''
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        '''
        Training use Experience Replay
        If replay memory length is too short, raise exception.
        '''
        # length check
        if len(self.memory) < self.batch_size:
            raise Exception('Replay memory length is too short!')
        
        # prepare training
        batch = sample(self.memory, self.batch_size)
        
        x = np.empty(0).reshape(0,self.state_size)
        y = np.empty(0).reshape(0,self.action_size)
        
        # calc target
        for state, action, reward, next_state, done in batch:
            target = self.model.predict(state.reshape(1,-1))
            if done: tmp = reward
            else: tmp = reward + self.gamma * np.amax(self.target_model.predict(next_state.reshape(1,-1)))
            target[0][action] = tmp
            
            x = np.vstack([x,state])
            y = np.vstack([y,target])
        
        # training
        self.model.fit(x,y, epochs=1, batch_size=self.batch_size, verbose=0)


