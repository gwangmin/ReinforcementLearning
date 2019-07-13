'''
This file defines all agent's base class.
'''

from keras.models import load_model, model_from_json


class BaseAgent(object):
    '''
    Base Agent.
    All RL Agent's Base Class.
    '''
    def __init__(self, state_size, action_size, gamma=0.99, weights_path=None):
        '''
        Base Agent's initializer
        
        state_size: Observation space size.
        action_size: Action space size.
        gamma: (optional) Discount factor. Default 0.99
        weights_path (optional) Load weights from this path. If None, do not load. Default None.
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # resume
        if weights_path != None: self.load_weights(weights_path)
    
    def save(self, path):
        '''
        Save the whole model(architecture + weights + optimizer state)
            from specified path
        
        path: Save from this path
        '''
        self.model.save(path)
    
    def load(self, path):
        '''
        Load the whole model(architecture + weights + optimizer state)
            from specified path
        
        path: Load from this path
        '''
        self.model = load_model(path)
    
    def save_json_architecture(self, path):
        '''
        Save the model's architecture from specified path
        
        path: Save from this path
        '''
        json = self.model.to_json()
        with open(path, 'w') as f:
            f.write(json)
    
    def load_json_architecture(self, path):
        '''
        Load the model's architecture from specified path
        
        path: Load from this path
        '''
        with open(path, 'r') as f:
            json = f.read()
        self.model = model_from_json(json)
    
    def save_weights(self,path):
        '''
        Save the model's weights from specified path
        
        path: Weights path.
        '''
        self.model.save_weights(path)
    
    def load_weights(self,path):
        '''
        Load the model's weights from specified path
        
        path: Weights path.
        '''
        self.model.load_weights(path)
    
    def get_action(self, state):
        '''
        Return selected action
        
        state: current state
        
        Return - selected action
        '''
        raise NotImplementedError()
    
    def train():
        '''
        Training
        '''
        raise NotImplementedError()


