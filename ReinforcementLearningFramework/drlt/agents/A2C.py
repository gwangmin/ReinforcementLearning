'''
This file implements A2C(Advantage Actor-Critic)
'''

import numpy as np
import keras.backend as t
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from .PolicyBased import PolicyBased


class A2C_Agent(PolicyBased):
    '''
    A2C(Advantage Actor-Critic Algorithm) Agent
        - PolicyBased(BaseAgent)
    '''
    def __init__(self, state_size, action_size, gamma=0.99, weights_path=None, optimizers=(Adam(lr=0.001),'adam')):
        '''
        Initializer

        state_size: Observation space size.
        action_size: Action space size.
        gamma: (optional) Discount factor. Default 0.99
        weights_path (optional) Load weights from this path. If None, do not load. Default None.
        optimizers: (optional) Tuple. Optimizers. 1st element is actor's optimizer, 2nd element is critic's optimizer. Default (Adam(lr=0.001),'adam').
        '''
        super().__init__(state_size, action_size, gamma, weights_path, optimizers)

        # model prepare
        self.actor = self.build_policy_net()
        self.actor_optimizer = self.build_policy_optimizer(self.actor, optimizers[0])
        self.critic = self.build_value_net(optimizers[1])

    def build_actor_update(self,optimizer):
        '''
        Build actor update

        optimizer: Optimizer.

        Return - actor update function
        '''
        action = t.placeholder(shape=[self.action_size])
        advantage = t.placeholder(shape=[None])

        prob = t.sum(action * self.actor.output,axis=1)
        j = t.log(prob) * advantage
        loss = -t.sum(j)

        updates = optimizer.get_updates(self.actor.trainable_weights,[],loss)
        return t.function([self.actor.input,action,advantage],[],updates=updates)

    def train(self,state,action,reward,next_state,done):
        '''
        Training

        state, action, reward, next_state, done: samples
        '''
        act = np.zeros([self.action_size])
        act[action] = 1

        value = self.critic.predict(state)[0][0]
        if done:
            target = reward
            advantage = reward - value
        else:
            next_value = self.critic.predict(next_state)[0][0]
            target = reward + self.gamma * next_value
            advantage = target - value

        self.actor_update([state,act,[advantage]])
        self.critic_update([state,[target]])


