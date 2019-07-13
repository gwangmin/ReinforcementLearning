'''
This file defines Environment Wrapper follows gym's interface
'''

import matplotlib.pylab as plt
import gym


class Env_wrapper(object):
    '''
    (Gym) Environment wrapper
    Provide equivalent interface and trace reward feature
    '''
    def __init__(self, env, render=False):
        '''
        env: Gym env name or env obj.
        render: (optional) If true, render every step. Default False.
        '''
        if isinstance(env,str): self.env = gym.make(env)
        else: self.env = env
        
        # render?
        self.render = render
        
        # to provide equivalent interface
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # for graph
        self.x_episodes = []
        self.y_rewards = []
        self.reward_sum = 0
        
        self.current_episode = 0
    
    def reset(self):
        '''
        This method is equivalent to env.reset()
        '''
        self.current_episode += 1
        self.reward_sum = 0
        observation = self.env.reset()
        if self.render: self.env.render()
        return observation
    
    def step(self, action):
        '''
        This method is equivalent to env.step()
        '''
        next_state,reward,done,info = self.env.step(action)
        if self.render: self.env.render()
        self.reward_sum += reward
        # for graph
        if done:
            self.x_episodes.append(self.current_episode)
            self.y_rewards.append(self.reward_sum)
            print('Episode: '+str(self.current_episode)+' with reward sum: '+str(self.y_rewards[-1])+' finished!')
        return next_state,reward,done,info
    
    def close(self, graph_path=None):
        '''
        Print finish message and show graph
        
        graph_path: (optional) Graph will be saved in this path. If None, no save. Default None
        '''
        # finish msg
        print('Training finished!')
        # graph
        plt.plot(self.x_episodes,self.y_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        # if save?
        if graph_path != None:
            plt.savefig(graph_path)
        plt.show()


