import gym
import pylab as plt

class Env_wrapper(object):
	'''
	Gym environment wrapper
	Provide equivalent interface and trace reward feature
	'''
	def __init__(self,env,render=False):
		'''
		env: Gym env name or env obj.
		render: (optional) If true, render every step. Default False.
		'''
		if isinstance(env,str): self.env = gym.make(env)
		else: self.env = env

		# to provide equivalent interface
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space

		# render?
		self.render = render

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
		observation = self.env.reset()
		if self.render: self.env.render()
		return observation

	def step(self,action):
		'''
		This method is equivalent to env.step()
		'''
		next_state,reward,done,info = self.env.step(action)
		if self.render: self.env.render()
		self.reward_sum += reward
		if done:
			# for graph
			self.x_episodes.append(self.current_episode)
			self.y_rewards.append(self.reward_sum)
			self.reward_sum = 0

			print('Episode: '+str(self.current_episode)+' with reward sum: '+str(self.y_rewards[-1])+' finished!')

		return next_state,reward,done,info

	def alert_finish(self,graph_path=None):
		'''
		Print finish message and show graph

		graph_path: Graph will be saved in this path.
		'''
		# finish msg
		print ('Training finished!')
		# graph
		plt.plot(self.x_episodes,self.y_rewards)
		plt.xlabel('Episode')
		plt.ylabel('Reward')
		# if save?
		if graph_path != None:
			plt.savefig(graph_path)
		plt.show()
