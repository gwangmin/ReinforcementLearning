import gym
import pylab as plt

# Gym environment wrapper
# Provide equivalent interface
class Env_wrapper(object):
	def __init__(self,env,render=False):
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

		self.current_episode = 1

	# This method is equivalence, env.reset()
	def reset(self):
		observation = self.env.reset()
		if self.render: self.env.render()
		return observation

	# This method is equivalence, env.step()
	def step(self,action):
		next_state,reward,done,info = self.env.step(action)
		if self.render: self.env.render()
		self.reward_sum += reward
		if done:
			# for graph
			self.x_episodes.append(self.current_episode)
			self.y_rewards.append(self.reward_sum)
			self.reward_sum = 0

			print('Episode: '+str(self.current_episode)+', Reward sum: '+str(self.y_rewards[-1]))
			self.current_episode += 1

		return next_state,reward,done,info

	# Print finish message and show graph
	def alert_finish(self,graph_path=None):
		print ('Training finished!')
		# graph prepare
		plt.plot(self.x_episodes,self.y_rewards)
		plt.xlabel('Episode')
		plt.ylabel('Reward')
		# if save?
		if graph_path != None:
			plt.savefig(graph_path)
		plt.show()

