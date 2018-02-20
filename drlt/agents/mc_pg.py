import numpy as np
import keras.backend as t
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Monte-Carlo Policy Gradient(REINFORCE Algorithm) Agent.
class REINFORCE_Agent(object):
	# Initializer
	#
	# state_size: Network input size.
	# action_size: Network output size.
	# gamma: (optional) Discount factor. Default 0.99
	# optimizer: (optional) Optimizer. Default Adam(lr=0.001).
	# resume: (optional) Load model from this path. If None, do not load. Default None.
	def __init__(self, state_size, action_size, gamma=0.99, optimizer=Adam(lr=0.001), resume=None):
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = gamma

		# model prepare
		self.model = self.build_model()
		self.update = self.build_update(optimizer)

		# for train()
		self.states = []
		self.actions = []
		self.rewards = []

		# resume
		if resume != None: self.load(resume)

	# Load the model
	#
	# path: Load from this path.
	def load(self,path):
		self.model.load_weights(path)

	# Save the model
	#
	# path: saved to this path.
	def save(self,path):
		self.model.save_weights(path)

	# Build model
	# If you want to change this model, override this method.
	#
	# Return - Policy net.
	def build_model(self):
		model = Sequential()
		model.add(Dense(50,input_dim=self.state_size,kernel_initializer='he_uniform',activation='relu'))
		model.add(Dense(50,kernel_initializer='he_uniform',activation='relu'))
		model.add(Dense(self.action_size,kernel_initializer='he_uniform',activation='softmax'))
		return model

	# Build model update
	#
	# optimizer: Optimizer.
	#
	# Return - Update op.
	def build_update(self,optimizer):
		action = t.placeholder(shape=[None,self.action_size])
		discounted_rewards = t.placeholder(shape=[None,])

		action_prob = t.sum(self.model.output * action,axis=1)
		cross_entropy = t.log(action_prob) * discounted_rewards
		loss = -t.sum(cross_entropy)

		updates = optimizer.get_updates(self.model.trainable_weights,[],loss)
		return t.function([self.model.input,action,discounted_rewards],[],updates=updates)

	# Return discounted rewards
	#
	# Return - Discounted rewards list.
	def get_discounted_rewards(self):
		discounted_rewards = np.zeros_like(self.rewards)
		ret = 0
		for t in reversed(range(len(self.rewards))):#4 ~ 0
			ret = self.rewards[t] + (self.gamma * ret)
			discounted_rewards[t] = ret
		return discounted_rewards

	# Training
	def train(self):
		# regulization
		discounted_rewards = np.float32(self.get_discounted_rewards())
		discounted_rewards -= np.mean(discounted_rewards)
		discounted_rewards /= np.std(discounted_rewards)

		self.update([self.states,self.actions,discounted_rewards])

		self.states = []
		self.actions = []
		self.rewards = []

	# Select action
	#
	# state: Network input.
	#
	# Return - Selected action.
	def select_action(self,state):
		policy = self.model.predict(state)[0]
		return np.random.choice(self.action_size,1,p=policy)[0]

	# Append sample(s,a,r)
	#
	# state: Network input.
	# action: Executed action.
	# reward: Current reward.
	def append(self,state,action,reward):
		self.states.append(state[0])
		self.rewards.append(reward)

		act = np.zeros(self.action_size)
		act[action] = 1
		self.actions.append(act)
