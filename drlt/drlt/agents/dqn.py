import numpy as np
from collections import deque
from random import sample
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

# DQN(Deep Q Network) Agent
# agent.train() raise exception. For more detailed information, please show the train() method.
class DQN_Agent(object):
	# Initializer
	#
	# state_size: Network input size.
	# action_size: Network output length.
	# gamma: (optional) Discount factor. Default 0.99
	# optimizer: (optional) Optimizer. Default Adam(lr=0.001)
	# epsilon: (optional) Tuple, (epsilon,decay_rate,epsilon_min). Default (1.0, 0.999, 0.1).
	# replay_size: (optional) Max length for replay memory. Default 2000.
	# batch_size: (optional) Batch size for one training. Default 64.
	# resume: (optional) Load model from this path. If None, do not load. Default None.
	def __init__(self, state_size, action_size, gamma=0.99, optimizer=Adam(lr=0.001), epsilon=(1.0,0.999,0.1), replay_size=2000, batch_size=64, resume=None):
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = gamma
		self.batch_size = batch_size

		# for epsilon greedy policy
		self.epsilon = epsilon[0]
		self.decay_rate = epsilon[1]
		self.epsilon_min = epsilon[2]

		# model
		self.model = self.build_model(optimizer)
		self.target_model = self.build_model(optimizer)
		self.sync_networks()

		# replay
		self.memory = deque(maxlen=replay_size)

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

	# Sync networks
	def sync_networks(self):
		self.target_model.set_weights(self.model.get_weights())

	# Build model.
	# If you want to change this model, override this method.
	#
	# optimizer: Optimizer.
	#
	# Return - Q network
	def build_model(self, optimizer):
		model = Sequential()
		model.add(Dense(50, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
		model.compile(loss='mse',optimizer=optimizer)
		return model

	# Select action
	#
	# state: Current state.
	#
	# Return - selected action
	def select_action(self, state):
		# epsilon decay
		if self.epsilon >= self.epsilon_min:
			tmp = self.epsilon * self.decay_rate
			if tmp >= self.epsilon_min: self.epsilon = tmp

		# select action
		if np.random.rand() <= self.epsilon:
			return np.random.choice(self.action_size,1)[0]
		else:
			q = self.model.predict(state)[0]
			return np.argmax(q)

	# Append sample to replay memory
	#
	# state, action, reward, next_state, done: Samples to appended.
	def append(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	# Experience replay
	# If replay memory length is too short, raise exception.
	def train(self):
		# length check
		if len(self.memory) < self.batch_size:
			raise Exception('Replay memory length is too short!')

		# prepare training
		batch = sample(self.memory, self.batch_size)

		x = np.empty(0).reshape(0,self.state_size)
		y = np.empty(0).reshape(0,self.action_size)

		# calc target
		for state, action, reward, next_state, done in batch:
			target = self.model.predict(state)
			if done: tmp = reward
			else: tmp = reward + self.gamma * np.amax(self.target_model.predict(next_state))
			target[0][action] = tmp

			x = np.vstack([x,state])
			y = np.vstack([y,target])

		# training
		self.model.fit(x,y, epochs=1, batch_size=self.batch_size, verbose=0)
