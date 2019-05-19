import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DeepSARSA_Agent(object):
	'''Deep SARSA Agent'''
	def __init__(self, state_size, action_size, gamma=0.99, optimizer=Adam(lr=0.001), epsilon=(1.0,0.999,0.1), resume_path=None):
		'''
		Initializer

		state_size: Observation space size.
		action_size: Action space size.
		gamma: (optional) Discount factor. Default 0.99
		optimizer: (optional) Optimizer. Default Adam(lr=0.001).
		epsilon: (optional) Tuple, (epsilon,decay_rate,epsilon_min). Default (1.0, 0.999, 0.1).
		resume_path: (optional) Load model from this path. If None, do not load. Default None.
		'''
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = gamma

		# model
		self.model = self.build_model(optimizer)

		# for epsilon greedy policy
		self.epsilon = epsilon[0]
		self.decay_rate = epsilon[1]
		self.epsilon_min = epsilon[2]

		# resume
		if resume_path != None: self.load(resume_path)

	def load(self,path):
		'''
		Load the model

		path: Load from this path.
		'''
		self.model.load_weights(path)

	def save(self,path):
		'''
		Save the model

		path: saved to this path.
		'''
		self.model.save_weights(path)

	def build_model(self, optimizer):
		'''
		Build_model
		If you want to change this model, override this method.

		optimizer: Optimizer.

		Return - network
		'''
		model = Sequential()
		model.add(Dense(50, input_dim=self.state_size, kernel_initializer='he_uniform', activation='relu'))
		model.add(Dense(50, kernel_initializer='he_uniform', activation='relu'))
		model.add(Dense(self.action_size, kernel_initializer='he_uniform', activation='linear'))
		model.compile(loss='mse',optimizer=optimizer)
		return model

	def select_action(self, state):
		'''
		Select action

		state: Current state

		Return - selected action
		'''
		# epsilon decay
		if self.epsilon > self.epsilon_min:
			tmp = self.epsilon * self.decay_rate
			if tmp >= self.epsilon_min: self.epsilon = tmp

		# select action
		if np.random.rand() <= self.epsilon:
			return np.random.choice(self.action_size,1)[0]
        else:
			q = self.model.predict(state)[0]
			return np.argmax(q)

	def train(self, state, action, reward, next_state, next_action, done):
		'''
		Train

		state, action, reward, next_state, next_action, done: Samples
		'''
		target = self.model.predict(state)[0]

		if done:
			target[action] = reward
		else:
			target[action] = reward + self.gamma * self.model.predict(next_state)[0][next_action]

		target = np.reshape(target, [1,-1])

		self.model.fit(state,target, epochs=1, verbose=0)
