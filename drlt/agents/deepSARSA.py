import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Deep SARSA Agent
class DeepSARSA_Agent(object):
	# Initializer
	#
	# state_size: Network input size.
	# action_size: Action space length.
	# gamma: (optional) Discount factor. Default 0.99
	# optimizer: (optional) Optimizer. Default Adam(lr=0.001).
	# epsilon: (optional) Tuple, (epsilon,decay_rate,epsilon_min). Default (1.0, 0.999, 0.1).
	# resume: (optional) Load model from this path. If None, do not load. Default None.
	def __init__(self, state_size, action_size, gamma=0.99, optimizer=Adam(lr=0.001), epsilon=(1.0,0.999,0.1), resume=None):
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
		if resume != None: self.load(resume)

	# Load the model
	#
	# path: Load from this path.
	def load(self,path):
		self.actor.load_weights(path)
		self.critic.load_weights(path)

	# Save the model
	#
	# path: saved to this path.
	def save(self,path):
		self.actor.save_weights(path)
		self.critic.save_weights(path)

	# Build_model
	# If you want to change this model, override this method.
	#
	# optimizer: Optimizer.
	#
	# Return - network
	def build_model(self, optimizer):
		model = Sequential()
		model.add(Dense(50, input_dim=self.state_size, kernel_initializer='he_uniform', activation='relu'))
		model.add(Dense(50, kernel_initializer='he_uniform', activation='relu'))
		model.add(Dense(self.action_size, kernel_initializer='he_uniform', activation='linear'))
		model.compile(loss='mse',optimizer=optimizer)
		return model

	# Select action
	#
	# state: Current state
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

	# Train
	#
	# state, action, reward, next_state, next_action, done: Samples
	def train(self, state, action, reward, next_state, next_action, done):
		target = self.model.predict(state)[0]

		if done:
			target[action] = reward
		else:
			target[action] = reward + self.gamma * self.model.predict(next_state)[0][next_action]

		target = np.reshape(target, [1,-1])

		self.model.fit(state,target, epochs=1, verbose=0)

