import numpy as np
import keras.backend as t
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

# A2C(Advantage Actor-Critic Algorithm) Agent
class A2C_Agent(object):
	# Initializer
	#
	# state_size: Network input size.
	# action_size: Policy length.
	# gamma: (optional) Discount factor. Default 0.99
	# actor_optimizer: (optional) Actor's optimizer. Default Adam(lr=0.001).
	# critic_optimizer: (optional) Critic's optimizer. Default Adam(lr=0.005).
	# resume: (optional) Load model from this path. If None, do not load. Default None.
	def __init__(self, state_size, action_size, gamma=0.99, actor_optimizer=Adam(lr=0.001), critic_optimizer=Adam(lr=0.005), resume=None):
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = gamma

		# model prepare
		self.actor, self.critic = self.build_models()

		self.actor_update = self.build_actor_update(actor_optimizer)
		self.critic_update = self.build_critic_update(critic_optimizer)

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

	# Build actor and critic
	# If you want to change this models, override this method.
	#
	# Return - (actor, critic)
	def build_models(self):
		input_ = Input(shape=[self.state_size])
		l1 = Dense(50, activation='relu', kernel_initializer='he_uniform')(input_)
		l2 = Dense(50, activation='relu', kernel_initializer='he_uniform')(l1)

		policy = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(l2)
		value = Dense(1, kernel_initializer='he_uniform', activation='linear')(l2)

		actor = Model(inputs=input_,outputs=policy)
		critic = Model(inputs=input_,outputs=value)

		actor._make_predict_function()
		critic._make_predict_function()

		return actor, critic

	# Select action using policy net.
	#
	# state: Network input.
	#
	# Return - Selected action
	def select_action(self,state):
		policy = self.actor.predict(state)[0]
		return np.random.choice(self.action_size,1,p=policy)[0]

	# Build actor update
	#
	# optimizer: Optimizer.
	#
	# Return - actor update function
	def build_actor_update(self,optimizer):
		action = t.placeholder(shape=[self.action_size])
		advantage = t.placeholder(shape=[None])

		action_prob = t.sum(action * self.actor.output,axis=1)
		cross_entropy = t.log(action_prob) * advantage
		loss = -t.sum(cross_entropy)

		updates = optimizer.get_updates(self.actor.trainable_weights,[],loss)
		return t.function([self.actor.input,action,advantage],[],updates=updates)

	# Build critic update
	#
	# optimizer: Optimizer.
	#
	# Return - Critic update function
	def build_critic_update(self,optimizer):
		target = t.placeholder(shape=[None])

		loss = t.mean(t.square(self.critic.output - target))

		updates = optimizer.get_updates(self.critic.trainable_weights,[],loss)
		return t.function([self.critic.input,target],[],updates=updates)

	# Training
	#
	# state: Network input.
	# action: Selected action.
	# reward: Current reward.
	# next_state: Next state, not current.
	# done: gym.step()'s return value.
	def train(self,state,action,reward,next_state,done):
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


