import numpy as np
import keras.backend as t
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

class A2C_Agent(object):
	'''A2C(Advantage Actor-Critic Algorithm) Agent'''
	def __init__(self, state_size, action_size, gamma=0.99, actor_optimizer=Adam(lr=0.001), critic_optimizer=Adam(lr=0.005), resume_path=None):
		'''
		Initializer

		state_size: Network input size.
		action_size: Policy length.
		gamma: (optional) Discount factor. Default 0.99
		actor_optimizer: (optional) Actor's optimizer. Default Adam(lr=0.001).
		critic_optimizer: (optional) Critic's optimizer. Default Adam(lr=0.005).
		resume_path: (optional) Tuple, (actor path, critic path). Load model from this path. If None, do not load. Default None.
		'''
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = gamma

		# model prepare
		self.actor, self.critic = self.build_models()

		self.actor_update = self.build_actor_update(actor_optimizer)
		self.critic_update = self.build_critic_update(critic_optimizer)

		# resume
		if resume_path != None: self.load(resume_path)

	def load(self,path):
		'''
		Load the model

		path: Load from this path. (actor path, critic path)
		'''
		self.actor.load_weights(path[0])
		self.critic.load_weights(path[1])

	def save(self,path):
		'''
		Save the model

		path: saved to this path. (actor path, critic path)
		'''
		self.actor.save_weights(path[0])
		self.critic.save_weights(path[1])

	def build_models(self):
		'''
		Build actor and critic
		If you want to change this models, override this method.

		Return - (actor, critic)
		'''
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

	def select_action(self,state):
		'''
		Select action using policy net.

		state: Network input.

		Return - Selected action
		'''
		policy = self.actor.predict(state)[0]
		return np.random.choice(self.action_size,1,p=policy)[0]

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

	def build_critic_update(self,optimizer):
		'''
		Build critic update

		optimizer: Optimizer.

		Return - Critic update function
		'''
		target = t.placeholder(shape=[None])

		loss = t.mean(t.square(self.critic.output - target))

		updates = optimizer.get_updates(self.critic.trainable_weights,[],loss)
		return t.function([self.critic.input,target],[],updates=updates)

	def train(self,state,action,reward,next_state,done):
		'''
		Training

		state: Current state.
		action: Selected action.
		reward: Reward.
		next_state: Next state, not current.
		done: gym.step()'s return value.
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
