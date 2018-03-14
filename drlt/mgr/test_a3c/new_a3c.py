import threading
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import numpy as np
import gym
import keras.backend as t

class A3C:
	def __init__(self):
		self.state_size = 4
		self.action_size = 2
		self.gamma = .99
		self.threads_num = 1

		# model
		self.actor, self.critic = self.build_models()
		self.actor_update = self.build_actor_update()
		self.critic_update = self.build_critic_update()

		self.start()

	def build_models(self):
		input_ = Input(shape=[self.state_size])
		l1 = Dense(100, activation='relu')(input_)
		l2 = Dense(100, activation='relu')(l1)

		policy = Dense(self.action_size, activation='softmax')(l1)
		value = Dense(1, activation='linear')(l1)

		actor = Model(inputs=input_, outputs=policy)
		critic = Model(inputs=input_, outputs=value)

		actor._make_predict_function()
		critic._make_predict_function()

		return actor, critic

	def build_actor_update(self):
		action = t.placeholder(shape=[None, self.action_size])
		advantages = t.placeholder(shape=[None])

		policy = self.actor.output

		action_prob = t.sum(action * policy, axis=1)
		cross_entropy = t.log(action_prob + 1e-10) * advantages
		cross_entropy = -t.sum(cross_entropy)

		entropy = t.sum(policy * t.log(policy + 1e-10), axis=1)
		entropy = t.sum(entropy)

		loss = cross_entropy + .01 * entropy

		updates = Adam(lr=.01).get_updates(self.actor.trainable_weights, [], loss)
		return t.function([self.actor.input, action, advantages],[],updates=updates)

	def build_critic_update(self):
		discounted_prediction = t.placeholder(shape = [None])

		value = self.critic.output

		loss = t.mean(t.square(discounted_prediction - value))

		updates = Adam(lr=.01).get_updates(self.critic.trainable_weights, [], loss)
		return t.function([self.critic.input, discounted_prediction], [], updates=updates)

	def start(self):
		for i in range(self.threads_num):
			Worker((self.actor, self.critic, self.actor_update, self.critic_update), gamma).start()



class Worker(threading.Thread):
	def __init__(self, global_models, gamma):
		threading.Thread.__init__(self)

		self.actor, self.critic, self.actor_update, self.critic_update = global_models
		self.state_size = int(self.actor.input.shape[1])
		self.action_size = int(self.actor.output.shape[1])
		self.gamma = gamma

		self.local_actor, self.local_critic = self.build_local_models()
		self.sync_networks()#

		self.states, self.actions, self.rewards = [], [], []

	def sync_networks(self):
		self.local_actor.set_weights(self.actor.get_weights())
		self.local_critic.set_weights(self.critic.get_weights())

	def build_local_models(self):
		input_ = Input(shape=[self.state_size])
		l1 = Dense(100, activation='relu')(input_)
		l2 = Dense(100, activation='relu')(l1)

		policy = Dense(self.action_size, activation='softmax')(l1)
		value = Dense(1, activation='linear')(l1)

		actor = Model(inputs=input_, outputs=policy)
		critic = Model(inputs=input_, outputs=value)

		actor._make_predict_function()
		critic._make_predict_function()

		return actor, critic

	def run(self):

	def append(self, state, action, reward):
		self.states.append(state)
		act = np.zeros(self.action_size)
		act[action] = 1
		self.actions.append(act)
		self.rewards.append(reward)

	def select_action(self, state):
		policy = self.local_actor.predict(state)[0]
		return np.random.choice(self.action_size, 1, p=policy)[0]

	def get_discounted_prediction(self, done):
		discounted_prediction = np.zeros_like(self.rewards)
		tmp = 0

		if not done:#
			tmp = self.critic.predict(self.states[-1].reshape([1,-1]))[0][0]

		for t in reversed(range(len(self.rewards))):
			tmp = self.rewards[t] + self.gamma * tmp
			discounted_prediction[t] = tmp
		return discounted_prediction

	def upload(self, done):
		self.states = np.array(self.states)
		self.actions = np.array(self.actions)#

		discounted_prediction = self.get_discounted_prediction(done)

		values = self.critic.predict(self.states).reshape(-1)
		advantages = discounted_prediction - values
		
		self.actor_update([self.states, self.actions, advantages])
		self.critic_update([self.states, discounted_prediction])

		self.states, self.actions, self.rewards = [], [], []

		self.sync_networks()

