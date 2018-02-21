import gym
import numpy as np
import keras.backend as t
import threading
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

# A3C(Asynchronous Advantage Actor-Critic) Agent
# Use multi-threading
class A3C_Agent(object):
	# Initializer
	#
	# state_size: Network input size.
	# action_size: Policy length.
	# gamma: (optional) Discount factor. Default 0.99
	# actor_optimizer: (optional) Actor's optimizer. Default Adam(lr=0.001).
	# critic_optimizer: (optional) Critic's optimizer. Default Adam(lr=0.005).
	# threads_num: (optional) Worker number. Default 10.
	# resume: (optional) Load model from this path. If None, do not load. Default None.
	def __init__(self, state_size, action_size, gamma=0.99, actor_optimizer=Adam(lr=0.001), critic_optimizer=Adam(lr=0.005), threads_num=10, resume=None):
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = gamma
		self.threads_num = threads_num

		# model
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

	# Build global models
	# If you want to change these models, override this method.
	#
	# Return - (actor,critic)
	def build_models(self):
		input_ = Input(shape=[self.state_size])
		l1 = Dense(50, kernel_initializer='he_uniform', activation='relu')(input_)
		l2 = Dense(50, kernel_initializer='he_uniform', activation='relu')(l1)

		policy = Dense(self.action_size, kernel_initializer='he_uniform', activation='softmax')(l2)
		value = Dense(1, kernel_initializer='he_uniform', activation='linear')(l2)

		actor = Model(inputs=input_,outputs=policy)
		critic = Model(inputs=input_,outputs=value)

		actor._make_predict_function()
		critic._make_predict_function()

		return actor, critic

	# Build actor update func.
	#
	# optimizer: Optimizer
	#
	# Return - actor update func
	def build_actor_update(self, optimizer):
		action = t.placeholder(shape=[None,self.action_size])
		advantages = t.placeholder(shape=[None])

		policy = self.actor.output

		action_prob = t.sum(action * policy, axis=1)
		cross_entropy = t.log(action_prob + 1e-10) * advantages
		cross_entropy = -t.sum(cross_entropy)

		entropy = t.sum(policy * t.log(policy + 1e-10), axis=1)
		entropy = t.sum(entropy)

		loss = cross_entropy + 0.01 * entropy

		updates = optimizer.get_updates(self.actor.trainable_weights,[],loss)
		return t.function([self.actor.input, action, advantages],[],updates=updates)

	# Build critic update func.
	#
	# optimizer: Optimizer.
	#
	# Return - Critic update function.
	def build_critic_update(self, optimizer):
		discounted_prediction = t.placeholder(shape=[None])

		values = self.critic.output
		loss = t.mean(t.square(discounted_prediction - values))

		updates = optimizer.get_updates(self.critic.trainable_weights,[],loss)
		return t.function([self.critic.input,discounted_prediction],[],updates=updates)

	# Execute worker process and start training
	def train(self):
		for i in range(self.threads_num):
			Worker(self.gamma, (self.actor,self.critic,self.actor_update,self.critic_update)).start()



class Worker(threading.Thread):
	# Initializer
	#
	# gamma: Discount factor.
	# global_models: Tuple, (actor,critic,actor_update,critic_update). About global networks.
	def __init__(self, gamma, global_models):
		threading.Thread.__init__(self)

		self.state_size = int(global_models[0].input.shape[1])
		self.action_size = int(global_models[0].output.shape[1])
		self.gamma = gamma

		# global networks
		self.actor, self.critic = global_models[:2]
		self.actor_update = global_models[2]
		self.critic_update = global_models[3]

		# local networks
		self.local_actor, self.local_critic = self.build_local_models()
		self.sync_networks()

		# for samples
		self.states, self.actions, self.rewards = [], [], []

	# Build local models
	# If you want to change these models, override this method.
	#
	# Return - (actor,critic)
	def build_local_models(self):
		input_ = Input(shape=[self.state_size])
		l1 = Dense(50, kernel_initializer='he_uniform', activation='relu')(input_)
		l2 = Dense(50, kernel_initializer='he_uniform', activation='relu')(l1)

		policy = Dense(self.action_size, kernel_initializer='he_uniform', activation='softmax')(l2)
		value = Dense(1, kernel_initializer='he_uniform', activation='linear')(l2)

		actor = Model(inputs=input_,outputs=policy)
		critic = Model(inputs=input_,outputs=value)

		actor._make_predict_function()
		critic._make_predict_function()

		return actor, critic

	# Sync networks.
	def sync_networks(self):
		self.local_actor.set_weights(self.actor.get_weights())
		self.local_critic.set_weights(self.critic.get_weights())

	# Select action.
	#
	# state: Current state.
	#
	# Return - selected action.
	def select_action(self, state):
		policy = self.local_actor.predict(state)[0]
		return np.random.choice(self.action_size, 1, p=policy)[0]

	# Append samples
	#
	# state, action, reward: Samples to appended.
	def append(self, state, action, reward):
		self.states.append(state[0])
		act = np.zeros(self.action_size)
		act[action] = 1
		self.actions.append(act)
		self.rewards.append(reward)

	# Upload samples to global network and train global network.
	#
	# done: If episode ends?
	def upload(self, done):
		discounted_prediction = self.get_discounted_prediction(done)

		self.states = np.array(self.states)

		values = self.critic.predict(self.states).reshape(-1)
		advantages = discounted_prediction - values

		self.actor_update([self.states, self.actions,advantages])
		self.critic_update([self.states, discounted_prediction])

		self.states, self.actions, self.rewards = [], [], []

		self.sync_networks()

	# Calc discounted prediction
	#
	# done: If episode ends?
	#
	# Return - discounted prediction
	def get_discounted_prediction(self, done):
		discounted_prediction = np.zeros_like(self.rewards)
		tmp = 0

		if not done:
			tmp = self.critic.predict(np.array([self.states[-1]]))[0][0]

		for t in reversed(range(len(self.rewards))):
			tmp = self.rewards[t] + tmp * self.gamma
			discounted_prediction[t] = tmp
		return discounted_prediction

	# Entry point of this thread.
	def run(self):
		self.sync_networks()
		# state preprocessing
		def preprocess(observation):
			return np.reshape(observation,[1,4])

		env = gym.make('CartPole-v1')# max step: 500
		state_size = env.observation_space.shape[0]
		action_size = env.action_space.n

		episodes = 500
	#	env.seed(777)
	#	fix_seed(777)

		for e in range(episodes):
			done = False
			steps = 0
			state = preprocess(env.reset())

			while not done:
				action = self.select_action(state)
				next_state,reward,done,info = env.step(action)
				next_state = preprocess(next_state)
				self.append(state,action,reward)
				state = next_state
				steps += 1
				if steps%50==0: self.upload(done)
			print('Episode',e+1,'finished')
		print('Training finished!')
