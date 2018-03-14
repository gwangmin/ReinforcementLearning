import sys
sys.path.append('../drlt/agents/')
sys.path.append('../drlt/trace/')

import a2c
import deepSARSA as ds
import numpy as np
import a3c
import dqn
import mc_pg
from env_wrapper import *

episodes = 800

def deepSARSA_test():
	# state preprocessing
	def preprocess(state):
		return np.reshape(state,[1,4])

	env = Env_wrapper('CartPole-v1')# max step: 500
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = ds.DeepSARSA_Agent(state_size,action_size)
#	env.seed(777)
#	fix_seed(777)

	for e in range(episodes):
		done = False
		score = 0
		state = preprocess(env.reset())

		while not done:
			action = agent.select_action(state)
			next_state,reward,done,info = env.step(action)
			next_state = preprocess(next_state)
			next_action = agent.select_action(next_state)
			agent.train(state,action,reward,next_state,next_action,done)
			state = next_state
	env.alert_finish()



def dqn_test():
	# state preprocessing
	def preprocess(state):
		return np.reshape(state,[1,4])

	env = Env_wrapper('CartPole-v1')# max step: 500
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = dqn.DQN_Agent(state_size,action_size)
#	env.seed(777)
#	fix_seed(777)

	for e in range(episodes):
		done = False
		score = 0
		state = preprocess(env.reset())

		while not done:
			action = agent.select_action(state)
			next_state,reward,done,info = env.step(action)
			next_state = preprocess(next_state)
			agent.append(state,action,reward,next_state,done)
			state = next_state
		if e%10==0 and e!=0:
			for _ in range(20):
				agent.train()
			agent.sync_networks()
	env.alert_finish()



def mc_pg_test():
	# state preprocessing
	def preprocess(state):
		return np.reshape(state,[1,4])

	env = Env_wrapper('CartPole-v1')# max step: 500
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = mc_pg.REINFORCE_Agent(state_size,action_size)
#	env.seed(777)
#	fix_seed(777)

	for e in range(episodes):
		done = False
		score = 0
		state = preprocess(env.reset())

		while not done:
			action = agent.select_action(state)
			next_state,reward,done,info = env.step(action)
			next_state = preprocess(next_state)
			agent.append(state,action,reward)
			state = next_state
		agent.train()
	env.alert_finish()



def a2c_test():
	# state preprocessing
	def preprocess(state):
		return np.reshape(state,[1,4])

	env = Env_wrapper('CartPole-v1')# max step: 500
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = a2c.A2C_Agent(state_size,action_size)
#	env.seed(777)
#	fix_seed(777)

	for e in range(episodes):
		done = False
		score = 0
		state = preprocess(env.reset())

		while not done:
			action = agent.select_action(state)
			next_state,reward,done,info = env.step(action)
			next_state = preprocess(next_state)
			agent.train(state,action,reward,next_state,done)
			state = next_state
	env.alert_finish()



def a3c_test():
	env = Env_wrapper('CartPole-v1')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	agent = a3c.A3C_Agent(state_size, action_size)
	agent.train()



if __name__ == '__main__':
	a2c_test()
