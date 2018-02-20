# Deep Reinforcement Learning Toolkit
* This toolkit tested on Python 2.7, Python 3.5
* I recommend you to know about deep reinforcement learning before use it.
* This toolkit doesn't provide the right agent for all problem, but provide the guideline.
* You can trace reward sum by Env_wrapper().
  
  
  
  
## Required package
* keras(recommended backend: tensorflow)
* gym
* matplotlib

## Classes
### Agent list(drlt.agents)
* Deep SARSA : deepSARSA.py - DeepSARSA_Agent() in /agents/deepSARSA.py
* DQN(Deep Q Network) : DQN_Agent() in /agents/dqn.py
* Monte-Carlo Policy Gradient(REINFORCE Algorithm) : REINFORCE_Agent() in /agents/mc_pg.py
* A2C(Advantage Actor-Critic) : A2C_Agent() in /agents/a2c.py
* A3C(Asynchronous Advantage Actor-Critic) : A3C_Agent() in /agents/a3c.py

### For trace(drlt.trace)
* Env_wapper() in /trace/env_wrapper.py

