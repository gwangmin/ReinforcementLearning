Deep Reinforcement Learning Toolkit v1.0.5
===========================================
* This toolkit provides Deep Reinforcement Learning agents.
* This toolkit tested on Python 2.7, Python 3.6
* I recommend you to know about deep reinforcement learning before use it.
* This toolkit doesn't provide the right agent for all problem, but provide the guideline.
* You can trace reward sum by Env_wrapper().
  
  
  
  
  
  
## Required packages
* tensorflow
* keras
* gym
* numpy
* matplotlib

## Contents
### Agent list(drlt.agents)
* Deep SARSA : DeepSARSA_Agent() in drlt/agents/DeepSARSA.py
* DQN(Deep Q Network) : DQN_Agent() in drlt/agents/DQN.py
* Monte-Carlo Policy Gradient(REINFORCE Algorithm) : REINFORCE_Agent() in drlt/agents/REINFORCE.py
* A2C(Advantage Actor-Critic) : A2C_Agent() in drlt/agents/A2C.py

### For trace(drlt.trace)
* Env_wapper() in drlt/trace/Env_wrapper.py


[한국어 간단 설명](https://gwangmin.github.io/intro/2018/05/20/drlt.html)
