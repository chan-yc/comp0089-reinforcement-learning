# UCL COMP0089 Reinforcement Learning (2023/24)

This repository contains the courseworks I completed for my MSc module [COMP0089 Reinforcement Learning](https://www.ucl.ac.uk/module-catalogue/modules/reinforcement-learning-COMP0089).


## Tasks

1. **Multi-armed Bernoulli Bandit Problem**
   - Implemented several agents with the following algorithms:
     - UCB
     - Greedy
     - $\epsilon$-greedy
     - Policy gradient (REINFORCE)

2. **Markov Decision Process**
   - Implementd several RL algorithms for a MDP:
     - Tabular TD learning
     - Policy iteration
     - Value iteration
   - Analysed a MDP

3. **Actor-Critics**
   - Implemented a deep RL agent using `jax`.

4. **Off-Policy Learning**
   - Implemented several off-policy multi-step return estimates:
     - Full importance sampling
     - Per-decision importance sampling (PDIS)
     - PDIS with control variates
     - PDIS with control variates and adaptive bootstrapping
   - Analysed the convergence and variance properties of a proposed TD error


## Tools and Libraries
- Python
- NumPy
- Jax


## Python Environment

Requirement: `python=3.11`

    pip install -r requirements.txt