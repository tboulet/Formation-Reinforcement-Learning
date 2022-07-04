import numpy as np

from utils import *
from riverEnv import transition_probability, reward_probability
from DP.dynamicProgramming import PolicyIteration

algo_IP = PolicyIteration()

print("\nFinding optimal policy...")
ideal_policy, state_values = algo_IP.find_optimal_policy(transition_probability, 
                                            reward_probability, 
                                            gamma=.98,
                                            n_iterations=5,
                                            verbose=1,
                                            return_state_values=True,
                                            )
print("Optimal policy:", ideal_policy.probs)

print("\nPolicy during the learning:")
res = algo_IP.find_optimal_policy_yielding( transition_probability, 
                                            reward_probability, 
                                            gamma=.98,
                                            n_iterations=5,
                                            return_state_values=True,
                                            return_state_actions_values=True,
                                            )
for policy, state_values, q_values in res:
    print("Policy:", policy.probs)
    print("State values:", state_values)
    print("Q values:", q_values)