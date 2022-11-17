import numpy as np

from src.utils import *
from environnements.oceanEnv import transition_probability_ocean, reward_probability_ocean
from DP.dynamicProgramming import PolicyIteration

algo_IP = PolicyIteration()

print("\nFinding optimal policy...")
ideal_policy, action_values = algo_IP.find_optimal_policy(transition_probability_ocean, 
                                            reward_probability_ocean, 
                                            gamma=.98,
                                            n_iterations=5,
                                            verbose=1,
                                            return_action_values=True,
                                            )
print("Optimal policy:", ideal_policy.probs)
print("Final action values:", action_values)

print("\nPolicy during the learning:")
src.policies_and_actions = algo_IP.find_optimal_policy_yielding( transition_probability_ocean, 
                                            reward_probability_ocean, 
                                            gamma=.98,
                                            n_iterations=5,
                                            return_action_values=True,
                                            )
for elem in src.policies_and_actions:
    print(elem)
