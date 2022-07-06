import numpy as np

from utils import *
from riverEnv import env
from MC.monteCarlo import MonteCarlo

algo_MC = MonteCarlo()
n_iterations = 10

print("\nFinding optimal policy...")
optimal_policy, action_values = algo_MC.find_optimal_policy(env = env,
                                                            gamma=.98,
                                                            n_iterations=n_iterations,
                                                            evaluation_episodes=400,
                                                            exploration_method='exploring_starts',
                                                            epsilon=.1,
                                                            visit_method="first_visit",
                                                            averaging_method="moving",
                                                            alpha=.1,
                                                            horizon=40,
                                                            return_action_values=True,
                                                            initial_action_values="optimistic",
                                                            typical_value=-10,
                                                            verbose=1,
                                                            done_states={0}
                                                            )
print("Optimal policy:", optimal_policy.probs)
print("Final action values:", action_values)

# print("\nPolicy during the learning:")
# policies_and_actions = algo_IP.find_optimal_policy_yielding( transition_probability, 
#                                             reward_probability, 
#                                             gamma=.98,
#                                             n_iterations=5,
#                                             return_action_values=True,
#                                             )
# for elem in policies_and_actions:
#     print(elem)
