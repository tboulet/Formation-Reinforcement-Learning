import numpy as np

from utils import *
from environnements.oceanEnv import OceanEnv
from MC.monteCarlo import MonteCarlo

algo_MC = MonteCarlo()
n_iterations = 10

print("\nFinding optimal policy...")
optimal_policy, action_values = algo_MC.find_optimal_policy(env = OceanEnv(),
                                                            gamma=.98,
                                                            n_iterations=n_iterations,
                                                            evaluation_episodes=100,
                                                            exploration_method='epsilon_greedy',
                                                            epsilon=.1,
                                                            visit_method="first_visit",
                                                            averaging_method="moving",
                                                            alpha=.1,
                                                            timelimit=40,
                                                            return_action_values=True,
                                                            initial_action_values="random",
                                                            typical_value=-10,
                                                            is_state_done=lambda state: state == 0,
                                                            verbose=1,     
                                                            )
print("Optimal policy's probs:", optimal_policy.probs)
print("Final action values:", action_values)

print("\nActions and action values during the learning:")
for elem in algo_MC.find_optimal_policy_yielding(   env = OceanEnv(),
                                                    gamma=.98,
                                                    n_iterations=2,
                                                    evaluation_episodes=3,
                                                    exploration_method='epsilon_greedy',
                                                    epsilon=.1,
                                                    visit_method="first_visit",
                                                    averaging_method="moving",
                                                    alpha=.1,
                                                    timelimit=40,
                                                    initial_action_values="optimistic",
                                                    typical_value=-10,
                                                    is_state_done=lambda state: state == 0,
                                                    yield_frequency="episode",
                                                    ):
    print(elem)
