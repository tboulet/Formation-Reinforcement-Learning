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
                                                            exploration_method='epsilon_greedy',
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
print("Optimal policy's probs:", optimal_policy.probs)
print("Final action values:", action_values)

print("\Actions and action values during the learning:")
for elem in algo_MC.find_optimal_policy_yielding(   env = env,
                                                    gamma=.98,
                                                    n_iterations=n_iterations,
                                                    evaluation_episodes=20,
                                                    exploration_method='epsilon_greedy',
                                                    epsilon=.1,
                                                    visit_method="first_visit",
                                                    averaging_method="moving",
                                                    alpha=.1,
                                                    horizon=40,
                                                    initial_action_values="optimistic",
                                                    typical_value=-10,
                                                    done_states={0},
                                                    ):
    print(elem)
