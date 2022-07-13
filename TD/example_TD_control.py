import numpy as np

from utils import *
from riverEnv import env
from TD.TDLearning import SARSA

algo_SARSA = SARSA()

print("\nFinding optimal policy...")
optimal_policy, action_values = algo_SARSA.find_optimal_policy( env = env,
                                                                gamma=.98,
                                                                n_episodes = 200,
                                                                n_steps = float("inf"),
                                                                exploration_method='epsilon_greedy',
                                                                epsilon=.1,
                                                                alpha=.5,
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
for elem in algo_SARSA.find_optimal_policy_yielding(env = env,
                                                    gamma=.98,
                                                    n_episodes = 10,
                                                    n_steps = float("inf"),
                                                    exploration_method='epsilon_greedy',
                                                    epsilon=.1,
                                                    alpha=.5,
                                                    timelimit=40,
                                                    return_action_values=True,
                                                    initial_action_values="random",
                                                    typical_value=-10,
                                                    is_state_done=lambda state: state == 0,
                                                    yielding_frequency="episode",
                                                    ):
    print(elem)
