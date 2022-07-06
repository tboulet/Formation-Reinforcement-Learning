import numpy as np

from utils import *
from riverEnv import RiverEnv
from MC.monteCarlo import MonteCarlo
from policies import DiscretePolicyForDiscreteState

policy_swim_randomly = DiscretePolicyForDiscreteState(probs = np.array([[0.8, 0.2] for _ in range(11)]))

algo_MC = MonteCarlo()

print("\nComputing state values for the policy policy_swim_randomly...")
estimated_state_values = algo_MC.find_state_values( policy = policy_swim_randomly,
                                                    env = RiverEnv(),
                                                    n_episodes = 100,
                                                    gamma=0.98)
print("Estimated state values :", estimated_state_values)

print("\nEstimated state values during the learning:")
estimated_state_values_during_training = algo_MC.find_state_values_yielding(   policy = policy_swim_randomly,
                                                                                env = RiverEnv(),
                                                                                n_episodes = 1,
                                                                                gamma = 0.98)
for estimated_state_values in estimated_state_values_during_training:
    print(estimated_state_values)

print("\nComputing action values for the policy policy_swim_randomly...")
estimated_action_values = algo_MC.find_action_values(   policy = policy_swim_randomly,
                                                        env = RiverEnv(),
                                                        n_episodes=100,
                                                        gamma=0.98,
                                                        visit_method="first_visit",
                                                        averaging_method="moving",
                                                        alpha=0.05,
                                                        horizon=40,
                                                        initial_action_values="random",
                                                        typical_value=-10,
                                                        # exploring_starts=True,
                                                        # done_states = {0},
                                                            )
print("Estimated action values :", estimated_action_values)

print("\nEstimated action values during the learning:")
estimated_action_values_during_training = algo_MC.find_action_values_yielding(  policy = policy_swim_randomly,
                                                                                env = RiverEnv(),
                                                                                n_episodes=1,
                                                                                gamma=0.98,
                                                                                visit_method="first_visit",
                                                                                averaging_method="moving",
                                                                                alpha=0.05,
                                                                                horizon=40,
                                                                                initial_action_values="zeros",
                                                                                typical_value=-10,
                                                                                # exploring_starts=True,
                                                                                # done_states = {0},
                                                                                    )
for estimated_action_values in estimated_action_values_during_training:
    print(estimated_action_values)