import numpy as np

from utils import *
from riverEnv import RiverEnv
from TD.TDLearning import TDLearning
from policies import DiscretePolicyForDiscreteState

policy_swim_randomly = DiscretePolicyForDiscreteState(probs = np.array([[1, 0] for _ in range(11)]))

algo_TD = TDLearning()

print("\nComputing state values for the policy policy_swim_randomly...")
estimated_state_values = algo_TD.find_state_values( policy = policy_swim_randomly,
                                                    env = RiverEnv(),
                                                    n_episodes = 500,
                                                    n_steps = 100000,
                                                    gamma=0.99,
                                                    alpha=0.1,
                                                    horizon=40,
                                                    initial_state_values="random",
                                                    typical_value = -5,
                                                    exploring_starts=False,
                                                    is_state_done=lambda state: state == 0,
                                                    verbose=0,
                                                    )
print("Estimated state values :", estimated_state_values)

print("\nEstimated state values during the learning:")
estimated_state_values_during_training = algo_TD.find_state_values_yielding(policy = policy_swim_randomly,
                                                                            env = RiverEnv(),
                                                                            n_episodes = 1,
                                                                            n_steps = 10,
                                                                            gamma=0.99,
                                                                            alpha=0.1,
                                                                            horizon=40,
                                                                            initial_state_values="random",
                                                                            typical_value = -5,
                                                                            exploring_starts=False,
                                                                            is_state_done=lambda state: state == 0,

                                                                            yield_frequency="step",
                                                                                )
for estimated_state_values in estimated_state_values_during_training:
    print(estimated_state_values)

print("\nComputing action values for the policy policy_swim_randomly...")
estimated_action_values = algo_TD.find_action_values(   policy = policy_swim_randomly,
                                                        env = RiverEnv(),
                                                        n_episodes = 100,
                                                        n_steps = float("inf"),
                                                        gamma=0.99,
                                                        alpha=0.1,
                                                        horizon=40,
                                                        initial_action_values="zeros",
                                                        typical_value = -5,
                                                        exploring_starts=False,
                                                        is_state_done=lambda state: state == 0,
                                                        verbose=0,
                                                            )
print("Estimated action values :", estimated_action_values)

print("\nEstimated action values during the learning:")
estimated_action_values_during_training = algo_TD.find_action_values_yielding(  policy = policy_swim_randomly,
                                                                                env = RiverEnv(),
                                                                                n_episodes = 1,
                                                                                n_steps = 10,
                                                                                gamma=0.99,
                                                                                alpha=0.1,
                                                                                horizon=40,
                                                                                initial_action_values="random",
                                                                                typical_value = -5,
                                                                                exploring_starts=False,
                                                                                is_state_done=lambda state: state == 0,

                                                                                yield_frequency="step",
                                                                                    )
for estimated_action_values in estimated_action_values_during_training:
    print(estimated_action_values)