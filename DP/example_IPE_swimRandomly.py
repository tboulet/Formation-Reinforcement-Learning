import numpy as np

from utils import *
from env.oceanEnv import transition_probability_ocean, reward_probability_ocean
from DP.dynamicProgramming import IterativePolicyEvaluation
from policies import DiscretePolicyForDiscreteState

policy_swim_randomly = DiscretePolicyForDiscreteState(probs = np.array([[0.8, 0.2] for _ in range(11)]))


algo_IPE = IterativePolicyEvaluation()

print("\nComputing state values for the policy swim_randomly...")
estimated_state_values = algo_IPE.find_state_values(policy = policy_swim_randomly,
                                                    transition_probability = transition_probability_ocean,
                                                    reward_probability = reward_probability_ocean,
                                                    n_iterations = 100,
                                                    maximal_error = 0.01,
                                                    gamma=.98)
print("Estimated state values :", estimated_state_values)

print("\nEstimated state values during the learning:")
estimated_state_values_during_training = algo_IPE.find_state_values_yielding(   policy = policy_swim_randomly,
                                                                                transition_probability = transition_probability_ocean,
                                                                                reward_probability = reward_probability_ocean,
                                                                                n_iterations = 1,
                                                                                maximal_error = 0.01,
                                                                                gamma = .98)
for n_iter, estimated_state_values in enumerate(estimated_state_values_during_training):
    print(f"Iteration {n_iter} :", estimated_state_values)

print("\nComputing action values for the policy swim_randomly...")
estimated_action_values = algo_IPE.find_action_values(  policy = policy_swim_randomly,
                                                        transition_probability=transition_probability_ocean,
                                                        reward_probability=reward_probability_ocean,
                                                        n_iterations=100,
                                                        maximal_error=0.01,
                                                        gamma=.98)
print("Estimated action values :", estimated_action_values)

print("\nEstimated action values during the learning:")
estimated_action_values_during_training = algo_IPE.find_action_values_yielding( policy = policy_swim_randomly,
                                                                                transition_probability = transition_probability_ocean,
                                                                                reward_probability = reward_probability_ocean,
                                                                                n_iterations = 1,
                                                                                maximal_error = 0.01,
                                                                                gamma = .98)
for n_iter, estimated_action_values in enumerate(estimated_action_values_during_training):
    print(f"Iteration {n_iter} :", estimated_action_values)