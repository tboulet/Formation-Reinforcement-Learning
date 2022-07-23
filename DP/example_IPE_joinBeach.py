import numpy as np

from utils import *
from env.oceanEnv import transition_probability_ocean, reward_probability_ocean
from DP.dynamicProgramming import IterativePolicyEvaluation
from policies import DiscretePolicyForDiscreteState

policy_join_beach = DiscretePolicyForDiscreteState(probs = np.array([[1, 0] for _ in range(11)]))

algo_IPE = IterativePolicyEvaluation()

print("\nComputing state values for the policy join_beach...")
estimated_state_values = algo_IPE.find_state_values(policy = policy_join_beach,
                                                    transition_probability = transition_probability_ocean,
                                                    reward_probability = reward_probability_ocean,
                                                    n_iterations = 5,
                                                    maximal_error = 0.01,
                                                    gamma=1,)
print("Estimated state values :", estimated_state_values)

print("\nEstimated state values during the learning:")
estimated_state_values_during_training = algo_IPE.find_state_values_yielding(   policy = policy_join_beach,
                                                                                transition_probability = transition_probability_ocean,
                                                                                reward_probability = reward_probability_ocean,
                                                                                n_iterations = 12,
                                                                                maximal_error = 0.01,
                                                                                gamma = 1,
                                                                                )
for n_iter, estimated_state_values in enumerate(estimated_state_values_during_training):
    print(estimated_state_values)

print("\nComputing action values for the policy join_beach...")
estimated_action_values = algo_IPE.find_action_values(  policy = policy_join_beach,
                                                        transition_probability=transition_probability_ocean,
                                                        reward_probability=reward_probability_ocean,
                                                        n_iterations=12,
                                                        maximal_error=0.01,
                                                        gamma=1)
print("Estimated action values :", estimated_action_values)

print("\nEstimated action values during the learning:")
estimated_action_values_during_training = algo_IPE.find_action_values_yielding( policy = policy_join_beach,
                                                                                transition_probability = transition_probability_ocean,
                                                                                reward_probability = reward_probability_ocean,
                                                                                n_iterations = 12,
                                                                                maximal_error = 0.01,
                                                                                gamma = 1)
for n_iter, estimated_action_values in enumerate(estimated_action_values_during_training):
    print(estimated_action_values)