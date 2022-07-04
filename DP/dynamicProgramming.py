from typing import Tuple, Union
import numpy as np

from policies import *

class IterativePolicyEvaluation:

    def find_state_values(self,  policy : DiscretePolicyForDiscreteState, 
                                transition_probability : np.ndarray,
                                reward_probability : np.ndarray,
                                gamma : float = 1,
                                n_iterations : int = None, 
                                maximal_error : float = None,
                                verbose = 1,
                                ) -> np.ndarray:
        """This method perform the IterativePolicyEvaluation algorithm. It computes an estimation of the state values for a given policy, in a given model (transition_probability and reward_probability).
        The algorithm stop either after a given number of iterations or when the worst error (among the states) between two V(s) estimation consecutive is below a given threshold.
        """

        assert n_iterations != None or maximal_error != None, "The stop condition is not well defined. Please specify either n_iterations or maximal_error."

        n_states, n_actions = reward_probability.shape
        state_values = np.random.normal(loc = 0, scale = 2 * np.max(np.abs(reward_probability)), size = (n_states,))
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            for state in range(n_states):
                state = 10-state
                value = state_values[state]
                state_values[state] = self.compute_state_value(state, policy, transition_probability, reward_probability, state_values, gamma)
                worst_error = max(worst_error, abs(value - state_values[state]))
            n_iter += 1
            
            if n_iterations != None and n_iter >= n_iterations:
                keep_iterating = False
                if verbose >= 1: print("The algorithm stopped after {} iterations. Stop condition : number of iteration reached.".format(n_iter))
            elif maximal_error != None and worst_error <= maximal_error:
                keep_iterating = False
                if verbose >= 1: print("The algorithm stopped after {} iterations. Stop condition : worst error ({}) inferior to the maximal error asked ({})".format(n_iter, worst_error, maximal_error))
        
        return state_values

        

    def compute_state_value(self,   state : int, 
                                    policy : DiscretePolicyForDiscreteState, 
                                    transition_probability : np.ndarray,
                                    reward_probability : np.ndarray,
                                    state_values : np.ndarray,
                                    gamma : float) -> float:
        """This function compute the state value for a given state, a given policy, a given model (transition_probability and reward_probability), and for the state values vector.
        It applies the Bellman Operator to state values (the Bellman Operator is the right term in the Dynamic Bellman Equation for state values).
        """
        n_states, n_actions = reward_probability.shape
        value = 0
        for action in range(n_actions):
            value += policy.get_prob(state, action) * (reward_probability[state, action] + 
                                                        gamma * transition_probability[state, action, :].dot(state_values))
        return value



    
    def find_state_values_yielding(self,  policy : DiscretePolicyForDiscreteState,
                                transition_probability : np.ndarray,
                                reward_probability : np.ndarray,
                                n_iterations : int = None, 
                                maximal_error : float = None,
                                gamma : float = 1) -> np.ndarray:
        """This function is the same as find_state_values, but it yields the state values at each iteration. Use for observe the convergence of the algorithm.
        """

        assert n_iterations != None or maximal_error != None, "The stop condition is not well defined. Please specify either n_iterations or maximal_error."
        
        n_states, n_actions = reward_probability.shape
        state_values = np.random.normal(loc = 0, scale = 2 * np.max(np.abs(reward_probability)), size = (n_states,))
        yield state_values
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            for state in range(n_states):
                state = 10-state
                value = state_values[state]
                state_values[state] = self.compute_state_value(state, policy, transition_probability, reward_probability, state_values, gamma)
                worst_error = max(worst_error, abs(value - state_values[state]))
            n_iter += 1
            yield state_values
            if n_iterations != None and n_iter >= n_iterations:
                keep_iterating = False
            elif maximal_error != None and worst_error <= maximal_error:
                keep_iterating = False


    

    def find_action_values(self,    policy : DiscretePolicyForDiscreteState,
                                    transition_probability : np.ndarray,
                                    reward_probability : np.ndarray,
                                    n_iterations : int = None, 
                                    maximal_error : float = None,
                                    gamma : float = 1,
                                    verbose = 1,
                                    ) -> np.ndarray:
        
        """This method perform the IterativePolicyEvaluation algorithm. It computes an estimation of the action values for a given policy, in a given model (transition_probability and reward_probability).
        The algorithm stop either after a given number of iterations or when the worst error (among the states+actions) between two Q(s,a) estimation consecutive is below a given threshold.
        """
        
        assert n_iterations != None or maximal_error != None, "The stop condition is not well defined. Please specify either n_iterations or maximal_error."

        n_states, n_actions = reward_probability.shape
        q_values = np.random.normal(loc = 0, scale = 2 * np.max(np.abs(reward_probability)), size = (n_states, n_actions))
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            for state in range(n_states):
                state = 10-state
                for action in range(n_actions):
                    value = q_values[state][action]
                    q_values[state][action] = self.compute_action_value(state, action, policy, transition_probability, reward_probability, q_values, gamma)
                    worst_error = max(worst_error, abs(value - q_values[state][action]))
            n_iter += 1
            if n_iterations != None and n_iter >= n_iterations:
                keep_iterating = False
                if verbose >= 1: print("The algorithm stopped after {} iterations. Stop condition : number of iteration reached.".format(n_iter))
            elif maximal_error != None and worst_error <= maximal_error:
                keep_iterating = False
                if verbose >= 1: print("The algorithm stopped after {} iterations. Stop condition : worst error ({}) inferior to the maximal error asked ({})".format(n_iter, worst_error, maximal_error))

        return q_values

    
    def compute_action_value(self,  state : int, 
                                    action : int,
                                    policy : DiscretePolicyForDiscreteState, 
                                    transition_probability : np.ndarray,
                                    reward_probability : np.ndarray,
                                    q_values : np.ndarray,
                                    gamma : float) -> float:
        """This function compute the action value for a given state, action, a given policy, a given model (transition_probability and reward_probability), and for the action values vector.
        It applies the Bellman Operator to action values (the Bellman Operator is the right term in the Dynamic Bellman Equation for action values).
        """
        n_states, n_actions = reward_probability.shape
        value = reward_probability[state, action]
        for next_state in range(n_states):
            value += gamma * transition_probability[state, action, next_state] * policy.probs[next_state].dot(q_values[next_state])
        return value


    def find_action_values_yielding(self,    policy : DiscretePolicyForDiscreteState,
                                    transition_probability : np.ndarray,
                                    reward_probability : np.ndarray,
                                    n_iterations : int = None, 
                                    maximal_error : float = None,
                                    gamma : float = 1) -> np.ndarray:
        
        """This function is the same as find_action_values, but it yields the action values at each iteration. Use for observe the convergence of the algorithm.
        """

        assert n_iterations != None or maximal_error != None, "The stop condition is not well defined. Please specify either n_iterations or maximal_error."

        q_values = np.random.normal(loc = 0, scale = 2 * np.max(np.abs(reward_probability)), size = (policy.n_states, policy.n_actions))
        yield q_values
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            for state in range(policy.n_states):
                state=10-state
                for action in range(policy.n_actions):
                    value = q_values[state][action]
                    q_values[state][action] = self.compute_action_value(state, action, policy, transition_probability, reward_probability, q_values, gamma)
                    worst_error = max(worst_error, abs(value - q_values[state][action]))
            n_iter += 1
            yield q_values
            if n_iterations != None and n_iter >= n_iterations:
                keep_iterating = False
            elif maximal_error != None and worst_error <= maximal_error:
                keep_iterating = False



class PolicyIteration:

    def find_optimal_policy(self,   transition_probability : np.ndarray,
                                    reward_probability : np.ndarray,
                                    gamma : float = 1,
                                    IPE_n_iterations : int = None, 
                                    IPE_maximal_error : float = None,
                                    n_iterations : int = float("inf"),
                                    return_state_values : bool = False,
                                    return_state_actions_values : bool = False,
                                    verbose : int = 0,
                                    ) -> Union[ DiscretePolicyForDiscreteState, 
                                                Tuple[DiscretePolicyForDiscreteState, np.ndarray], 
                                                Tuple[DiscretePolicyForDiscreteState, np.ndarray, np.ndarray]]:

        """This method performs the Policy Iteration algorithm. It computes an optimal policy for a given model (transition_probability and reward_probability).
        The algorithm stop either when the policy is stable (no change in the policy) or when the number of iterations is reached.

        transition_probability : a numpy array of shape (n_states, n_actions, n_states) representing the transition probability between states and actions.
        reward_probability : a numpy array of shape (n_states, n_actions) representing the reward expected for each state and action.
        gamma : the discount factor
        IPE_n_iterations : the number of iterations for the IPE algorithm.
        IPE_maximal_error : the maximal error allowed for the IPE algorithm.
        n_iterations : the number of iterations for the policy iteration algorithm.
        return_state_values : if True, the state values are returned with the policy
        return_state_actions_values : if True, the actions values are returned with the policy and the state values.
        verbose : the verbosity level. 0 : no print, 1 : print when PI has finished.
        """
        assert n_iterations >= 1, "The number of iterations must be strictly positive."

        if IPE_maximal_error is None and IPE_n_iterations is None:
            IPE_maximal_error = 0.01

        n_states, n_actions = reward_probability.shape
        actions = np.random.choice(np.array([a for a in range(n_actions)]), size = n_states,)

        algo_IPE = IterativePolicyEvaluation()

        n_iter = 0
        while n_iter < n_iterations:

            #Iterative Policy Evaluation
            probs = np.zeros((n_states, n_actions))         # convert deterministic actions to stochastic policy
            probs[np.arange(n_states), actions] = 1
            policy = DiscretePolicyForDiscreteState(probs)
            state_values = algo_IPE.find_state_values(policy, transition_probability, reward_probability, gamma, 
                                                                                                        n_iterations = IPE_n_iterations, 
                                                                                                        maximal_error = IPE_maximal_error,
                                                                                                        verbose = 0,)
            #Policy improvement
            actions_old = actions.copy()
            for state in range(n_states):
                action = actions[state]
                actions[state] = np.argmax(reward_probability + gamma * transition_probability[state, action, :].dot(state_values))
            
            n_iter += 1
            if (actions == actions_old).all():
                break
        
        if verbose >= 1:
            if n_iter < n_iterations:
                print("Policy Iteration stopped after {} iterations. Stop condition : policy is stable.".format(n_iter))
            else:
                print("Policy Iteration stopped after {} iterations. Stop condition : maximal number of iterations reached.".format(n_iter))

        if return_state_actions_values:
            q_values = reward_probability + gamma * transition_probability[:, :, :].dot(state_values)
            return policy, state_values, q_values
        elif return_state_values:
            return policy, state_values
        else:
            return policy
    
    

    def find_optimal_policy_yielding(self,   transition_probability : np.ndarray,
                                    reward_probability : np.ndarray,
                                    gamma : float = 1,
                                    IPE_n_iterations : int = None, 
                                    IPE_maximal_error : float = None,
                                    n_iterations : int = float("inf"),
                                    return_state_values : bool = False,
                                    return_state_actions_values : bool = False,
                                    ) -> tuple:

        """This method performs the Policy Iteration algorithm as find_optimal_policy but yield probs (and eventually V and/or Q).
        """
        assert n_iterations >= 1, "The number of iterations must be strictly positive."

        if IPE_maximal_error is None and IPE_n_iterations is None:
            IPE_maximal_error = 0.01

        n_states, n_actions = reward_probability.shape
        actions = np.random.choice(np.array([a for a in range(n_actions)]), size = n_states,)

        algo_IPE = IterativePolicyEvaluation()

        n_iter = 0
        while n_iter < n_iterations:

            #Iterative Policy Evaluation
            probs = np.zeros((n_states, n_actions))         # convert deterministic actions to stochastic policy
            probs[np.arange(n_states), actions] = 1
            policy = DiscretePolicyForDiscreteState(probs)
            state_values = algo_IPE.find_state_values(policy, transition_probability, reward_probability, gamma, 
                                                                                                        n_iterations = IPE_n_iterations, 
                                                                                                        maximal_error = IPE_maximal_error,
                                                                                                        verbose = 0,)
            if return_state_actions_values:  
                q_values = reward_probability + gamma * transition_probability[:, :, :].dot(state_values)
                yield policy, state_values, q_values
            elif return_state_values:
                yield policy, state_values
            else:
                yield policy 

            #Policy improvement
            actions_old = actions.copy()
            for state in range(n_states):
                action = actions[state]
                actions[state] = np.argmax(reward_probability + gamma * transition_probability[state, action, :].dot(state_values))
            
            n_iter += 1
            if (actions == actions_old).all():
                break
        
        