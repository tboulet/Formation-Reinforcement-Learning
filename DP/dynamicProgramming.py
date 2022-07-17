from time import sleep, time
from typing import Iterator, Tuple, Union
import numpy as np

from policies import *
from utils import * 

class IterativePolicyEvaluation:

    def find_state_values(self,  policy : DiscretePolicyForDiscreteState, 
                                transition_probability : np.ndarray,
                                reward_probability : np.ndarray,
                                n_iterations : int = None, 
                                maximal_error : float = None,
                                gamma : float = 1,
                                sweep_order : str = "normal", # "normal" or "reverse" or "random"
                                initial_state_values : Union[np.ndarray, str] = "random", # "random", "zeros", "optimistic" or a numpy array
                                typical_value : float = 1,
                                verbose = 1,
                                **kwargs,
                                ) -> np.ndarray:
        """This method perform the IterativePolicyEvaluation algorithm. It computes an estimation of the state values for a given policy, in a given model (transition_probability and reward_probability).
        The algorithm stop either after a given number of iterations or when the worst error (among the states) between two V(s) estimation consecutive is below a given threshold.
        
        transition_probability : a numpy array of shape (n_states, n_actions, n_states) representing the transition probability between states.
        reward_probability : a numpy array of shape (n_states, n_actions) representing the reward probability for each action in each state.
        n_iterations : the number of iterations to perform.
        maximal_error : the error between 2 consecutives state value below what the algorithm will stop, considering that it has converged.
        gamma : the discount factor.
        sweep_order : the order in which we will iterate over the states. "normal" or "reverse" or "random". This can have a significant impact on the convergence of the algorithm.
        initial_state_values : the initial values of the state values. Can be "random", "zeros", "optimistic" or a numpy array.
        typical_value : the typical value of the state values. Used to initialize the state values if initial_state_values is "random".
        verbose : the verbosity level, 0 for no output, 1 for an end output.
        """

        assert n_iterations != None or maximal_error != None, "The stop condition is not well defined. Please specify either n_iterations or maximal_error."

        # Define the order in which we will iterate over the states
        n_states, n_actions = reward_probability.shape
        states_sweep = np.arange(n_states)
        if sweep_order == "reverse":
            states_sweep = np.flip(states_sweep)
        elif sweep_order == "random":
            np.random.shuffle(states_sweep)

        # Initialize the state values   
        state_values = initialize_values(   shape = (n_states,),
                                            initial_values=initial_state_values,
                                            typical_value=typical_value)
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            # Iterate over the states, update state value in an in-place manner (using only one array).
            for state in range(n_states):
                value = state_values[state]
                state_values[state] = self.compute_state_value(state, policy, transition_probability, reward_probability, state_values, gamma)
                worst_error = max(worst_error, abs(value - state_values[state]))
            # Stop algorithm if we reached the maximum number of iterations or if the error is below the threshold
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
                                gamma : float = 1,
                                sweep_order : str = "normal", # "normal" or "reverse" or "random"
                                initial_state_values : Union[np.ndarray, str] = "random", # "random", "zeros", "optimistic" or a numpy array
                                typical_value : float = 1,
                                yield_frequency : str = "step", # "step" or "iteration"
                                **kwargs,
                                ) -> Iterator:
        """This function is the same as find_state_values, but it yields the state values at each iteration. Use for observe the convergence of the algorithm.

        yield_frequency : "step" or "iteration", the frequency at which the state values are yielded.
        """

        assert n_iterations != None or maximal_error != None, "The stop condition is not well defined. Please specify either n_iterations or maximal_error."
        
        n_states, n_actions = reward_probability.shape
        states_sweep = np.arange(n_states)
        if sweep_order == "reverse":
            states_sweep = np.flip(states_sweep)
        elif sweep_order == "random":
            np.random.shuffle(states_sweep)

        state_values = initialize_values(   shape = (n_states,),
                                            initial_values=initial_state_values,
                                            typical_value=typical_value)
        if yield_frequency != "global_iteration": yield state_values
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            yield f"DP Prediction of V (IPE) - Iteration {n_iter} :"
            for state in states_sweep:
                value = state_values[state]
                state_values[state] = self.compute_state_value(state, policy, transition_probability, reward_probability, state_values, gamma)
                worst_error = max(worst_error, abs(value - state_values[state]))
                if yield_frequency == "step" : yield state_values
            n_iter += 1
            if n_iterations != None and n_iter >= n_iterations:
                keep_iterating = False
            elif maximal_error != None and worst_error <= maximal_error:
                keep_iterating = False
            if yield_frequency == "iteration": yield state_values
        if yield_frequency == "global_iteration" : yield state_values

    

    def find_action_values(self,    policy : DiscretePolicyForDiscreteState,
                                    transition_probability : np.ndarray,
                                    reward_probability : np.ndarray,
                                    n_iterations : int = None, 
                                    maximal_error : float = None,
                                    gamma : float = 1,
                                    sweep_order : str = "random", # "normal" or "reverse" or "random"
                                    initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                    typical_value : float = 1,
                                    verbose = 1,
                                    **kwargs,
                                    ) -> np.ndarray:
        
        """This method perform the IterativePolicyEvaluation algorithm. It computes an estimation of the action values for a given policy, in a given model (transition_probability and reward_probability).
        The algorithm stop either after a given number of iterations or when the worst error (among the states+actions) between two Q(s,a) estimation consecutive is below a given threshold.
        
        transition_probability : a numpy array of shape (n_states, n_actions, n_states) representing the transition probability between states.
        reward_probability : a numpy array of shape (n_states, n_actions) representing the reward probability for each action in each state.
        n_iterations : the number of iterations to perform.
        maximal_error : the error between 2 consecutives state value below what the algorithm will stop, considering that it has converged.
        gamma : the discount factor.
        sweep_order : the order in which we will iterate over the states. "normal" or "reverse" or "random". This can have a significant impact on the convergence of the algorithm.
        initial_action_values : the initial values of the action values. Can be "random", "zeros", "optimistic" or a numpy array.
        typical_value : the typical value of the action values. Used to initialize the action values if initial_action_values is "random".
        verbose : the verbosity level, 0 for no output, 1 for an end output.
        """
        
        assert n_iterations != None or maximal_error != None, "The stop condition is not well defined. Please specify either n_iterations or maximal_error."

        # Define the order in which we will iterate over the states
        n_states, n_actions = reward_probability.shape
        states_sweep = np.arange(n_states)
        if sweep_order == "reverse":
            states_sweep = np.flip(states_sweep)
        elif sweep_order == "random":
            np.random.shuffle(states_sweep)

        # Initialize the action values
        action_values = initialize_values(  shape = (n_states, n_actions), 
                                            initial_values = initial_action_values, 
                                            typical_value = typical_value)
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            # Iterate over the states and actions, update actions values value in an in-place manner (using only one array).
            for state in states_sweep:
                for action in range(n_actions):
                    value = action_values[state][action]
                    action_values[state][action] = self.compute_action_value(state, action, policy, transition_probability, reward_probability, action_values, gamma)
                    worst_error = max(worst_error, abs(value - action_values[state][action]))
            # Stop algorithm if we reached the maximum number of iterations or if the error is below the threshold
            n_iter += 1
            if n_iterations != None and n_iter >= n_iterations:
                keep_iterating = False
                if verbose >= 1: print("The algorithm stopped after {} iterations. Stop condition : number of iteration reached.".format(n_iter))
            elif maximal_error != None and worst_error <= maximal_error:
                keep_iterating = False
                if verbose >= 1: print("The algorithm stopped after {} iterations. Stop condition : worst error ({}) inferior to the maximal error asked ({})".format(n_iter, worst_error, maximal_error))

        return action_values

    
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


    def find_action_values_yielding(self,   policy : DiscretePolicyForDiscreteState,
                                            transition_probability : np.ndarray,
                                            reward_probability : np.ndarray,
                                            n_iterations : int = None, 
                                            maximal_error : float = None,
                                            gamma : float = 1,
                                            sweep_order : str = "random", # "normal" or "reverse" or "random"
                                            initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                            typical_value : float = 1,
                                            yield_frequency : str = "step", # "step" or "iteration"
                                            **kwargs,
                                            ) -> Iterator:
        
        """This function is the same as find_action_values, but it yields the action values at each iteration. Use for observe the convergence of the algorithm.

        yield_frequency : "step" or "iteration", the frequency at which the action values are yielded.
        """

        assert n_iterations != None or maximal_error != None, "The stop condition is not well defined. Please specify either n_iterations or maximal_error."

        n_states, n_actions = reward_probability.shape
        states_sweep = np.arange(n_states)
        if sweep_order == "reverse":
            states_sweep = np.flip(states_sweep)
        elif sweep_order == "random":
            np.random.shuffle(states_sweep)

        action_values = initialize_values(  shape = (n_states, n_actions), 
                                            initial_values = initial_action_values, 
                                            typical_value = typical_value)
        if yield_frequency != "global_iteration": yield action_values
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            yield f"DP Prediction of Q (IPE) - Iteration {n_iter} :"
            for state in states_sweep:
                for action in range(n_actions):
                    value = action_values[state][action]
                    action_values[state][action] = self.compute_action_value(state, action, policy, transition_probability, reward_probability, action_values, gamma)
                    worst_error = max(worst_error, abs(value - action_values[state][action]))
                    if yield_frequency == "step" : yield action_values
            n_iter += 1
            if n_iterations != None and n_iter >= n_iterations:
                keep_iterating = False
            elif maximal_error != None and worst_error <= maximal_error:
                keep_iterating = False
            if yield_frequency == "iteration": yield action_values
        if yield_frequency == "global_iteration" : yield action_values



class PolicyIteration:

    def find_optimal_policy(self,   transition_probability : np.ndarray,
                                    reward_probability : np.ndarray,
                                    n_iterations : int = float("inf"),
                                    IPE_n_iterations : int = None, 
                                    IPE_maximal_error : float = None,
                                    gamma : float = 1,
                                    sweep_order : str = "normal", # "normal" or "reverse" or "random"
                                    initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                    typical_value : float = 1,
                                    return_action_values : bool = False,
                                    verbose : int = 1,
                                    stop_if_policy_stable = True,
                                    ) -> DiscretePolicyForDiscreteState :

        """This method performs the Policy Iteration algorithm. It computes an optimal policy for a given model (transition_probability and reward_probability).
        The algorithm stop either when the policy is stable (no change in the policy) or when the number of iterations is reached.

        transition_probability : a numpy array of shape (n_states, n_actions, n_states) representing the transition probability between states and actions.
        reward_probability : a numpy array of shape (n_states, n_actions) representing the reward expected for each state and action.
        n_iterations : the number of iterations for the policy iteration algorithm.
        IPE_n_iterations : the number of iterations for the IPE algorithm.
        IPE_maximal_error : the maximal error allowed for the IPE algorithm.
        gamma : the discount factor
        sweep_order : the order in which we will iterate over the states. "normal" or "reverse" or "random". This can have a significant impact on the convergence of the algorithm.
        initial_values : the initial values for the action values ("random", "zeros", "optimistic" or a numpy array)
        typical_value : the typical value for the action values, used for scaling the "random" and "optimistic" value-initialization methods.
        return_action_values : if True, the action values are returned with the policy
        verbose : the verbosity level. 0 : no print, 1 : print when PI has finished.
        stop_if_policy_stable : if True, the algorithm stops when the policy is stable because it consider the policy has converged.
        """
        assert n_iterations >= 1, "The number of iterations must be strictly positive."

        if IPE_maximal_error is None and IPE_n_iterations is None:
            IPE_maximal_error = 0.01

        n_states, n_actions = reward_probability.shape
        actions = np.random.choice(np.array([a for a in range(n_actions)]), size = n_states,)
        action_values = initialize_values( shape = (n_states, n_actions), 
                                                    initial_values = initial_action_values, 
                                                    typical_value = typical_value)
        algo_IPE = IterativePolicyEvaluation()

        n_iter = 0
        while n_iter < n_iterations:

            #Iterative Policy Evaluation
            probs = np.zeros((n_states, n_actions))         # convert deterministic actions to stochastic policy
            probs[np.arange(n_states), actions] = 1
            policy = DiscretePolicyForDiscreteState(probs)
            action_values = algo_IPE.find_action_values(policy,                                 #Evaluate the policy
                                                        transition_probability, 
                                                        reward_probability,                                    
                                                        n_iterations = IPE_n_iterations,        #Convergence criteria for the IPE
                                                        maximal_error = IPE_maximal_error,
                                                        gamma = gamma,
                                                        sweep_order=sweep_order,

                                                        initial_action_values = action_values,  #Initialize the IPE with the previous action values computed, increase convergence a bit
                                                        verbose = 0,                            #Silence the IPE method
                                                        )
            #Policy improvement
            actions_old = actions.copy()
            for state in range(n_states):
                actions[state] = np.argmax(action_values[state])
            
            n_iter += 1
            if stop_if_policy_stable and (actions == actions_old).all():
                break
        
        if verbose >= 1:
            if n_iter < n_iterations:
                print("Policy Iteration stopped after {} iterations. Stop condition : policy is stable.".format(n_iter))
            else:
                print("Policy Iteration stopped after {} iterations. Stop condition : maximal number of iterations reached.".format(n_iter))

        if return_action_values:
            return policy, action_values
        else:
            return policy
    
    

    def find_optimal_policy_yielding(self,  transition_probability : np.ndarray,
                                            reward_probability : np.ndarray,
                                            IPE_n_iterations : int = None, 
                                            IPE_maximal_error : float = None,
                                            n_iterations : int = float("inf"),
                                            gamma : float = 1,
                                            sweep_order : str = "normal", # "normal" or "reverse" or "random"
                                            initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                            typical_value : float = 1,
                                            yield_frequency : str = "step", # "step", "iteration" or "global_iteration"
                                            stop_if_policy_stable = True,
                                            **kwargs,
                                            ) -> Iterator:

        """This function is the same as find_optimal_policy, but it yields the actions and action values at each iteration. Use for observe the convergence of the algorithm.

        yield_frequency : "step" or "iteration", the frequency at which the state values are yielded.
        """
        assert n_iterations >= 1, "The number of iterations must be strictly positive."

        if IPE_maximal_error is None and IPE_n_iterations is None:
            IPE_maximal_error = 0.01

        n_states, n_actions = reward_probability.shape
        actions = np.random.choice(np.array([a for a in range(n_actions)]), size = n_states,)
        action_values = initialize_values( shape = (n_states, n_actions), 
                                                    initial_values = initial_action_values, 
                                                    typical_value = typical_value)
        yield actions
        yield action_values
        algo_IPE = IterativePolicyEvaluation()

        n_iter = 0
        while n_iter < n_iterations:
            yield f"DP Control (PI or VI) - Iteration {n_iter}"
            #Iterative Policy Evaluation
            probs = np.zeros((n_states, n_actions))         # convert deterministic actions to stochastic policy
            probs[np.arange(n_states), actions] = 1
            policy = DiscretePolicyForDiscreteState(probs)
            for action_values_or_str in algo_IPE.find_action_values_yielding(   policy,                                 #Evaluate the policy
                                                                    transition_probability, 
                                                                    reward_probability,                                    
                                                                    n_iterations = IPE_n_iterations,        #Convergence criteria for the IPE
                                                                    maximal_error = IPE_maximal_error,
                                                                    gamma = gamma,
                                                                    sweep_order=sweep_order,

                                                                    initial_action_values = action_values,  #Initialize the IPE with the previous action values computed, increase convergence a bit
                                                                    yield_frequency=yield_frequency,
                                                                    ):
                yield action_values_or_str
            #Policy improvement
            actions_old = actions.copy()
            for state in range(n_states):
                actions[state] = np.argmax(action_values[state])
            yield actions
            yield action_values
            n_iter += 1
            if stop_if_policy_stable and (actions == actions_old).all():
                break
        


class ValueIteration(PolicyIteration):
    
    algo_PI = PolicyIteration()

    def find_optimal_policy(self,   transition_probability : np.ndarray,
                                    reward_probability : np.ndarray,
                                    n_iterations : int = None, 
                                    gamma : float = 1,
                                    sweep_order : str = "normal", # "normal" or "reverse" or "random"
                                    initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                    typical_value : float = 1,
                                    return_action_values : bool = False,
                                    verbose : int = 1,
                                    ) -> DiscretePolicyForDiscreteState:
        """This class implements the Value Iteration algorithm. It computes an optimal value function for a given model (transition_probability and reward_probability).
        The algorithm stop either when the value function is stable (no change in the value function) or when the number of iterations is reached.

        transition_probability : a numpy array of shape (n_states, n_actions, n_states) representing the transition probability between states and actions.
        reward_probability : a numpy array of shape (n_states, n_actions) representing the reward expected for each state and action.
        n_iterations : the number of iterations for the policy iteration algorithm.
        gamma : the discount factor
        sweep_order : the order in which we will iterate over the states. "normal" or "reverse" or "random". This can have a significant impact on the convergence of the algorithm.
        initial_values : the initial values for the action values ("random", "zeros", "optimistic" or a numpy array)
        typical_value : the typical value for the action values, used for scaling the "random" and "optimistic" value-initialization methods.
        return_action_values : if True, the action values are returned with the policy
        verbose : the verbosity level. 0 : no print, 1 : print when PI has finished.
        """
        results = self.algo_PI.find_optimal_policy( transition_probability = transition_probability,
                                                    reward_probability = reward_probability,
                                                    n_iterations=n_iterations,
                                                    IPE_n_iterations=1,
                                                    gamma = gamma,
                                                    sweep_order=sweep_order,
                                                    initial_action_values=initial_action_values,
                                                    typical_value=typical_value,
                                                    return_action_values = return_action_values,
                                                    stop_if_policy_stable = False,
                                                    verbose = 0,)
        
        if verbose >= 1:
            print("Value Iteration finished.")
        
        return results



    def find_optimal_policy_yielding(self,  transition_probability : np.ndarray,
                                            reward_probability : np.ndarray,
                                            n_iterations : int = None, 
                                            gamma : float = 1,
                                            sweep_order : str = "normal", # "normal" or "reverse" or "random"
                                            initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                            typical_value : float = 1,
                                            yield_frequency : str = "step", # "step", "iteration" or "global_iteration"
                                            **kwargs,
                                            ) -> Iterator:
        """This method performs the Policy Iteration algorithm as find_optimal_policy but yield pi(s) (the actions) and Q(s,a).
        """
        results = self.algo_PI.find_optimal_policy_yielding(    transition_probability = transition_probability,
                                                                reward_probability = reward_probability,
                                                                n_iterations=n_iterations,
                                                                IPE_n_iterations=1,
                                                                gamma = gamma,
                                                                sweep_order=sweep_order,
                                                                initial_action_values=initial_action_values,
                                                                typical_value=typical_value,
                                                                yield_frequency=yield_frequency,
                                                                stop_if_policy_stable=False,)
        
        return results