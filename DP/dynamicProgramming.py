from policies import *

class IterativePolicyEvaluation:

    def find_state_values(self,  policy : DiscretePolicyForDiscreteState, 
                                transition_probability : np.ndarray,
                                reward_probability : np.ndarray,
                                gamma : float = 1,
                                n_iterations : int = None, 
                                maximal_error : float = None) -> np.ndarray:
        """This method perform the IterativePolicyEvaluation algorithm. It computes an estimation of the state values for a given policy, in a given model (transition_probability and reward_probability).
        The algorithm stop either after a given number of iterations or when the worst error (among the states) between two V(s) estimation consecutive is below a given threshold.
        """

        assert n_iterations != None or maximal_error != None, "The stop condition is not well defined. Please specify either n_iterations or maximal_error."

        state_values = np.random.normal(loc = 0, scale = 2 * np.max(np.abs(reward_probability)), size = (policy.n_states,))
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            for state in range(policy.n_states):
                state = 10-state
                value = state_values[state]
                state_values[state] = self.compute_state_value(state, policy, transition_probability, reward_probability, state_values, gamma)
                worst_error = max(worst_error, abs(value - state_values[state]))
            n_iter += 1
            if n_iterations != None and n_iter >= n_iterations:
                keep_iterating = False
                print("The algorithm stopped after {} iterations. Stop condition : number of iteration reached.".format(n_iter))
            elif maximal_error != None and worst_error <= maximal_error:
                keep_iterating = False
                print("The algorithm stopped after {} iterations. Stop condition : worst error ({}) inferior to the maximal error asked ({})".format(n_iter, worst_error, maximal_error))
        
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
        value = 0
        for action in range(policy.n_actions):
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

        state_values = np.random.normal(loc = 0, scale = 2 * np.max(np.abs(reward_probability)), size = (policy.n_states,))
        yield state_values
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            for state in range(policy.n_states):
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
                                    gamma : float = 1) -> np.ndarray:
        
        """This method perform the IterativePolicyEvaluation algorithm. It computes an estimation of the action values for a given policy, in a given model (transition_probability and reward_probability).
        The algorithm stop either after a given number of iterations or when the worst error (among the states+actions) between two Q(s,a) estimation consecutive is below a given threshold.
        """

        assert n_iterations != None or maximal_error != None, "The stop condition is not well defined. Please specify either n_iterations or maximal_error."

        q_values = np.random.normal(loc = 0, scale = 2 * np.max(np.abs(reward_probability)), size = (policy.n_states, policy.n_actions))
        n_iter = 0
        keep_iterating = True

        while keep_iterating:
            worst_error = 0
            for state in range(policy.n_states):
                state = 10-state
                for action in range(policy.n_actions):
                    value = q_values[state][action]
                    q_values[state][action] = self.compute_action_value(state, action, policy, transition_probability, reward_probability, q_values, gamma)
                    worst_error = max(worst_error, abs(value - q_values[state][action]))
            n_iter += 1
            if n_iterations != None and n_iter >= n_iterations:
                keep_iterating = False
                print("The algorithm stopped after {} iterations. Stop condition : number of iteration reached.".format(n_iter))
            elif maximal_error != None and worst_error <= maximal_error:
                keep_iterating = False
                print("The algorithm stopped after {} iterations. Stop condition : worst error ({}) inferior to the maximal error asked ({})".format(n_iter, worst_error, maximal_error))

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
        value = reward_probability[state, action]
        for next_state in range(policy.n_states):
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