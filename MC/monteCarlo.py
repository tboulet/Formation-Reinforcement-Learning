import numpy as np
import gym

from policies import *
from utils import Q_State

class MonteCarlo:

    def find_state_values(self,  policy : DiscretePolicyForDiscreteState,
                                env : gym.Env,
                                n_episodes : int,
                                gamma : float = 0.99,
                                method : str = "first_visit", # "first_visit" or "every_visit"
                                horizon : int = float("inf"),
                                ) -> np.ndarray:
        """
        This method perform the MonteCarlo algorithm. It computes an estimation of the state values for a given policy.
        The algorithm stop after a certain number of iterations.
        """
        assert env.observation_space.n == policy.n_states, "The number of states in the environment must be equal to the number of states in the policy."
        assert env.action_space.n == policy.n_actions, "The number of actions in the environment must be equal to the number of actions in the policy."

        state_values = np.zeros(policy.n_states)
        nbr_state_seen_in_episode = dict()
        num_ep = 0
        initial_value = lambda: 0

        while num_ep < n_episodes:
            state = env.reset()
            t = 0
            states_returns = dict()                                       #dict mapping states to returns G = sum(gamma^t * r) of the coming episode
            states_first_time_seen = dict()                               #dict mapping states to the first time they are seen in the coming episode
            states_was_seen = set()                                       #set of seen states in the coming episode

            states_returns[state] = initial_value()     
            states_first_time_seen[state] = t    
            states_was_seen.add(state) 

            #Run one episode
            done = False
            while not done:
                action = np.random.choice(policy.n_actions, p=policy.probs[state])
                next_state, reward, done, _ = env.step(action)
                if t >= horizon: done = True
                if done: continue  # if the episode is done, we don't need to update the state values, by convention V(s_terminal) = 0

                if not next_state in states_was_seen:
                    states_returns[next_state] = initial_value()     
                    states_first_time_seen[next_state] = t    
                    states_was_seen.add(next_state)     

                for state in states_was_seen:
                    states_returns[state] += reward * (gamma ** (t-states_first_time_seen[state]))

                t += 1
                state = next_state
                            
            #Update incrementally the state values
            for state in states_was_seen:
                #Add 1 to the number of times the state was seen, define N by the way
                if state in nbr_state_seen_in_episode:
                    N = nbr_state_seen_in_episode[state]
                    nbr_state_seen_in_episode[state] = N + 1
                else: 
                    N = 0
                    nbr_state_seen_in_episode[state] = N + 1
                #Update the state value
                G = states_returns[state]
                state_values[state] = (N/(N+1)) * state_values[state] + (1/(N+1)) * G
            num_ep += 1
        
        return state_values

    

    def find_state_values_yielding(self,     policy : DiscretePolicyForDiscreteState,
                                            env : gym.Env,
                                            n_episodes : int,
                                            gamma : float = 0.99,
                                            method : str = "first_visit", # "first_visit" or "every_visit"
                                            horizon : int = float("inf"),
                                ) -> np.ndarray:
        """
        Same as find_state_values but yielding state values at each iteration instead of returning them at the end.
        """
        assert env.observation_space.n == policy.n_states, "The number of states in the environment must be equal to the number of states in the policy."
        assert env.action_space.n == policy.n_actions, "The number of actions in the environment must be equal to the number of actions in the policy."

        state_values = np.zeros(policy.n_states)
        yield state_values
        nbr_state_seen_in_episode = np.zeros(policy.n_states)
        num_ep = 0
        initial_value = lambda: 0

        while num_ep < n_episodes:
            state = env.reset()
            t = 0
            states_returns = dict()                                       #dict mapping states to returns G = sum(gamma^t * r) of the coming episode
            states_first_time_seen = dict()                               #dict mapping states to the first time they are seen in the coming episode
            states_was_seen = set()                                       #set of seen states in the coming episode

            states_returns[state] = initial_value()     
            states_first_time_seen[state] = t    
            states_was_seen.add(state) 

            #Run one episode
            done = False
            while not done:
                action = np.random.choice(policy.n_actions, p=policy.probs[state])
                next_state, reward, done, _ = env.step(action)
                if t >= horizon: done = True
                if done: continue  # if the episode is done, we don't need to update the state values, by convention V(s_terminal) = 0
                if not next_state in states_was_seen:
                    states_returns[next_state] = initial_value()     
                    states_first_time_seen[next_state] = t    
                    states_was_seen.add(next_state)     

                for state in states_was_seen:
                    states_returns[state] += reward * (gamma ** (t-states_first_time_seen[state]))

                t += 1
                state = next_state
                            
            #Update incrementally the state values
            for state in states_was_seen:
                #Add 1 to the number of times the state was seen, define N by the way
                if state in nbr_state_seen_in_episode:
                    N = nbr_state_seen_in_episode[state]
                    nbr_state_seen_in_episode[state] = N + 1
                else: 
                    N = 0
                    nbr_state_seen_in_episode[state] = N + 1
                #Update the state value
                G = states_returns[state]
                state_values[state] = (N/(N+1)) * state_values[state] + (1/(N+1)) * G
                yield state_values
            num_ep += 1
            



    def find_action_values(self,    policy : DiscretePolicyForDiscreteState,
                                    env : gym.Env,
                                    n_episodes : int,
                                    gamma : float = 0.99,
                                    method : str = "first_visit", # "first_visit" or "every_visit"
                                    horizon : int = float("inf"),
                                    ) -> np.ndarray:
            """
            This method perform the MonteCarlo algorithm. It computes an estimation of the action values for a given policy in a given env.
            The algorithm stop after a certain number of iterations.
            """
            assert env.observation_space.n == policy.n_states, "The number of states in the environment must be equal to the number of states in the policy."
            assert env.action_space.n == policy.n_actions, "The number of actions in the environment must be equal to the number of actions in the policy."

            action_values = np.zeros((policy.n_states, policy.n_actions))
            nbr_qstate_seen_in_episode = dict()
            num_ep = 0
            initial_value = lambda: 0

            while num_ep < n_episodes:
                state = env.reset()
                t = 0
                qstates_returns = dict()                                     
                qstates_first_time_seen = dict()                        
                qstates_was_seen = set()                                     

                #Run one episode
                done = False
                while not done:
                    action = np.random.choice(policy.n_actions, p=policy.probs[state])
                    qstate = Q_State(state, action)
                    
                    if not qstate in qstates_was_seen:
                        qstates_returns[qstate] = initial_value()     
                        qstates_first_time_seen[qstate] = t    
                        qstates_was_seen.add(qstate) 

                    next_state, reward, done, _ = env.step(action)

                    for qstate in qstates_was_seen:
                        qstates_returns[qstate] += reward * (gamma ** (t-qstates_first_time_seen[qstate]))

                    t += 1
                    state = next_state
                    if t >= horizon: done = True
                                
                #Update incrementally the state values
                for qstate in qstates_was_seen:
                    #Add 1 to the number of times the state was seen, define N by the way
                    if qstate in nbr_qstate_seen_in_episode:
                        N = nbr_qstate_seen_in_episode[qstate]
                        nbr_qstate_seen_in_episode[qstate] = N + 1
                    else: 
                        N = 0
                        nbr_qstate_seen_in_episode[qstate] = N + 1
                    #Update the state value
                    G = qstates_returns[qstate]
                    state, action = qstate.observation, qstate.action
                    action_values[state][action] = (N/(N+1)) * action_values[state][action] + (1/(N+1)) * G
                num_ep += 1
            
            return action_values
    

    def find_action_values_yielding(self,    policy : DiscretePolicyForDiscreteState,
                                    env : gym.Env,
                                    n_episodes : int,
                                    gamma : float = 0.99,
                                    method : str = "first_visit", # "first_visit" or "every_visit"
                                    horizon : int = float("inf"),
                                    ) -> np.ndarray:
            """
            This method perform the MonteCarlo algorithm. It computes an estimation of the action values for a given policy in a given env.
            The algorithm stop after a certain number of iterations.
            """
            assert env.observation_space.n == policy.n_states, "The number of states in the environment must be equal to the number of states in the policy."
            assert env.action_space.n == policy.n_actions, "The number of actions in the environment must be equal to the number of actions in the policy."

            action_values = np.zeros((policy.n_states, policy.n_actions))
            yield action_values
            nbr_qstate_seen_in_episode = dict()
            num_ep = 0
            initial_value = lambda: 0

            while num_ep < n_episodes:
                state = env.reset()
                t = 0
                qstates_returns = dict()                                     
                qstates_first_time_seen = dict()                        
                qstates_was_seen = set()                                     

                #Run one episode
                done = False
                while not done:
                    action = np.random.choice(policy.n_actions, p=policy.probs[state])
                    qstate = Q_State(state, action)
                    
                    if not qstate in qstates_was_seen:
                        qstates_returns[qstate] = initial_value()     
                        qstates_first_time_seen[qstate] = t    
                        qstates_was_seen.add(qstate) 

                    next_state, reward, done, _ = env.step(action)

                    for qstate in qstates_was_seen:
                        qstates_returns[qstate] += reward * (gamma ** (t-qstates_first_time_seen[qstate]))

                    t += 1
                    state = next_state
                    if t >= horizon: done = True
                                
                #Update incrementally the state values
                for qstate in qstates_was_seen:
                    #Add 1 to the number of times the state was seen, define N by the way
                    if qstate in nbr_qstate_seen_in_episode:
                        N = nbr_qstate_seen_in_episode[qstate]
                        nbr_qstate_seen_in_episode[qstate] = N + 1
                    else: 
                        N = 0
                        nbr_qstate_seen_in_episode[qstate] = N + 1
                    #Update the state value
                    G = qstates_returns[qstate]
                    state, action = qstate.observation, qstate.action
                    action_values[state][action] = (N/(N+1)) * action_values[state][action] + (1/(N+1)) * G
                    yield action_values
                num_ep += 1
            
