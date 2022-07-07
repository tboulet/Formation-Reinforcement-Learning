from typing import Callable, Iterator, Tuple, Union
import numpy as np
import gym

from policies import *
from utils import Q_State, Scheduler, pretty_announcer

class MonteCarlo:

    def find_state_values(self,  policy : DiscretePolicyForDiscreteState,
                                env : gym.Env,
                                n_episodes : int,
                                gamma : float = 0.99,
                                visit_method : str = "first_visit", # "first_visit" or "every_visit"
                                averaging_method : str = "cumulative", # "cumulative" or "moving"
                                alpha : float = 0.1,
                                horizon : int = float("inf"),
                                initial_state_values : Union[np.ndarray, str] = "random", # "random", "zeros", "optimistic" or a numpy array
                                typical_value : float = 1.,
                                exploring_starts : bool = False,
                                is_state_done : Callable = None,
                                verbose : int = 1,
                                ) -> np.ndarray:
        """
        This method perform the MonteCarlo algorithm. It computes an estimation of the state values for a given policy.
        The algorithm stop after a certain number of iterations.

        policy : the policy to evaluate
        env : the environment to evaluate the policy on
        n_episodes : the number of episodes of interaction with the env to perform the algorithm
        gamma : the discount factor
        visit_method : the method to use to update the state values, currently only "first_visit" is supported
        averaging_method : the method to use to update the action values. Cumulative is to "tend to" V, while moving is to permanently "track" V. Use the latest for non stationary env.
        alpha : the learning rate
        horizon : the number of maximal steps in an episode. After that the episode will be considered done. Use for non terminal env.
        initial_state_values : the initial values of the state values. Can be "random", "zeros", "optimistic" or a numpy array.
        typical_value : the typical value of the state values. Used to initialize the state values if initial_state_values is "random".
        exploring_starts : if True, the algorithm will start at a random-non terminal state. Use IF accessible env. Use for create minimum exploration in the case of deterministic policies.
        is_state_done : a function returning whether a state is terminal. Used if exploring_starts is True for no initialization in the terminal states
        """

        if verbose >= 1 : 
            print(pretty_announcer(f"Start algorithm Monte Carlo Prediction for V ({visit_method}, {averaging_method}-average).\nExploring starts : {exploring_starts}\nEpisodes : {n_episodes}."))

        assert env.observation_space.n == policy.n_states, "The number of states in the environment must be equal to the number of states in the policy."
        assert env.action_space.n == policy.n_actions, "The number of actions in the environment must be equal to the number of actions in the policy."
        assert n_episodes > 0, "The number of episodes must be positive."
        assert not exploring_starts or is_state_done is not None, "The is_state_done function must be provided if exploring_starts is True."

        # Initialize the state values
        state_values = self.initialize_values(shape = (policy.n_states,),
                                                initial_values = initial_state_values,
                                                typical_value = typical_value)
        nbr_state_seen_in_episode = dict()
        num_ep = 0
        initial_value = lambda: 0

        while num_ep < n_episodes:
            if verbose >= 1 : print(f"MC Prediction of V - Episode {num_ep}/{n_episodes}")

            # Reset the env.
            if not exploring_starts:
                state = env.reset()
            # If exploring_starts is True, the algorithm will start at a random-non terminal state. This can only be used if you can define correctly a is_don_state(state) function AND if you can set the state of your env using "env.state = the_new_state".
            else:
                state_temp = np.random.randint(0, policy.n_states-1)
                if not is_state_done(state_temp):
                    state = state_temp
                    env.state = state_temp
                else:
                    state = env.reset()

            t = 0
            states_returns = dict()                                       #dict mapping states to returns G = sum(gamma^t * r) of the coming episode
            states_first_time_seen = dict()                               #dict mapping states to the first time they are seen in the coming episode
            states_was_seen = set()                                       #set of seen states in the coming episode

            #Run one episode
            done = False
            while not done:
                action = np.random.choice(policy.n_actions, p=policy.probs[state])
                next_state, reward, done, _ = env.step(action)

                # First visit of a state : we remember the instant we saw it.
                if not state in states_was_seen:
                    states_returns[state] = initial_value()   
                    states_first_time_seen[state] = t    
                    states_was_seen.add(state)     

                # Add a discounted reward to the return of each already seen state
                for state in states_was_seen:
                    states_returns[state] += reward * (gamma ** (t-states_first_time_seen[state]))

                t += 1
                state = next_state

                # If s' is a terminal state, we will still take into account the returns of this terminal state (which is 0 by convention) in the computation of V(s')
                if done: 
                    if not state in states_was_seen:
                        states_returns[state] = 0
                        states_first_time_seen[state] = t    
                        states_was_seen.add(state) 

                # Horizon : we artificially set the episode as done if the horizon is reached
                if t >= horizon: done = True    

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
                if averaging_method == "cumulative":
                    state_values[state] = (N/(N+1)) * state_values[state] + (1/(N+1)) * G
                elif averaging_method == "moving":
                    state_values[state] = (1-alpha) * state_values[state] + alpha * G
                else:
                    raise ValueError("The averaging method must be either 'cumulative' or 'moving'.")
            
            num_ep += 1
        
        if verbose >= 1:
            print(f"MonteCarlo Prediction of V finished after {num_ep} episodes. State values found : {state_values}")
            
        return state_values

    

    def find_state_values_yielding(self,  policy : DiscretePolicyForDiscreteState,
                                env : gym.Env,
                                n_episodes : int,
                                gamma : float = 0.99,
                                visit_method : str = "first_visit", # "first_visit" or "every_visit"
                                averaging_method : str = "cumulative", # "cumulative" or "moving"
                                alpha : float = 0.1,
                                horizon : int = float("inf"),
                                initial_state_values : Union[np.ndarray, str] = "random", # "random", "zeros", "optimistic" or a numpy array
                                typical_value : float = 1.,
                                exploring_starts : bool = False,
                                is_state_done : Callable = None,
                                ) -> Iterator:
        """
        Same as find_state_values, but yields the state values at each step.
        """
        assert env.observation_space.n == policy.n_states, "The number of states in the environment must be equal to the number of states in the policy."
        assert env.action_space.n == policy.n_actions, "The number of actions in the environment must be equal to the number of actions in the policy."
        assert n_episodes > 0, "The number of episodes must be positive."
        assert not exploring_starts or is_state_done is not None, "The is_state_done function must be provided if exploring_starts is True."

        # Initialize the state values
        state_values = self.initialize_values(shape = (policy.n_states,),
                                                initial_values = initial_state_values,
                                                typical_value = typical_value)
        yield state_values
        nbr_state_seen_in_episode = dict()
        num_ep = 0
        initial_value = lambda: 0

        while num_ep < n_episodes:
            yield f"MC Prediction of V - Episode {num_ep}/{n_episodes}"

            # Reset the env.
            if not exploring_starts:
                state = env.reset()
            # If exploring_starts is True, the algorithm will start at a random-non terminal state. This can only be used if you can define correctly a is_don_state(state) function AND if you can set the state of your env using "env.state = the_new_state".
            else:
                state_temp = np.random.randint(0, policy.n_states-1)
                if not is_state_done(state_temp):
                    state = state_temp
                    env.state = state_temp
                else:
                    state = env.reset()

            t = 0
            states_returns = dict()                                       #dict mapping states to returns G = sum(gamma^t * r) of the coming episode
            states_first_time_seen = dict()                               #dict mapping states to the first time they are seen in the coming episode
            states_was_seen = set()                                       #set of seen states in the coming episode

            #Run one episode
            done = False
            while not done:
                action = np.random.choice(policy.n_actions, p=policy.probs[state])
                next_state, reward, done, _ = env.step(action)

                # First visit of a state : we remember the instant we saw it.
                if not state in states_was_seen:
                    states_returns[state] = initial_value()   
                    states_first_time_seen[state] = t    
                    states_was_seen.add(state)     

                # Add a discounted reward to the return of each already seen state
                for state in states_was_seen:
                    states_returns[state] += reward * (gamma ** (t-states_first_time_seen[state]))

                t += 1
                state = next_state

                # If s' is a terminal state, we will still take into account the returns of this terminal state (which is 0 by convention) in the computation of V(s')
                if done: 
                    if not state in states_was_seen:
                        states_returns[state] = 0
                        states_first_time_seen[state] = t    
                        states_was_seen.add(state) 

                # Horizon : we artificially set the episode as done if the horizon is reached
                if t >= horizon: done = True    

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
                if averaging_method == "cumulative":
                    state_values[state] = (N/(N+1)) * state_values[state] + (1/(N+1)) * G
                elif averaging_method == "moving":
                    state_values[state] = (1-alpha) * state_values[state] + alpha * G
                else:
                    raise ValueError("The averaging method must be either 'cumulative' or 'moving'.")
                yield state_values
            num_ep += 1            



    def find_action_values(self,    policy : DiscretePolicyForDiscreteState,
                                    env : gym.Env,
                                    n_episodes : int,
                                    gamma : float = 0.99,
                                    visit_method : str = "first_visit", # "first_visit" or "every_visit"
                                    averaging_method : str = "cumulative", # "cumulative" or "moving"
                                    alpha : float = 0.1, 
                                    horizon : int = float("inf"),
                                    initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                    typical_value : float = 1,
                                    exploring_starts : bool = False,
                                    exploring_starts_lenght : int = 1, #WIP. This will increase the duration of the explorative policy at the beginning of each episode, thus leading to even more exploration. Slightly off policy and so biased.
                                    is_state_done : Callable = None,
                                    verbose = 1,
                                    ) -> np.ndarray:
            """
            This method perform the MonteCarlo algorithm. It computes an estimation of the action values for a given policy.
            The algorithm stop after a certain number of iterations.
                
            policy : the policy to evaluate
            env : the environment to evaluate the policy on
            n_episodes : the number of episodes of interaction with the env to perform the algorithm
            gamma : the discount factor
            visit_method : the method to use to update the state values, currently only "first_visit" is supported
            averaging_method : the method to use to update the action values. Cumulative is to "tend to" Q, while moving is to permanently "track" Q. Use the latest for non stationary env.
            alpha : the learning rate (for the moving average)
            horizon : the number of maximal steps in an episode. After that the episode will be considered done. Use for non terminal env.
            initial_action_values : the initial action values. Can be "random", "zeros", "optimistic" or a numpy array.
            typical_value : the typical value of the action values. Used to initialize the action values if initial_action_values is "optimistic".
            exploring_starts : if True, each env will start at a random state and a random action will be played. Use IF accessible env. Use for create minimum exploration in the case of deterministic policies.
            is_state_done : a function returning whether a state is terminal. Used if exploring_starts is True for no initialization in the terminal states
            verbose : verbosity level. 0 is silent, 1 is the default
            """
            if verbose >= 1 : 
                print(pretty_announcer(f"Start algorithm Monte Carlo Prediction for Q ({visit_method}, {averaging_method}-average).\nExploring starts : {exploring_starts}\nEpisodes : {n_episodes}."))

            assert env.observation_space.n == policy.n_states, "The number of states in the environment must be equal to the number of states in the policy."
            assert env.action_space.n == policy.n_actions, "The number of actions in the environment must be equal to the number of actions in the policy."
            assert n_episodes > 0, "The number of episodes must be strictly positive."
            assert not exploring_starts or is_state_done is not None, "If exploring_starts is True, done_states must be a set of terminal states."

            # Initialize the action values
            n_states, n_actions = env.observation_space.n, env.action_space.n
            action_values = self.initialize_values( shape = (n_states, n_actions), 
                                                    initial_values = initial_action_values, 
                                                    typical_value = typical_value)

            nbr_qstate_seen_in_episode = dict()
            num_ep = 0
            initial_value = lambda: 0

            while num_ep < n_episodes:
                if verbose >= 1 : print(f"MC Prediction of Q - Episode {num_ep}/{n_episodes}")

                state = env.reset()                    

                t = 0
                qstates_returns = dict()                                     
                qstates_first_time_seen = dict()                        
                qstates_was_seen = set()                                     

                #Run one episode
                done = False
                while not done:
                    # We play an action from the policy and define the qstate = (s,a)
                    if not exploring_starts or t >= 1:
                        action = np.random.choice(policy.n_actions, p=policy.probs[state])
                # If exploring_starts is True, the algorithm will start at a random-non terminal state and play a random action. This can only be used if you can define correctly a is_done_state(state) function AND if you can set the state of your env using "env.state = the_new_state".
                    else:
                        state_temp = np.random.choice(n_states)
                        if not is_state_done(state_temp):
                            env.state = state_temp
                            state = state_temp
                            action = np.random.choice(n_actions)
                        else:
                            action = np.random.choice(policy.n_actions, p=policy.probs[state])
                        
                    qstate = Q_State(state, action)
                    
                    # First time we see the qstate, we initialize the returns, remember the first time we see the qstate and add it to the set of seen qstates
                    if not qstate in qstates_was_seen:
                        qstates_returns[qstate] = initial_value()     
                        qstates_first_time_seen[qstate] = t    
                        qstates_was_seen.add(qstate) 

                    next_state, reward, done, _ = env.step(action)

                    # We add the reward to the returns of each qstate already seen in the episode, the reward being discounted by a factor that depends on the ancienty of the qstate
                    for qstate in qstates_was_seen:
                        qstates_returns[qstate] += reward * (gamma ** (t-qstates_first_time_seen[qstate]))

                    t += 1
                    state = next_state

                    # If s' is a true terminal state, we will still take into account the returns of this terminal state with all actions (which is 0 by convention) in the computation of the action values of this terminal state
                    if done: 
                        for action in range(n_actions):
                            qstate = Q_State(state, action)
                            if not qstate in qstates_was_seen:
                                qstates_returns[qstate] = 0    
                                qstates_first_time_seen[qstate] = t    
                                qstates_was_seen.add(qstate)

                    # Horizon : we artificially set the episode as done if a certain number of steps is reached
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
                    if averaging_method == "cumulative":
                        action_values[state][action] = (N/(N+1)) * action_values[state][action] + (1/(N+1)) * G
                    elif averaging_method == "moving":
                        action_values[state][action] = (1-alpha) * action_values[state][action] + alpha * G
                    else:
                        raise ValueError("Unknown averaging method : {}".format(averaging_method))
                num_ep += 1
            
            if verbose >= 1:
                print(f"MonteCarlo Prediction of Q finished after {num_ep} episodes. Action values found : {action_values}")

            return action_values
    

    def find_action_values_yielding(self,    policy : DiscretePolicyForDiscreteState,
                                    env : gym.Env,
                                    n_episodes : int,
                                    gamma : float = 0.99,
                                    visit_method : str = "first_visit", # "first_visit" or "every_visit"
                                    averaging_method : str = "cumulative", # "cumulative" or "moving"
                                    alpha : float = 0.1, 
                                    horizon : int = float("inf"),
                                    initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                    typical_value = 1,
                                    exploring_starts : bool = False,
                                    is_state_done : Callable = None,
                                    ) -> Iterator:
        """
        Same as find_action_values, but yields the action values at each step.
        """
        assert env.observation_space.n == policy.n_states, "The number of states in the environment must be equal to the number of states in the policy."
        assert env.action_space.n == policy.n_actions, "The number of actions in the environment must be equal to the number of actions in the policy."
        assert n_episodes > 0, "The number of episodes must be strictly positive."
        assert not exploring_starts or is_state_done is not None, "If exploring_starts is True, done_states must be a set of terminal states."

        # Initialize the action values
        n_states, n_actions = env.observation_space.n, env.action_space.n
        action_values = self.initialize_values( shape = (n_states, n_actions), 
                                                initial_values = initial_action_values, 
                                                typical_value = typical_value)

        yield action_values
        nbr_qstate_seen_in_episode = dict()
        num_ep = 0
        initial_value = lambda: 0

        while num_ep < n_episodes:
            yield f"MC Prediction of Q - Episode {num_ep+1}/{n_episodes}"
            state = env.reset()
            t = 0
            qstates_returns = dict()                                     
            qstates_first_time_seen = dict()                        
            qstates_was_seen = set()                                     

            #Run one episode
            done = False
            while not done:
                if not exploring_starts or t >= 1:
                    action = np.random.choice(policy.n_actions, p=policy.probs[state])
                else:
                    state_temp = np.random.choice(n_states)
                    if not is_state_done(state_temp):
                        env.state = state_temp
                        state = state_temp
                        action = np.random.choice(n_actions)
                    else:
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
                
                if done: 
                    for action in range(n_actions):
                        qstate = Q_State(state, action)
                        if not qstate in qstates_was_seen:
                            qstates_returns[qstate] = 0    
                            qstates_first_time_seen[qstate] = t    
                            qstates_was_seen.add(qstate)
                
                if t >= horizon: done = True

            #Update incrementally the state values
            for qstate in qstates_was_seen:
                if qstate in nbr_qstate_seen_in_episode:
                    N = nbr_qstate_seen_in_episode[qstate]
                    nbr_qstate_seen_in_episode[qstate] = N + 1
                else: 
                    N = 0
                    nbr_qstate_seen_in_episode[qstate] = N + 1
                G = qstates_returns[qstate]
                state, action = qstate.observation, qstate.action
                if averaging_method == "cumulative":
                    action_values[state][action] = (N/(N+1)) * action_values[state][action] + (1/(N+1)) * G
                elif averaging_method == "moving":
                    action_values[state][action] = (1-alpha) * action_values[state][action] + alpha * G
                else:
                    raise ValueError("Unknown averaging method : {}".format(averaging_method))
                yield action_values
            num_ep += 1
            

    def find_optimal_policy(self,   env : gym.Env,
                                    gamma : float = 1,
                                    n_iterations : int = 10,
                                    evaluation_episodes : int = 1,
                                    exploration_method : str = "epsilon_greedy", # "epsilon_greedy" or "UCB"
                                    epsilon : Union[float, Scheduler] = 0.1,
                                    visit_method : str = "first_visit", # "first_visit" or "every_visit"
                                    averaging_method : str = "cumulative", # "cumulative" or "moving"
                                    alpha : float = 0.1,
                                    horizon : int = float("inf"),
                                    initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                    typical_value : float = 1,
                                    return_action_values : bool = False,
                                    is_state_done : Callable = None,
                                    verbose : int = 1,
                                    ) -> Union[DiscretePolicyForDiscreteState, Tuple[DiscretePolicyForDiscreteState, np.ndarray]]:
        """This method perform a MonteCarlo method for the Control Problem.
        It learns the optimal policy among the explorative policies for the given env.

        env : the envirronment to learn from
        gamma : the discount factor
        n_iterations : the number of iterations to perform (evaluation + policy improvement)
        evaluation_episodes : the number of episodes to evaluate the policy
        exploration_method : the method to use for exploration ("epsilon_greedy", "UCB", "exploring_starts" or "greedy")
        epsilon : the epsilon parameter for the epsilon-greedy method, can be a scalar or a Scheduler that returns a scalar given a timestep/episode
        visit_method : the method to use for the exploration ("first_visit" or "every_visit")
        averaging_method : the method to use for the averaging ("cumulative" or "moving")
        alpha : the alpha parameter for the moving average method
        horizon : the horizon of the episode (use for non terminal env)
        initial_values : the initial values for the action values ("random", "zeros", "optimistic" or a numpy array)
        typical_value : the typical value for the action values, used for scaling the "random" and "optimistic" value-initialization methods.
        return_action_values : if True, the method returns the action values along  with the policy
        is_state_done : function return whether a state is terminal, used for the "exploring_starts" method
        verbose : the verbosity level
        """

        if verbose >= 1 : 
            print(pretty_announcer(f"Start algorithm Monte Carlo Control ({visit_method}, {averaging_method}-average).\nExploration method used : {exploration_method}\nIterations : {n_iterations}. Evaluation episodes : {evaluation_episodes}"))
        
        assert exploration_method in ["epsilon_greedy", "UCB", "exploring_starts", "greedy"], "Unknown exploration method : {}".format(exploration_method)
        assert n_iterations >= 1, "The number of iterations must be at least 1"
        assert evaluation_episodes >= 1, "The number of evaluation episodes must be at least 1"

        n_states, n_actions = env.observation_space.n, env.action_space.n
        greedy_actions = np.random.choice(np.array([a for a in range(n_actions)]), size = n_states,)
        action_values = self.initialize_values( shape = (n_states, n_actions), 
                                                    initial_values = initial_action_values, 
                                                    typical_value = typical_value)

        n_iter = 0
        while True:
            if verbose >= 1 : print(f"MC Control - Iteration {n_iter}/{n_iterations}")
            num_episode = n_iter * evaluation_episodes

            #Policy improvement
            if exploration_method == "epsilon_greedy":
                eps = epsilon if np.isscalar(epsilon) else epsilon(timestep=None, episode=num_episode)
                probs = np.ones((n_states, n_actions)) * (eps/n_actions)
                probs[range(n_states), greedy_actions] = 1 - eps + (eps/n_actions)
            elif exploration_method == "UCB":
                raise NotImplementedError("UCB exploration method is not implemented yet")
            elif exploration_method == "exploring_starts":
                probs = np.zeros((n_states, n_actions))
                probs[range(n_states), greedy_actions] = 1
                assert is_state_done is not None, "The is_state_done parameter must be set for the exploring starts method"
            elif exploration_method == "greedy":
                probs = np.zeros((n_states, n_actions))
                probs[range(n_states), greedy_actions] = 1
            policy = DiscretePolicyForDiscreteState(probs = probs)

            if n_iter >= n_iterations:
                break   #We put a break condition here rather than one in the While so that we have (explorative_policy(Q), Q) at the end.

            #Evaluate the policy
            action_values = self.find_action_values(policy = policy,
                                                    env = env,
                                                    n_episodes=evaluation_episodes,
                                                    gamma=gamma,
                                                    visit_method=visit_method,
                                                    averaging_method=averaging_method,
                                                    alpha=alpha,
                                                    horizon=horizon,
                                                    initial_action_values=action_values,
                                                    typical_value=typical_value,
                                                    exploring_starts=exploration_method == "exploring_starts",
                                                    is_state_done=is_state_done,
                                                    verbose=1 if verbose>=2 else 0,
                                                    )
            greedy_actions = np.argmax(action_values, axis=1)

            n_iter += 1

        if verbose >= 1:
            print(f"MonteCarlo finished after {n_iter} iterations. Policy's probs found : {policy.probs}")

        if return_action_values:
            return policy, action_values
        else:
            return policy



    def find_optimal_policy_yielding(self,  env : gym.Env,
                                            gamma : float = 1,
                                            n_iterations : int = 10,
                                            evaluation_episodes : int = 1,
                                            exploration_method : str = "epsilon_greedy", # "epsilon_greedy" or "UCB"
                                            epsilon : Union[float, Scheduler] = 0.1,
                                            visit_method : str = "first_visit", # "first_visit" or "every_visit"
                                            averaging_method : str = "cumulative", # "cumulative" or "moving"
                                            alpha : float = 0.1,
                                            horizon : int = float("inf"),
                                            initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                            typical_value : float = 1,
                                            is_state_done : Callable = None,
                                    ) -> Iterator:
        """Same as find_optimal_policy, but yields the action values along with the actions through the training
        """
        
        assert exploration_method in ["epsilon_greedy", "UCB", "exploring_starts", "greedy"], "Unknown exploration method : {}".format(exploration_method)
        assert n_iterations >= 1, "The number of iterations must be at least 1"
        assert evaluation_episodes >= 1, "The number of evaluation episodes must be at least 1"

        n_states, n_actions = env.observation_space.n, env.action_space.n
        greedy_actions = np.random.choice(np.array([a for a in range(n_actions)]), size = n_states,)
        action_values = self.initialize_values( shape = (n_states, n_actions), 
                                                    initial_values = initial_action_values, 
                                                    typical_value = typical_value)
        yield greedy_actions
        yield action_values

        n_iter = 0
        while True:
            yield f"MC Control - Iteration {n_iter}/{n_iterations}"
            num_episode = n_iter * evaluation_episodes

            #Policy improvement : policy is defined from the greedy actions which are defined from Q. The policy obtained can be explorative (eps-greedy, UCB) in order to increase exploration.
            if exploration_method == "epsilon_greedy":
                eps = epsilon if np.isscalar(epsilon) else epsilon(timestep=None, episode=num_episode)
                probs = np.ones((n_states, n_actions)) * (eps/n_actions)
                probs[range(n_states), greedy_actions] = 1 - eps + (eps/n_actions)
            elif exploration_method == "UCB":
                raise NotImplementedError("UCB exploration method is not implemented yet")
            elif exploration_method == "exploring_starts":
                probs = np.zeros((n_states, n_actions))
                probs[range(n_states), greedy_actions] = 1
                assert is_state_done is not None, "The done_states parameter must be set for the exploring starts method"
            elif exploration_method == "greedy":
                probs = np.zeros((n_states, n_actions))
                probs[range(n_states), greedy_actions] = 1
            policy = DiscretePolicyForDiscreteState(probs = probs)

            if n_iter >= n_iterations:
                break   #We put a break condition here rather than one in the While so that we have (explorative_policy(Q), Q) at the end.

            #Evaluate the policy
            for action_values_or_str in self.find_action_values_yielding(policy = policy,
                                                    env = env,
                                                    n_episodes=evaluation_episodes,
                                                    gamma=gamma,
                                                    visit_method=visit_method,
                                                    averaging_method=averaging_method,
                                                    alpha=alpha,
                                                    horizon=horizon,
                                                    initial_action_values=action_values,
                                                    typical_value=typical_value,
                                                    exploring_starts=exploration_method == "exploring_starts",
                                                    is_state_done=is_state_done,):
                yield action_values_or_str
            action_values = action_values_or_str.copy()
            greedy_actions = np.argmax(action_values, axis=1)
            yield greedy_actions
            n_iter += 1




    def initialize_values(self, 
        shape : Tuple,
        initial_values : Union[np.ndarray, str] = "random", # "random", "zeros", "optimistic" or a numpy array
        typical_value : float = 1,
        ) -> np.ndarray: 
        """This method initialize the state or action values and return it.
        shape : the shape of the values
        initial_values : the initial values
        typical_value : the typical value for the action values, used for scaling the "random" and "optimistic" value-initialization methods.
        """


        if type(initial_values) == str:
            if initial_values == "random":
                values = np.random.normal(loc = 0, scale = abs(typical_value), size = shape)
            elif initial_values == "zeros":
                values = np.zeros(shape)
            elif initial_values == "optimistic":         # Optimistic initialization is a trick that consist to overestimate the action values initially. This increase exploration for the greedy algorithms.
                optimistic_value = 2 * typical_value if typical_value > 0 else typical_value / 2
                values = np.ones(shape) * optimistic_value     # An order of the magnitude of the reward is used to initialize optimistically the action values.
            else:
                raise ValueError("The initial action values must be either 'random', 'zeros', 'optimistic' or a numpy array.")
        elif isinstance(initial_values, np.ndarray):
            values = initial_values
        else:
            raise ValueError("The initial action values must be either 'random', 'zeros', 'optimistic' or a numpy array.")
        
        return values

