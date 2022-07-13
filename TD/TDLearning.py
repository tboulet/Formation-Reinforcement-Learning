from typing import Callable, Iterator, Tuple, Union
import numpy as np
import gym

from policies import *
from utils import *

class TD:

    def find_state_values(self, policy : DiscretePolicyForDiscreteState,
                                env : gym.Env,
                                n_episodes : int = float("inf"),
                                n_steps : int = float("inf"),
                                gamma : float = 0.99,
                                alpha : float = 0.1,
                                timelimit : int = float("inf"),
                                initial_state_values : Union[np.ndarray, str] = "random", # "random", "zeros", "optimistic" or a numpy array
                                typical_value : float = 1,
                                exploring_starts : bool = False,
                                is_state_done : Callable = None,
                                verbose : int = 1,
                                    ) -> np.ndarray:
        """This method performs TD(0) for state values, an online on-policy TD Learning algorithm aiming to estimates the state value.
        The algorithm stop after a certain number of episodes or steps done.

        policy : the policy to evaluate
        env : the environment to evaluate the policy on
        n_episodes : the maximal number of episodes of interaction with the env to perform the algorithm
        n_steps : the maximal number of steps of interaction with the env to perform the algorithm
        gamma : the discount factor
        alpha : the learning rate
        timelimit : the number of maximal steps in an episode. After that the episode will be considered done. Use for non terminal env.
        initial_state_values : the initial values of the state values. Can be "random", "zeros", "optimistic" or a numpy array.
        typical_value : the typical value of the state values. Used to initialize the state values if initial_state_values is "random".
        exploring_starts : if True, the algorithm will start at a random-non terminal state. Use IF accessible env. Use for create minimum exploration in the case of deterministic policies.
        is_state_done : a function returning whether a state is terminal. Used if exploring_starts is True for no initialization in the terminal states
        verbose : the verbosity level. 0 for no output, 1 for output.
        """

        assert n_episodes != float("inf") or n_steps != float("inf"), "Either n_episodes or n_steps must be specified."

        if verbose >= 1 : 
            print(pretty_announcer(f"Start algorithm TD(0) for V.\nExploring starts : {exploring_starts}\nFor {n_episodes} episodes or {n_steps} steps."))

        # Initialize the state values
        state_values = initialize_values(   shape = (policy.n_states,),
                                            initial_values = initial_state_values,
                                            typical_value = typical_value)
        num_episode = 0
        num_total_step = 0

        while num_episode < n_episodes and num_total_step < n_steps:
            if verbose >= 1 : print(f"TD(0) Prediction of V - Episode {num_episode}/{n_episodes} - Step {num_total_step}/{n_steps}")
            # Initialize the state
            if exploring_starts:
                state_temp = env.reset()
                if not is_state_done(state_temp):
                    state = state_temp
                    env.state = state
                else:
                    state = env.reset()
            else:
                state = env.reset()

            # Loop through the episode
            t = 0
            done = False
            while not done and t < timelimit and num_total_step < n_steps:
                # Take action, observe the next state and reward
                action = np.random.choice(policy.n_actions, p=policy.probs[state])
                next_state, reward, done, _ = env.step(action)
                # Update the state values online                
                state_values[state] += alpha * (reward + gamma * state_values[next_state] * (1-done) - state_values[state])             
                # timelimit : we artificially set the episode as done if the timelimit is reached
                if t >= timelimit: done = True
                # If done, we additonally learn V(s_next) to be 0.
                if done:
                    state_values[next_state] += alpha * (0 - state_values[next_state])

                state = next_state
                t += 1
                num_total_step += 1
            
            num_episode += 1

        if verbose >= 1:
            print(f"TD(0) Prediction of V finished after {num_episode} episodes and {num_total_step} steps. State values found : {state_values}")

        return state_values
    


    def find_state_values_yielding(self,policy : DiscretePolicyForDiscreteState,
                                        env : gym.Env,
                                        n_episodes : int = float("inf"),
                                        n_steps : int = float("inf"),
                                        gamma : float = 0.99,
                                        alpha : float = 0.1,
                                        timelimit : int = float("inf"),
                                        initial_state_values : Union[np.ndarray, str] = "random", # "random", "zeros", "optimistic" or a numpy array
                                        typical_value : float = 1,
                                        exploring_starts : bool = False,
                                        is_state_done : Callable = None,
                                        yield_frequency : str = "step", # "iteration", "episode", "step"
                                        **kwargs,
                                            ) -> Iterator:
        """
        Same as find_state_values, but yields the state values at each step.

        yield_frequency : "step" or "episode" or "iteration", the frequency at which the state values are yielded.
        """

        assert n_episodes != float("inf") or n_steps != float("inf"), "Either n_episodes or n_steps must be specified."
        assert yield_frequency in ["step", "episode", "iteration"], "yield_frequency must be 'step', 'episode' or 'iteration'"

        # Initialize the state values
        state_values = initialize_values(   shape = (policy.n_states,),
                                            initial_values = initial_state_values,
                                            typical_value = typical_value)
        if yield_frequency != "iterations" : yield state_values
        num_episode = 0
        num_total_step = 0

        while num_episode < n_episodes and num_total_step < n_steps:

            if exploring_starts:
                state_temp = env.reset()
                if not is_state_done(state_temp):
                    state = state_temp
                    env.state = state
                else:
                    state = env.reset()
            else:
                state = env.reset()

            # Loop through the episode
            t = 0
            done = False
            while not done and t < timelimit and num_total_step < n_steps:
                yield f"TD(0) Prediction of V - Episode {num_episode}/{n_episodes} - Step {num_total_step}/{n_steps}"
                # Take action, observe the next state and reward
                action = np.random.choice(policy.n_actions, p=policy.probs[state])
                next_state, reward, done, _ = env.step(action)
                # Update the state values online
                state_values[state] += alpha * (reward + gamma * state_values[next_state] * (1-done) - state_values[state])
                if yield_frequency == "step": yield state_values                 
                # timelimit : we artificially set the episode as done if the timelimit is reached
                if t >= timelimit: done = True
                # If done, we additonally learn V(s_next) to be 0.
                if done:
                    state_values[next_state] += alpha * (0 - state_values[next_state])

                state = next_state
                t += 1
                num_total_step += 1

            if yield_frequency == "episode": yield state_values
            num_episode += 1
        if yield_frequency == "iteration": yield state_values


class SARSA:

    def find_action_values(self,policy : DiscretePolicyForDiscreteState,
                                env : gym.Env,
                                n_episodes : int = float("inf"),
                                n_steps : int = float("inf"),
                                gamma : float = 0.99,
                                alpha : float = 0.1,
                                timelimit : int = float("inf"),
                                initial_action_values : Union[np.ndarray, str] = "random", # "random", "zeros", "optimistic" or a numpy array
                                typical_value : float = 1,
                                exploring_starts : bool = False,
                                is_state_done : Callable = None,
                                verbose : int = 1,
                                    ) -> np.ndarray:
        """This method performs SARSA for action values, an online on-policy TD Learning algorithm aiming to estimates the action value.
        The algorithm stop after a certain number of episodes or steps done.

        policy : the policy to evaluate
        env : the environment to evaluate the policy on
        n_episodes : the maximal number of episodes of interaction with the env to perform the algorithm
        n_steps : the maximal number of steps of interaction with the env to perform the algorithm
        gamma : the discount factor
        alpha : the learning rate
        timelimit : the number of maximal steps in an episode. After that the episode will be considered done. Use for non terminal env.
        initial_action_values : the initial values of the action values. Can be "random", "zeros", "optimistic" or a numpy array.
        typical_value : the typical value of the action values. Used to initialize the action values if initial_action_values is "random".
        exploring_starts : if True, the algorithm will start at a random-non terminal qstate. Use IF accessible env. Use for create minimum exploration in the case of deterministic policies.
        is_state_done : a function returning whether a state is terminal. Used if exploring_starts is True for no initialization in the terminal states
        verbose : the verbosity level. 0 for no output, 1 for output.
        """

        assert n_episodes != float("inf") or n_steps != float("inf"), "Either n_episodes or n_steps must be specified."
        assert not exploring_starts or is_state_done is not None, "is_state_done must be specified if exploring_starts is True."

        if verbose >= 1 : 
            print(pretty_announcer(f"Start algorithm SARSA for Q.\nExploring starts : {exploring_starts}\nFor {n_episodes} episodes or {n_steps} steps."))

        # Initialize the state values
        action_values = initialize_values(  shape = (policy.n_states, policy.n_actions),
                                            initial_values = initial_action_values,
                                            typical_value = typical_value)
        num_episode = 0
        num_total_step = 0
        state = env.reset()

        while num_episode < n_episodes and num_total_step < n_steps:
            if verbose >= 1 : print(f"SARSA Prediction of Q - Episode {num_episode}/{n_episodes} - Step {num_total_step}/{n_steps}")
            # Initialize the qstate
            state = env.reset()
            action = np.random.choice(policy.n_actions, p=policy.probs[state])

            if exploring_starts:  # If exploring starts, we try to choose randomly a qstate (s,a) with s non terminal. This unsure minimum exploration.
                state_temp = np.random.choice(policy.n_states)
                if not is_state_done(state_temp):
                    state = state_temp
                    env.state = state
                    action = np.random.choice(policy.n_actions)

            # Loop through the episode
            t = 0
            done = False
            while not done and t < timelimit and num_total_step < n_steps:
                # Take action, observe the next state and reward, take next action
                next_state, reward, done, _ = env.step(action)
                next_action = np.random.choice(policy.n_actions, p=policy.probs[next_state])
                # Update the action values online
                action_values[state][action] += alpha * (reward + gamma * action_values[next_state][next_action] * (1-done) - action_values[state][action])             
                # timelimit : we artificially set the episode as done if the timelimit is reached
                if t >= timelimit: done = True
                # If done, we additonally learn Q(s_next, a_next) to be 0.
                if done:
                    action_values[next_state][next_action] += alpha * (0 - action_values[next_state][next_action])  

                # Update the qstate
                state = next_state
                action = next_action
                t += 1
                num_total_step += 1
            
            num_episode += 1

        if verbose >= 1:
            print(f"SARSA Prediction of Q finished after {num_episode} episodes and {num_total_step} steps. Action values found : {action_values}")

        return action_values



    def find_action_values_yielding(self,   policy : DiscretePolicyForDiscreteState,
                                            env : gym.Env,
                                            n_episodes : int = float("inf"),
                                            n_steps : int = float("inf"),
                                            gamma : float = 0.99,
                                            alpha : float = 0.1,
                                            timelimit : int = float("inf"),
                                            initial_action_values : Union[np.ndarray, str] = "random", # "random", "zeros", "optimistic" or a numpy array
                                            typical_value : float = 1,
                                            exploring_starts : bool = False,
                                            is_state_done : Callable = None,
                                            yield_frequency : str = "step",
                                            **kwargs,
                                                ) -> Iterator:
        """
        Same as find_action_values, but yields the action values at each step.

        yield_frequency : "step" or "episode" or "iteration", the frequency at which the action values are yielded.
        """

        assert n_episodes != float("inf") or n_steps != float("inf"), "Either n_episodes or n_steps must be specified."
        assert not exploring_starts or is_state_done is not None, "is_state_done must be specified if exploring_starts is True."

        # Initialize the state values
        action_values = initialize_values(  shape = (policy.n_states, policy.n_actions),
                                            initial_values = initial_action_values,
                                            typical_value = typical_value)
        if yield_frequency != "iterations" : yield action_values
        num_episode = 0
        num_total_step = 0
        state = env.reset()

        while num_episode < n_episodes and num_total_step < n_steps:
            # Initialize the qstate
            state = env.reset()
            action = np.random.choice(policy.n_actions, p=policy.probs[state])
            # Loop through the episode
            t = 0
            done = False
            while not done and t < timelimit and num_total_step < n_steps:
                yield f"SARSA Prediction of Q - Episode {num_episode}/{n_episodes} - Step {num_total_step}/{n_steps}"
                # Take action, observe the next state and reward
                if not exploring_starts or t>=1:
                    next_state, reward, done, _ = env.step(action)
                    next_action = np.random.choice(policy.n_actions, p=policy.probs[next_state])
                else:
                    pass
                # Update the action values online
                action_values[state][action] += alpha * (reward + gamma * action_values[next_state][next_action] * (1-done) - action_values[state][action])  
                if yield_frequency == "step": yield action_values           
                # timelimit : we artificially set the episode as done if the timelimit is reached
                if t >= timelimit: done = True
                # If done, we additonally learn Q(s_next, a_next) to be 0.
                if done:
                    action_values[next_state][next_action] += alpha * (0 - action_values[next_state][next_action])  

                state = next_state
                action = next_action
                t += 1
                num_total_step += 1
            
            if yield_frequency == "episode": yield action_values
            num_episode += 1
        if yield_frequency == "iteration": yield action_values


    def find_optimal_policy(self,   env : gym.Env,
                                    gamma : float = 1,
                                    n_episodes : int = float("inf"),
                                    n_steps : int = float("inf"),
                                    exploration_method : str = "epsilon_greedy", # "epsilon_greedy" or "UCB"
                                    epsilon : Union[float, Scheduler] = 0.1,
                                    alpha : float = 0.1,
                                    timelimit : int = float("inf"),
                                    initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                    typical_value : float = 1,
                                    return_action_values : bool = False,
                                    is_state_done : Callable = None,
                                    verbose : int = 1,
                                    ) -> Union[DiscretePolicyForDiscreteState, Tuple[DiscretePolicyForDiscreteState, np.ndarray]]:
        """This method performs SARSA Control, an on-policy online Control algorithm.
        It aims to find the optimal policy (among an explorative subset of every policies).

        env : the envirronment to learn from
        gamma : the discount factor
        n_episodes : the number of episodes to learn from
        exploration_method : the method to use for exploration ("epsilon_greedy", "UCB", "exploring_starts" or "greedy")
        epsilon : the epsilon parameter for the epsilon-greedy method, can be a scalar or a Scheduler that returns a scalar given a timestep/episode
        alpha : the alpha parameter for the moving average method
        timelimit : the timelimit of the episode (use for non terminal env)
        initial_values : the initial values for the action values ("random", "zeros", "optimistic" or a numpy array)
        typical_value : the typical value for the action values, used for scaling the "random" and "optimistic" value-initialization methods.
        return_action_values : if True, the method returns the action values along  with the policy
        is_state_done : function return whether a state is terminal, used for the "exploring_starts" method
        verbose : the verbosity level
        """

        if verbose >= 1 : 
            print(pretty_announcer(f"Start algorithm SARSA Control.\nExploration method used : {exploration_method}\nFor {n_episodes} episodes or {n_steps} steps."))

        assert n_episodes != float("inf") or n_steps != float("inf"), "Either n_episodes or n_steps must be specified."
        assert exploration_method in ["epsilon_greedy", "UCB", "exploring_starts", "greedy"], "Unknown exploration method : {}".format(exploration_method)
        assert n_episodes > 0, "The number of episodes must be positive."

        # Initialize the action values
        n_states, n_actions = env.observation_space.n, env.action_space.n
        action_values = initialize_values(  shape = (n_states, n_actions),
                                            initial_values = initial_action_values,
                                            typical_value = typical_value)

        # Loop through the episodes
        num_episode = 0
        num_total_step = 0
        state = env.reset()

        while num_episode < n_episodes:
            if verbose >= 1 : print(f"SARSA Control - Episode {num_episode}/{n_episodes}")
            # Initialize the qstate
            state = env.reset()
            if exploration_method == "greedy":
                action = np.argmax(action_values[state])
            elif exploration_method == "epsilon_greedy":
                eps = epsilon if np.isscalar(epsilon) else epsilon(timestep=num_total_step, episode=num_episode)
                action = np.random.choice(n_actions) if np.random.random() < eps else np.argmax(action_values[state])
            elif exploration_method == "UCB":
                raise NotImplementedError("UCB exploration method is not implemented yet.")
            elif exploration_method == "exploring_starts":
                assert is_state_done is not None, "is_state_done must be specified if exploring_starts is True."
                state_temp = np.random.choice(n_states)
                if is_state_done(state_temp):
                    action = np.argmax(action_values[state_temp])
                else:
                    action = np.random.choice(n_actions)
                    state = state_temp
                    env.state = state_temp
            else:
                raise NotImplementedError("Unknown exploration method : {}".format(exploration_method))

            # Loop through the episode
            t=0
            done = False
            while not done and t < timelimit and num_total_step < n_steps:
                # Take action, observe the next state and reward, choose next action
                next_state, reward, done, _ = env.step(action)
                if exploration_method == "greedy" or exploration_method == "exploring_starts":
                    next_action = np.argmax(action_values[next_state])
                elif exploration_method == "epsilon_greedy":
                    eps = epsilon if np.isscalar(epsilon) else epsilon(timestep=num_total_step, episode=num_episode)
                    next_action = np.random.choice(n_actions) if np.random.random() < eps else np.argmax(action_values[next_state])
                elif exploration_method == "UCB":
                    raise NotImplementedError("UCB exploration method is not implemented yet.")
                else:
                    raise NotImplementedError("Unknown exploration method : {}".format(exploration_method))
                # Update the action values online
                action_values[state][action] += alpha * (reward + gamma * action_values[next_state][next_action] * (1-done) - action_values[state][action])
                # timelimit : we artificially set the episode as done if the timelimit is reached
                if t >= timelimit: done = True
                # If done, we additonally learn Q(s_next, a_next) to be 0, since by conventon values of terminal states are 0
                if done:
                    action_values[next_state][next_action] += alpha * (0 - action_values[next_state][next_action])  

                # Update the state and action
                state = next_state
                action = next_action
                t += 1
                num_total_step += 1

            num_episode += 1

        if verbose >= 1:
            print(f"SARSA Control finished after {num_episode} episodes and {num_total_step} steps. Action values found : {action_values}")

        probs = np.array([[int(action == np.argmax(action_values[state])) for action in range(n_actions)] for state in range(n_states)])
        optimal_policy = DiscretePolicyForDiscreteState(probs)
        if return_action_values: 
            return optimal_policy, action_values
        else:
            return optimal_policy

    

    def find_optimal_policy_yielding(self,  env : gym.Env,
                                            gamma : float = 1,
                                            n_episodes : int = float("inf"),
                                            n_steps : int = float("inf"),
                                            exploration_method : str = "epsilon_greedy", # "epsilon_greedy" or "UCB"
                                            epsilon : Union[float, Scheduler] = 0.1,
                                            alpha : float = 0.1,
                                            timelimit : int = float("inf"),
                                            initial_action_values : Union[np.ndarray, str] = "random", # "random" or "zeros" or "optimistic" or a numpy array
                                            typical_value : float = 1,
                                            return_action_values : bool = False,
                                            is_state_done : Callable = None,
                                            yielding_frequency : str = "step", # "step" or "episode"
                                            **kwargs,
                                            ) -> Iterator:
        """Same as find_optimal_policy, but yields the action values along with the actions through the training
        
        yield_frequency : "step" or "episode", the frequency at which the state values are yielded.
        """
        assert n_episodes != float("inf") or n_steps != float("inf"), "Either n_episodes or n_steps must be specified."
        assert exploration_method in ["epsilon_greedy", "UCB", "exploring_starts", "greedy"], "Unknown exploration method : {}".format(exploration_method)
        assert n_episodes > 0, "The number of episodes must be positive."

        # Initialize the action values
        n_states, n_actions = env.observation_space.n, env.action_space.n
        action_values = initialize_values(  shape = (n_states, n_actions),
                                            initial_values = initial_action_values,
                                            typical_value = typical_value)
        greedy_actions = np.argmax(action_values, axis=1)
        yield greedy_actions
        yield action_values

        # Loop through the episodes
        num_episode = 0
        num_total_step = 0
        state = env.reset()

        while num_episode < n_episodes:
            yield f"SARSA Control - Episode {num_episode}/{n_episodes} - Step {num_total_step}/{n_steps}"
            # Initialize the qstate
            state = env.reset()
            if exploration_method == "greedy":
                action = np.argmax(action_values[state])
            elif exploration_method == "epsilon_greedy":
                eps = epsilon if np.isscalar(epsilon) else epsilon(timestep=num_total_step, episode=num_episode)
                action = np.random.choice(n_actions) if np.random.random() < eps else np.argmax(action_values[state])
            elif exploration_method == "UCB":
                raise NotImplementedError("UCB exploration method is not implemented yet.")
            elif exploration_method == "exploring_starts":
                assert is_state_done is not None, "is_state_done must be specified if exploring_starts is True."
                state_temp = np.random.choice(n_states)
                if is_state_done(state_temp):
                    action = np.argmax(action_values[state_temp])
                else:
                    action = np.random.choice(n_actions)
                    state = state_temp
                    env.state = state_temp
            else:
                raise NotImplementedError("Unknown exploration method : {}".format(exploration_method))

            # Loop through the episode
            t=0
            done = False
            while not done and t < timelimit and num_total_step < n_steps:
                # Take action, observe the next state and reward, choose next action
                next_state, reward, done, _ = env.step(action)
                if exploration_method == "greedy" or exploration_method == "exploring_starts":
                    next_action = np.argmax(action_values[next_state])
                elif exploration_method == "epsilon_greedy":
                    eps = epsilon if np.isscalar(epsilon) else epsilon(timestep=num_total_step, episode=num_episode)
                    next_action = np.random.choice(n_actions) if np.random.random() < eps else np.argmax(action_values[next_state])
                elif exploration_method == "UCB":
                    raise NotImplementedError("UCB exploration method is not implemented yet.")
                else:
                    raise NotImplementedError("Unknown exploration method : {}".format(exploration_method))
                # Update the action values online
                action_values[state][action] += alpha * (reward + gamma * action_values[next_state][next_action] * (1-done) - action_values[state][action])
                # timelimit : we artificially set the episode as done if the timelimit is reached
                if t >= timelimit: done = True
                # If done, we additonally learn Q(s_next, a_next) to be 0, since by conventon values of terminal states are 0
                if done:
                    action_values[next_state][next_action] += alpha * (0 - action_values[next_state][next_action])  

                # Update the state and action
                state = next_state
                action = next_action
                t += 1
                num_total_step += 1

                if yielding_frequency == "step":
                    greedy_actions = np.argmax(action_values, axis=1)
                    yield action_values
                    yield greedy_actions

            greedy_actions = np.argmax(action_values, axis=1)
            yield action_values
            yield greedy_actions
            num_episode += 1

