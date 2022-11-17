import random

from src.utils import *
import gym
from gym import spaces

class NimEnv(gym.Env):
    num_initial_objects = 21    # Number of objects initially in the pile
    num_rm = 3                  # Maximal number of objects removable each turn

    def __init__(self):
        # Define gym spaces
        self.action_space = spaces.Discrete(self.num_rm)
        self.observation_space = spaces.Discrete(self.num_initial_objects + 1)
        super().__init__()
    
    def reset(self) -> Observation:
        # Define initial state
        self.num_objects = self.num_initial_objects
        return self.num_objects

    def step(self, action) -> Tuple[Observation, Reward, bool]:
        # Check if action is valid (between 0 and num_rm - 1).
        assert self.num_objects > 0, "The game should not be finished when step() is called."
        assert action in range(self.num_rm), "Action must be in {0, 1, ..., self.num_rm - 1} for the nimEnv environnement."
        action += 1
        # Action has effect on the environment
        self.num_objects -= action
        # Compute reward and done
        if self.num_objects <= 0:
            reward = -1
            done = True
            self.num_objects = 0
        else:
            action_opponent = self.opponent_act(state = self.num_objects)
            self.num_objects -= action_opponent
            if self.num_objects <= 0:
                reward = 1
                done = True
                self.num_objects = 0
            else:
                reward = 0
                done = False
        # Return observation, reward, done, and info
        return self.num_objects, reward, done, {}

    def opponent_act(self, state : Observation = None) -> Action:
        # Choose action according to opponent policy (uniformly random)
        action = random.choice(range(self.num_rm)) + 1
        return action

    def render(self, **kwargs):
        print(f"{self.num_objects}/{self.num_initial_objects} objects remaining.")


import numpy as np
n_states = NimEnv.num_initial_objects + 1
n_actions = NimEnv.num_rm
transition_probability_nim = np.array([[[0 for _ in range(n_states)] for _ in range(n_actions)] for _ in range(n_states)], dtype = float)
reward_probability_nim = np.array([[0 for _ in range(n_actions)] for _ in range(n_states)], dtype = float)
env = NimEnv()
for state in range(1, n_states):
    for action in range(n_actions):
        num_objects_removed = action + 1
        num_objects_remaining = state - num_objects_removed

        # Here the agent failed and remove the last object, reaching state 0 and receiving a reward of -1
        if num_objects_remaining <= 0:
            reward_probability_nim[state, action] = -1
            transition_probability_nim[state, action, 0] = 1
        # Here the agent did not remove the last object, and the opponent may remove it.
        else:
            prob = 1 / n_actions
            for action_opponent in range(n_actions):
                num_objects_removed_opponent = action_opponent + 1
                num_objects_remaining_opponent = num_objects_remaining - num_objects_removed_opponent
                # Here the agent did not remove the last object, and the opponent did not remove it.
                if num_objects_remaining_opponent > 0:
                    transition_probability_nim[state, action, num_objects_remaining_opponent] = prob
                # Here the agent did not remove the last object, and the opponent removed the last object.
                # The agent receives a reward of 1 and reaches state 0.
                else:
                    reward_probability_nim[state, action] += prob
                    transition_probability_nim[state, action, 0] += prob

if __name__ == "__main__":
    print("Transition probability for state 5 P(5,a,s'):", transition_probability_nim[5])
    print("Reward probability:", reward_probability_nim)