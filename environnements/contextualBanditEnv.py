import random

from src.utils import *
import gym
from gym import spaces

class ContextualBanditEnv(gym.Env):
    n_states = 4
    n_actions = n_states
    is_terminal = True      # True if 1 episode= 1step, False for a non terminal episode

    means = [k for k in range(n_states)]
    stds = [k+1 for k in range(n_states)]

    time_limit = float("inf")   # careful, if not +oo, may lead to strange behavior perhaps for TD

    def __init__(self):
        # Define gym spaces
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states + int(self.is_terminal))
        super().__init__()

    def reset(self) -> Observation:
        # Define initial state
        self.state = self.random_state()
        return self.state

    def step(self, action : Action) -> Tuple[Observation, Reward, bool]:
        # Check if action is valid (between 0 and num_rm - 1).
        assert action in range(self.n_actions), "Action must be in {0, 1, ..., self.n_actions - 1} for the contextualBanditEnv environnement."
        
        # Action has effect on the environment
        k = (self.state - action - 1) % self.n_states
        mean, std = self.means[k], self.stds[k]
        reward = np.random.normal(mean, std)
        if self.is_terminal:
            done = True
            self.state = self.n_states
        else:
            done = False
            self.state = self.random_state()

        return self.state, reward, done, {}

    def random_state(self):
        return random.choice(range(self.n_states))

    def render(self, **kwargs):
        pass

# For non terminal bandit problem
if ContextualBanditEnv.is_terminal:
    transition_probability_CB = np.array([[[0 for _ in range(ContextualBanditEnv.n_states + 1)] for _ in range(ContextualBanditEnv.n_actions)] for _ in range(ContextualBanditEnv.n_states + 1)], dtype = float)
    for state in range(ContextualBanditEnv.n_states):
        for action in range(ContextualBanditEnv.n_actions):
            transition_probability_CB[state][action][ContextualBanditEnv.n_states] = 1
    reward_probability_CB = np.array([[0 for _ in range(ContextualBanditEnv.n_actions)] for _ in range(ContextualBanditEnv.n_states + 1)], dtype = float)
    for state in range(ContextualBanditEnv.n_states):
        for action in range(ContextualBanditEnv.n_actions):
            k = (state - action - 1) % ContextualBanditEnv.n_states
            reward_probability_CB[state][action] = ContextualBanditEnv.means[k]
# For terminal bandit problem (1 episode = 1 step)
else:
    transition_probability_CB = np.array([[[0 for _ in range(ContextualBanditEnv.n_states)] for _ in range(ContextualBanditEnv.n_actions)] for _ in range(ContextualBanditEnv.n_states)], dtype = float)
    reward_probability_CB = np.array([[0 for _ in range(ContextualBanditEnv.n_actions)] for _ in range(ContextualBanditEnv.n_states)], dtype = float)
    for state in range(ContextualBanditEnv.n_states):
        for action in range(ContextualBanditEnv.n_actions):
            k = (state - action - 1) % ContextualBanditEnv.n_states
            reward_probability_CB[state][action] = ContextualBanditEnv.means[k]



if __name__ == "__main__":
    print("Transition probability for state 0 P(0,a,s'):", transition_probability_CB[0])
    print("Reward probability:", reward_probability_CB)