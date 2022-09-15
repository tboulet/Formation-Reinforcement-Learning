from utils import *
import gym
from gym import spaces

class OceanEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(11)
        super().__init__()

    def reset(self) -> Observation:
        self.state = 10
        return self.state

    def step(self, action: Action) -> Tuple[Observation, Reward, bool]:
        assert action == 0 or action == 1, "Action must be in {0, 1} for the OceanEnv environnement."
        assert 1 <= self.state <= 10, "The agent should be between 1 and 10 meters when step is called."
        
        # Action has effect on the environment
        if action == 0:
            self.state -= 1
        elif action == 1:
            self.state += 1
            if self.state > 10: self.state = 10
        
        # Compute reward
        reward = -1

        # Check if env is terminated
        done = self.state <= 0

        return self.state, reward, done, {}





    def render(self):
        print(f"Agent is at {self.state} meters of the beach.")



import numpy as np
transition_probability_ocean = np.array([[[0 for _ in range(11)] for _ in range(2)] for _ in range(11)])
reward_probability_ocean = np.array([[0 for _ in range(2)] for _ in range(11)])
env = OceanEnv()
for state in range(1, 11):
    for action in [0, 1]:
        env.state = state
        next_state, reward, done, info = env.step(action)
        transition_probability_ocean[state, action, next_state] = 1               
        reward_probability_ocean[state, action] = reward

if __name__ == "__main__":
    print("Transition probability:", transition_probability_ocean)
    print("Reward probability:", reward_probability_ocean)