from utils import *
import gym

class RiverEnv(gym.Env):

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(11)
        super().__init__()

    def reset(self) -> Observation:
        self.state = 10
        return self.state

    def step(self, action: Action) -> Tuple[Observation, Reward, bool]:
        assert action == 0 or action == 1, "Action must be in {0, 1} for the riverEnv environnement."
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
        print(f"L'agent est à {self.state} mètres de la rive.")



import numpy as np
transition_probability = np.array([[[0 for _ in range(11)] for _ in range(2)] for _ in range(11)])
reward_probability = np.array([[0 for _ in range(2)] for _ in range(11)])
env = RiverEnv()
for state in range(1, 11):
    for action in [0, 1]:
        env.state = state
        next_state, reward, done, info = env.step(action)
        transition_probability[state, action, next_state] = 1               
        reward_probability[state, action] = reward

if __name__ == "__main__":
    print("Transition probability:", transition_probability)
    print("Reward probability:", reward_probability)