from utils import *
import gym

class RiverEnv(gym.Env):

    def reset(self) -> Observation:
        self.state = 10
        return 10

    def step(self, action: Action) -> Tuple[Observation, Reward, bool]:
        self.state += action
        if self.state == 0:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        
        return self.state, reward, done

    def render(self):
        print(f"L'agent est à {self.state} mètres de la rive.")
