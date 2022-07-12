from typing import Tuple, Callable, Union
import numpy as np

class Observation: pass
class Action: pass
class Reward: pass

class RL_algorithm: pass

class Q_State:
    """A Q_State is a tuple of (observation, action)"""
    def __init__(self, observation: Observation, action: Action):
        self.observation = observation
        self.action = action
    def __hash__(self):
        return hash((self.observation, self.action))
    def __eq__(self, other):
        return self.observation == other.observation and self.action == other.action
    def __str__(self):
        return f"({self.observation}, {self.action})"

class Scheduler(Callable):
    """A Scheduler is a callable that given a number of episode or steps, returns the value of an hyper-parameter (learning rate, epsilon) to apply."""
    def __init__(self, unit):
        if not unit in ["episodes", "steps"]:
            raise ValueError("Scheduler unit must be either 'episodes' or 'steps'")
        self.unit = unit
        super().__init__()
    def __call__(self, timestep: Union[int, None], episode : Union[int, None]):
        raise NotImplementedError("Scheduler must be implemented")

def pretty_announcer(string):
    return    "\n==========================================================\n" \
            + string \
            + "\n==========================================================\n"


def initialize_values( 
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