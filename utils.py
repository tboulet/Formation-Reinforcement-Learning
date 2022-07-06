from typing import Tuple, Callable, Union

class Observation: pass
class Action: pass
class Reward: pass

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
