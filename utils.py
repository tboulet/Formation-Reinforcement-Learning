from typing import Tuple

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
