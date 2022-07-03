import numpy as np

class Policy: pass
class PolicyForDiscreteState(Policy): pass

class DiscretePolicyForDiscreteState(PolicyForDiscreteState):
    def __init__(self, probs : np.ndarray):
        self.probs = probs
        self.n_states, self.n_actions = probs.shape
        """
        Example for 2 state and 4 actions.
        >>> probs = np.array([[0.1, 0.1, 0.7, 0.1], [0.7, 0.1, 0.2, 0.]])
        >>> policy = DiscretePolicyForDiscreteState(probs)
        >>> state = 0
        >>> action = 0
        >>> prob_to_do_action_in_state = policy.get_prob(state, action)
        """
    
    def get_prob(self, state : int, action : int) -> float:
        return self.probs[state, action]

