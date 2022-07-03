import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import *
from riverEnv import transition_probability, reward_probability
from DP.dynamicProgramming import IterativePolicyEvaluation
from policies import DiscretePolicyForDiscreteState

policy_join_beach = DiscretePolicyForDiscreteState(probs = np.array([[1, 0] for _ in range(11)]))

algo_IPE = IterativePolicyEvaluation()

n_iterations = 15
S = np.arange(0,11)

### Plot the state values estimated through training
estimated_state_values_during_training = algo_IPE.find_state_values_yielding(   policy = policy_join_beach,
                                                                                transition_probability = transition_probability,
                                                                                reward_probability = reward_probability,
                                                                                n_iterations = n_iterations,
                                                                                maximal_error = 0.01,
                                                                                gamma=0.98)
VS = [e.copy() for e in estimated_state_values_during_training]
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-n_iterations-2, 1)
ax.set_xlabel("s")
ax.set_ylabel("V(s)")
ax.set_title(f"Policy join_beach : Iteration 0/{len(VS)}")


points, = ax.plot(S, VS[0], ".b", label = "Estimated State Values")
line, = ax.plot(S, -S, "-r", label="True State Values (-s)")
ax.legend()

def update(n):
    ax.set_title(f"Policy join_beach : Iteration {n}/{len(VS)}")
    points.set_ydata(VS[n])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(VS)),
                        interval = 500)

anim.save("figure/v_values_joinBeach_estimated.gif", writer = "ffmpeg", fps = 2)
plt.show()






### Plot the action values estimated through training
estimated_action_values_during_training = algo_IPE.find_action_values_yielding( policy = policy_join_beach,
                                                                                transition_probability = transition_probability,
                                                                                reward_probability = reward_probability,
                                                                                n_iterations = n_iterations,
                                                                                maximal_error = 0.01,
                                                                                gamma = 0.98)
QSA = [e.copy() for e in estimated_action_values_during_training] 
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-n_iterations-2, 1)
ax.set_xlabel("s")
ax.set_ylabel("Q(s, a)")
ax.set_title(f"Policy join_beach : Iteration 0/{len(QSA)}")

points_get_closer, = ax.plot(S, QSA[0][:, 0], ".g", label = "Estimated Q(s,a) for a = get_closer_to_beach")
points_get_far, =    ax.plot(S, QSA[0][:, 1], "xr", label = "Estimated Q(s,a) for a = get_far_from_beach")
ax.legend()

def update(n):
    ax.set_title(f"Policy join_beach : Iteration {n}/{len(QSA)}")
    points_get_closer.set_ydata(QSA[n][:, 0])
    points_get_far.set_ydata(QSA[n][:, 1])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(QSA)),
                        interval = 500)

anim.save("figure/q_values_joinBeach_estimated.gif", writer = "ffmpeg", fps = 2)
plt.show()