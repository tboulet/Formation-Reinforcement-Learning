import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import *
from riverEnv import RiverEnv
from MC.monteCarlo import MonteCarlo
from policies import DiscretePolicyForDiscreteState


algo_IPE = MonteCarlo()

n_episodes = 10
S = np.arange(0,11)
y_low_lim = -20







### ====================================================================================================================== ###
policy_join_beach = DiscretePolicyForDiscreteState(probs = np.array([[1, 0] for _ in range(11)]))
### ====================================================================================================================== ###


### Plot the state values estimated through training
estimated_state_values_during_training = algo_IPE.find_state_values_yielding(   policy = policy_join_beach,
                                                                                env = RiverEnv(),
                                                                                n_episodes = n_episodes,
                                                                                gamma=0.98)
VS = [e.copy() for e in estimated_state_values_during_training]
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-13, 1)
ax.set_xlabel("s")
ax.set_ylabel("V(s)")
ax.set_title(f"Policy join_beach : Episode 0/{n_episodes}")


points, = ax.plot(S, VS[0], ".b", label = "Estimated State Values")
line, = ax.plot(S, -S, "-r", label="True State Values (-s)")
ax.legend()

def update(n):
    ax.set_title(f"Policy join_beach : Episode ~ {n//(len(VS)//n_episodes)}/{n_episodes}")
    points.set_ydata(VS[n])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(VS)),
                        interval = 100)

anim.save("figure/MC/v_values_joinBeach_estimated.gif", writer = "ffmpeg", fps = 2)
plt.show()





### Plot the action values estimated through training
estimated_action_values_during_training = algo_IPE.find_action_values_yielding( policy = policy_join_beach,
                                                                                env = RiverEnv(),
                                                                                n_episodes = n_episodes,
                                                                                gamma=0.98)
QSA = [e.copy() for e in estimated_action_values_during_training] 
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-13, 1)
ax.set_xlabel("s")
ax.set_ylabel("Q(s, a)")
ax.set_title(f"Policy join_beach : Episode ~ 0/{n_episodes}")

points_get_closer, = ax.plot(S, QSA[0][:, 0], ".g", label = "Estimated Q(s,a) for a = get_closer_to_beach")
points_get_far, =    ax.plot(S, QSA[0][:, 1], "xr", label = "Estimated Q(s,a) for a = get_far_from_beach")
ax.legend()

def update(n):
    ax.set_title(f"Policy join_beach : Episode ~ {n//(len(QSA)//n_episodes)}/{n_episodes}")
    points_get_closer.set_ydata(QSA[n][:, 0])
    points_get_far.set_ydata(QSA[n][:, 1])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(QSA)),
                        interval = 100)

anim.save("figure/MC/q_values_joinBeach_estimated.gif", writer = "ffmpeg", fps = 2)
plt.show()




### ====================================================================================================================== ###
policy_leave_beach = DiscretePolicyForDiscreteState(probs = np.array([[0, 1] for _ in range(11)]))
### ====================================================================================================================== ###

### Plot the state values estimated through training
estimated_state_values_during_training = algo_IPE.find_state_values_yielding(   policy = policy_leave_beach,
                                                                                env = RiverEnv(),
                                                                                n_episodes = n_episodes,
                                                                                gamma=0.8,
                                                                                horizon = 30,)
VS = [e.copy() for e in estimated_state_values_during_training]
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-13, 1)
ax.set_xlabel("s")
ax.set_ylabel("V(s)")
ax.set_title(f"Policy leave_beach : Episode 0/{n_episodes}")


points, = ax.plot(S, VS[0], ".b", label = "Estimated State Values")
ax.legend()

def update(n):
    ax.set_title(f"Policy leave_beach : Episode ~ {n//(len(VS)//n_episodes)}/{n_episodes}")
    points.set_ydata(VS[n])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(VS)),
                        interval = 100)

anim.save("figure/MC/v_values_leaveBeach_estimated.gif", writer = "ffmpeg", fps = 2)
plt.show()






### Plot the action values estimated through training
estimated_action_values_during_training = algo_IPE.find_action_values_yielding( policy = policy_leave_beach,
                                                                                env = RiverEnv(),
                                                                                n_episodes = n_episodes,
                                                                                gamma=0.8,
                                                                                horizon = 30,)
QSA = [e.copy() for e in estimated_action_values_during_training] 
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-13, 1)
ax.set_xlabel("s")
ax.set_ylabel("Q(s, a)")
ax.set_title(f"Policy leave_beach : Iteration 0/{n_episodes}")

points_get_closer, = ax.plot(S, QSA[0][:, 0], ".g", label = "Estimated Q(s,a) for a = get_closer_to_beach")
points_get_far, =    ax.plot(S, QSA[0][:, 1], "xr", label = "Estimated Q(s,a) for a = get_far_from_beach")
ax.legend()

def update(n):
    ax.set_title(f"Policy leave_beach : Episode ~ {n//(len(QSA)//n_episodes)}/{n_episodes}")
    points_get_closer.set_ydata(QSA[n][:, 0])
    points_get_far.set_ydata(QSA[n][:, 1])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(QSA)),
                        interval = 100)

anim.save("figure/MC/q_values_leaveBeach_estimated.gif", writer = "ffmpeg", fps = 2)
plt.show()







### ====================================================================================================================== ###
policy_swim_randomly = DiscretePolicyForDiscreteState(probs = np.array([[0.8, 0.2] for _ in range(11)]))
### ====================================================================================================================== ###

### Plot the state values estimated through training
estimated_state_values_during_training = algo_IPE.find_state_values_yielding(   policy = policy_swim_randomly,
                                                                                env = RiverEnv(),
                                                                                n_episodes = n_episodes,
                                                                                gamma=0.98)
VS = [e.copy() for e in estimated_state_values_during_training]
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(y_low_lim, 1)
ax.set_xlabel("s")
ax.set_ylabel("V(s)")
ax.set_title(f"Policy swim_randomly : Episode 0/{n_episodes}")


points, = ax.plot(S, VS[0], ".b", label = "Estimated State Values")
ax.legend()

def update(n):
    ax.set_title(f"Policy swim_randomly : Episode ~ {n//(len(VS)//n_episodes)}/{n_episodes}")
    points.set_ydata(VS[n])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(VS)),
                        interval = 100)

anim.save("figure/MC/v_values_swim_randomly_estimated.gif", writer = "ffmpeg", fps = 2)
plt.show()






### Plot the action values estimated through training
estimated_action_values_during_training = algo_IPE.find_action_values_yielding( policy = policy_swim_randomly,
                                                                                env = RiverEnv(),
                                                                                n_episodes = n_episodes,
                                                                                gamma=0.98)
QSA = [e.copy() for e in estimated_action_values_during_training] 
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(y_low_lim, 1)
ax.set_xlabel("s")
ax.set_ylabel("Q(s, a)")
ax.set_title(f"Policy swim_randomly : Iteration 0/{n_episodes}")

points_get_closer, = ax.plot(S, QSA[0][:, 0], ".g", label = "Estimated Q(s,a) for a = get_closer_to_beach")
points_get_far, =    ax.plot(S, QSA[0][:, 1], "xr", label = "Estimated Q(s,a) for a = get_far_from_beach")
ax.legend()

def update(n):
    ax.set_title(f"Policy swim_randomly : Episode ~ {n//(len(QSA)//n_episodes)}/{n_episodes}")
    points_get_closer.set_ydata(QSA[n][:, 0])
    points_get_far.set_ydata(QSA[n][:, 1])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(QSA)),
                        interval = 100)

anim.save("figure/MC/q_values_swim_randomly_estimated.gif", writer = "ffmpeg", fps = 2)
plt.show()