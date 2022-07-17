import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import *
from riverEnv import RiverEnv
from TD.TDLearning import TD, SARSA
from policies import DiscretePolicyForDiscreteState


algo_TD = TD()
algo_SARSA = SARSA()

n_episodes = 30
S = np.arange(0,11)
y_low_lim = -20
fps = 30






### ====================================================================================================================== ###
policy_join_beach = DiscretePolicyForDiscreteState(probs = np.array([[1, 0] for _ in range(11)]))
### ====================================================================================================================== ###


### Plot the state values estimated through training
estimated_state_values_during_training = algo_TD.find_state_values_yielding(policy = policy_join_beach,
                                                                            env = RiverEnv(),
                                                                            n_episodes = n_episodes,
                                                                            n_steps = float("inf"),
                                                                            gamma=0.99,
                                                                            alpha=0.5,
                                                                            timelimit=40,
                                                                            initial_state_values="random",
                                                                            typical_value = -5,
                                                                            exploring_starts=False,
                                                                            is_state_done=lambda state: state == 0,

                                                                            yield_frequency="step",
                                                                                )
VS = [e.copy() if type(e) == np.ndarray else e for e in estimated_state_values_during_training]
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-13, 1)
ax.set_xlabel("s")
ax.set_ylabel("V(s)")


points, = ax.plot(S, VS[0], ".b", label = "Estimated State Values")
line, = ax.plot(S, -S, "-r", label="True State Values (-s)")
ax.legend()

def update(n):
    data = VS[n]
    if type(data) == str:
        ax.set_title(f"Policy join_beach : {data}")
    elif type(data) == np.ndarray:
        points.set_ydata(VS[n])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(VS)),
                        interval = 30)

anim.save("figure/TD/v_values_joinBeach_estimated.gif", writer = "ffmpeg", fps = fps)
plt.show()




### Plot the action values estimated through training
estimated_action_values_during_training = algo_SARSA.find_action_values_yielding(  policy = policy_join_beach,
                                                                                env = RiverEnv(),
                                                                                n_episodes = n_episodes,
                                                                                n_steps = float("inf"),
                                                                                gamma=0.99,
                                                                                alpha=0.5,
                                                                                timelimit=40,
                                                                                initial_action_values="random",
                                                                                typical_value = -5,
                                                                                exploring_starts=False,
                                                                                is_state_done=lambda state: state == 0,

                                                                                yield_frequency="step",
                                                                                    )
QSA = [e.copy() if type(e) == np.ndarray else e for e in estimated_action_values_during_training]                                                                         
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-13, 1)
ax.set_xlabel("s")
ax.set_ylabel("Q(s, a)")

points_get_closer, = ax.plot(S, QSA[0][:, 0], ".g", label = "Estimated Q(s,a) for a = get_closer_to_beach")
points_get_far, =    ax.plot(S, QSA[0][:, 1], "xr", label = "Estimated Q(s,a) for a = get_far_from_beach")
ax.legend()

def update(n):
    data = QSA[n]
    if type(data) == str:
        ax.set_title(f"Policy join_beach : {data}")
    elif type(data) == np.ndarray:
        points_get_closer.set_ydata(QSA[n][:, 0])
        points_get_far.set_ydata(QSA[n][:, 1])


anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(QSA)),
                        interval = 30)

anim.save("figure/TD/q_values_joinBeach_estimated.gif", writer = "ffmpeg", fps = fps)
plt.show()




### ====================================================================================================================== ###
policy_leave_beach = DiscretePolicyForDiscreteState(probs = np.array([[0, 1] for _ in range(11)]))
### ====================================================================================================================== ###

### Plot the state values estimated through training
estimated_state_values_during_training = algo_TD.find_state_values_yielding(policy = policy_leave_beach,
                                                                            env = RiverEnv(),
                                                                            n_episodes = 5,
                                                                            n_steps = float("inf"),
                                                                            gamma=0.8,
                                                                            alpha=0.5,
                                                                            timelimit=40,
                                                                            initial_state_values="random",
                                                                            typical_value = -5,
                                                                            exploring_starts=False,
                                                                            is_state_done=lambda state: state == 0,

                                                                            yield_frequency="step",
                                                                                )
VS = [e.copy() if type(e) == np.ndarray else e for e in estimated_state_values_during_training]
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-13, 1)
ax.set_xlabel("s")
ax.set_ylabel("V(s)")


points, = ax.plot(S, VS[0], ".b", label = "Estimated State Values")
ax.legend()

def update(n):
    data = VS[n]
    if type(data) == str:
        ax.set_title(f"Policy leave_beach : {data}")
    elif type(data) == np.ndarray:
        points.set_ydata(VS[n])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(VS)),
                        interval = 30)

anim.save("figure/TD/v_values_leaveBeach_estimated.gif", writer = "ffmpeg", fps = fps)
plt.show()






### Plot the action values estimated through training
estimated_action_values_during_training = algo_SARSA.find_action_values_yielding(  policy = policy_leave_beach,
                                                                                env = RiverEnv(),
                                                                                n_episodes = 5,
                                                                                n_steps = float("inf"),
                                                                                gamma=0.8,
                                                                                alpha=0.5,
                                                                                timelimit=40,
                                                                                initial_action_values="random",
                                                                                typical_value = -5,
                                                                                exploring_starts=False,
                                                                                is_state_done=lambda state: state == 0,

                                                                                yield_frequency="step",
                                                                                    )
QSA = [e.copy() if type(e) == np.ndarray else e for e in estimated_action_values_during_training]                                                                         
                                                                         

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
    data = QSA[n]
    if type(data) == str:
        ax.set_title(f"Policy leave_beach : {data}")
    elif type(data) == np.ndarray:
        points_get_closer.set_ydata(QSA[n][:, 0])
        points_get_far.set_ydata(QSA[n][:, 1])


anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(QSA)),
                        interval = 100)

anim.save("figure/TD/q_values_leaveBeach_estimated.gif", writer = "ffmpeg", fps = fps)
plt.show()







### ====================================================================================================================== ###
policy_swim_randomly = DiscretePolicyForDiscreteState(probs = np.array([[0.8, 0.2] for _ in range(11)]))
### ====================================================================================================================== ###

### Plot the state values estimated through training
estimated_state_values_during_training = algo_TD.find_state_values_yielding(policy = policy_swim_randomly,
                                                                            env = RiverEnv(),
                                                                            n_episodes = n_episodes,
                                                                            n_steps = float("inf"),
                                                                            gamma=0.99,
                                                                            alpha=0.5,
                                                                            timelimit=40,
                                                                            initial_state_values="random",
                                                                            typical_value = -5,
                                                                            exploring_starts=False,
                                                                            is_state_done=lambda state: state == 0,

                                                                            yield_frequency="step",
                                                                                )
VS = [e.copy() if type(e) == np.ndarray else e for e in estimated_state_values_during_training]
                                                                         

fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(y_low_lim, 1)
ax.set_xlabel("s")
ax.set_ylabel("V(s)")


points, = ax.plot(S, VS[0], ".b", label = "Estimated State Values")
ax.legend()

def update(n):
    data = VS[n]
    if type(data) == str:
        ax.set_title(f"Policy swim_randomly : {data}")
    elif type(data) == np.ndarray:
        points.set_ydata(VS[n])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(VS)),
                        interval = 30)

anim.save("figure/TD/v_values_swim_randomly_estimated.gif", writer = "ffmpeg", fps = fps)
plt.show()






### Plot the action values estimated through training
estimated_action_values_during_training = algo_SARSA.find_action_values_yielding(  policy = policy_swim_randomly,
                                                                                env = RiverEnv(),
                                                                                n_episodes = n_episodes,
                                                                                n_steps = float("inf"),
                                                                                gamma=0.99,
                                                                                alpha=0.5,
                                                                                timelimit=40,
                                                                                initial_action_values="random",
                                                                                typical_value = -5,
                                                                                exploring_starts=False,
                                                                                is_state_done=lambda state: state == 0,

                                                                                yield_frequency="step",
                                                                                    )
QSA = [e.copy() if type(e) == np.ndarray else e for e in estimated_action_values_during_training]                                                                         
                                                                         

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
    data = QSA[n]
    if type(data) == str:
        ax.set_title(f"Policy swim_randomly : {data}")
    elif type(data) == np.ndarray:
        points_get_closer.set_ydata(QSA[n][:, 0])
        points_get_far.set_ydata(QSA[n][:, 1])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(0, len(QSA)),
                        interval = 100)

anim.save("figure/TD/q_values_swim_randomly_estimated.gif", writer = "ffmpeg", fps = fps)
plt.show()