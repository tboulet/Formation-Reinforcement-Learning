import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils import *
from environnements.oceanEnv import transition_probability_ocean, reward_probability_ocean
from DP.dynamicProgramming import PolicyIteration, ValueIteration

n_iterations = 10
S = np.arange(0,11)








### ====================================================================================================================== ###
### ============================================ Policy Iteration ======================================================== ###  
### ====================================================================================================================== ###

algo_PI = PolicyIteration()

### Plot the state values estimated through training
src.policies_and_actions = algo_PI.find_optimal_policy_yielding(transition_probability=transition_probability_ocean,
                                                            reward_probability=reward_probability_ocean,
                                                            gamma=.98,
                                                            n_iterations=8,
                                                            IPE_n_iterations=5,
                                                            IPE_threshold=.05,
                                                            sweep_order="random",
                                                            initial_action_values="random",
                                                            typical_value=-1,
                                                            yield_frequency="step",
                                                              )


results = [e.copy() if type(e) == np.ndarray else e for e in src.policies_and_actions]

bact = 4                                                                   
fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-20, bact + 2)
ax.set_xlabel("s")
title_control = f"DP Control (PI or VI) - Iteration 0"
title_prediction = f"DP Prediction of Q (IPE) - Iteration 0"

# actions, = ax.plot(S, results[1] + bact, ".b", label = "Actions")
actions_join, =ax.plot(S[results[0] == 0], [bact] * (len(S)-np.sum(results[0])), "<g", label = "Actions")
actions_leave, =ax.plot(S[results[0] == 1], [bact] * np.sum(results[0]), ">r")
qvalues_closer, = ax.plot(S, results[1][:, 0], ".g", label = "Estimated Q(s,a) for a = get_closer_to_beach")
qvalues_far,    = ax.plot(S, results[1][:, 1], "xr", label = "Estimated Q(s,a) for a = get_far_from_beach")
ax.legend()

def update(n):
    global title_control, title_prediction
    if n>= len(results):
        ax.set_title("Policy Iteration (ended)")
        return
    data = results[n]
    if type(data) == str:
        if "Control" in data:
            title_control = data
            ax.set_title(title_control + " | " + title_prediction)
        elif "Prediction" in data:
            title_prediction = data
            ax.set_title(title_control + " | " + title_prediction)
    elif type(data) == np.ndarray:
        if len(data.shape) == 1:
            actions_join.set_data(S[data == 0], [bact] * (len(S)-np.sum(data)))
            actions_leave.set_data(S[data == 1], [bact] * np.sum(data))
        elif len(data.shape) == 2:
            qvalues_closer.set_ydata(data[:, 0])
            qvalues_far.set_ydata(data[:, 1])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(2, len(results)),
                        interval = 100)

plt.show()
anim.save("figure/DP/policy_iteration.gif", writer = "ffmpeg", fps = 30)






### ====================================================================================================================== ###
### ============================================ Value Iteration ========================================================= ###  
### ====================================================================================================================== ###

algo_VI = ValueIteration()

### Plot the state values estimated through training
src.policies_and_actions = algo_VI.find_optimal_policy_yielding(transition_probability=transition_probability_ocean,
                                                            reward_probability=reward_probability_ocean,
                                                            gamma=.98,
                                                            n_iterations=15,
                                                            sweep_order="random",
                                                            initial_action_values="random",
                                                            typical_value=-1,
                                                            yield_frequency="step",
                                                              )


results = [e.copy() if type(e) == np.ndarray else e for e in src.policies_and_actions]

bact = 4                                                                   
fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-20, bact + 2)
ax.set_xlabel("s")
title_control = f"DP Control (PI or VI) - Iteration 0"
title_prediction = f"DP Prediction of Q (IPE) - Iteration 0"

# actions, = ax.plot(S, results[1] + bact, ".b", label = "Actions")
actions_join, =ax.plot(S[results[0] == 0], [bact] * (len(S)-np.sum(results[0])), "<g", label = "Actions")
actions_leave, =ax.plot(S[results[0] == 1], [bact] * np.sum(results[0]), ">r")
qvalues_closer, = ax.plot(S, results[1][:, 0], ".g", label = "Estimated Q(s,a) for a = get_closer_to_beach")
qvalues_far,    = ax.plot(S, results[1][:, 1], "xr", label = "Estimated Q(s,a) for a = get_far_from_beach")
ax.legend()

def update(n):
    global title_control, title_prediction
    if n>= len(results):
        ax.set_title("Value Iteration (ended)")
        return
    data = results[n]
    if type(data) == str:
        if "Control" in data:
            title_control = data
            ax.set_title(title_control + " | " + title_prediction)
        elif "Prediction" in data:
            title_prediction = data
            ax.set_title(title_control + " | " + title_prediction)
    elif type(data) == np.ndarray:
        if len(data.shape) == 1:
            actions_join.set_data(S[data == 0], [bact] * (len(S)-np.sum(data)))
            actions_leave.set_data(S[data == 1], [bact] * np.sum(data))
        elif len(data.shape) == 2:
            qvalues_closer.set_ydata(data[:, 0])
            qvalues_far.set_ydata(data[:, 1])

anim = FuncAnimation(   fig = fig,
                        func = update,
                        repeat = True,
                        frames = np.arange(2, len(results)),
                        interval = 100)

anim.save("figure/DP/value_iteration.gif", writer = "ffmpeg", fps = 30)
plt.show()
