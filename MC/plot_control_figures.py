import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils import *
from environnements.oceanEnv import OceanEnv
from MC.monteCarlo import MonteCarlo
from src.policies import DiscretePolicyForDiscreteState

algo_MC = MonteCarlo()

n_iterations = 8
n_iterations_evaluation = 40
S = np.arange(0,11)
fps = 30







### ====================================================================================================================== ###
### ============================================ Eps Greedy ============================================================== ###  
### ====================================================================================================================== ###

### Plot the action values estimated through training
policies_and_actions = algo_MC.find_optimal_policy_yielding(    env = OceanEnv(),
                                                                gamma=.98,
                                                                n_iterations=n_iterations,
                                                                evaluation_episodes=n_iterations_evaluation,
                                                                exploration_method='epsilon_greedy',
                                                                epsilon=.1,
                                                                visit_method="first_visit",
                                                                averaging_method="moving",
                                                                alpha=.1,
                                                                timelimit=40,
                                                                initial_action_values="random",
                                                                typical_value=-10,
                                                                is_state_done=lambda state: state == 0,
                                                              )


results = [e.copy() if type(e) == np.ndarray else e for e in policies_and_actions]

bact = 4                                                                   
fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-20, bact + 2)
ax.set_xlabel("s")
title_control = f"MC Control : 0/{n_iterations}"
title_prediction = f"MC Prediction : 0/{n_iterations_evaluation}"

actions_join, =ax.plot(S[results[0] == 0], [bact] * (len(S)-np.sum(results[0])), "<g")
actions_leave, =ax.plot(S[results[0] == 1], [bact] * np.sum(results[0]), ">r")
qvalues_closer, = ax.plot(S, results[1][:, 0], ".g", label = "Q(s,<)")
qvalues_far,    = ax.plot(S, results[1][:, 1], "xr", label = "Q(s,>)")
ax.legend()

def update(n):
    global title_control, title_prediction
    if n>= len(results):
        ax.set_title("MC Control (ended)")
        return
    data = results[n]
    if type(data) == str:
        if "MC Control" in data:
            title_control = data
            ax.set_title(title_control + " - " + title_prediction)
        elif "MC Prediction" in data:
            title_prediction = data
            ax.set_title(title_control + " - " + title_prediction)
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
                        frames = np.arange(len(results)),
                        interval = 20)
plt.show()
anim.save("figure/MC/MC_Control_eps_greedy.gif", writer = "ffmpeg", fps = 30)
