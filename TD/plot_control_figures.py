import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import *
from environnements.oceanEnv import OceanEnv
from TD.TDLearning import SARSA
from policies import DiscretePolicyForDiscreteState

algo_SARSA = SARSA()

S = np.arange(0,11)
n_episodes = 50
fps = 30







### ====================================================================================================================== ###
### ============================================ Eps Greedy ============================================================== ###  
### ====================================================================================================================== ###

### Plot the action values estimated through training
policies_and_actions = algo_SARSA.find_optimal_policy_yielding(env = OceanEnv(),
                                                    gamma=.98,
                                                    n_episodes = n_episodes,
                                                    n_steps = float("inf"),
                                                    exploration_method='epsilon_greedy',
                                                    epsilon=.1,
                                                    alpha=.5,
                                                    timelimit=40,
                                                    return_action_values=True,
                                                    initial_action_values="random",
                                                    typical_value=-1,
                                                    is_state_done=lambda state: state == 0,
                                                    yielding_frequency="step",
                                                    )


results = [e.copy() if type(e) == np.ndarray else e for e in policies_and_actions]

bact = 4                                                                   
fig, ax = plt.subplots()
ax.set_xlim(-1, 11)
ax.set_ylim(-20, bact + 2)
ax.set_xlabel("s")
title = "Algorithm starting"

actions_join, =ax.plot(S[results[0] == 0], [bact] * (len(S)-np.sum(results[0])), "<g")
actions_leave, =ax.plot(S[results[0] == 1], [bact] * np.sum(results[0]), ">r")
qvalues_closer, = ax.plot(S, results[1][:, 0], ".g", label = "Q(s,<)")
qvalues_far,    = ax.plot(S, results[1][:, 1], "xr", label = "Q(s,>)")
ax.legend()

def update(n):
    data = results[n]
    if type(data) == str:
        ax.set_title(data)
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

anim.save("figure/TD/SARSA_Control_eps_greedy.gif", writer = "ffmpeg", fps = fps)
plt.show()