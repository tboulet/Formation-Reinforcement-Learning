import gradio as gr
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import numpy as np
matplotlib.use('Agg')

from utils import *
from riverEnv import RiverEnv
from policies import DiscretePolicyForDiscreteState
from MC.monteCarlo import MonteCarlo
from DP.dynamicProgramming import IterativePolicyEvaluation, PolicyIteration


algo_IPE = IterativePolicyEvaluation()
algo_PI = PolicyIteration()
algo_MC = MonteCarlo()


def fn(n_episodes : int):

    # Compute values
    policy = DiscretePolicyForDiscreteState(probs = np.array([[0.8, 0.2] for _ in range(11)]))
    env = RiverEnv()

    estimated_state_values_during_training = algo_MC.find_state_values_yielding( policy = policy,
                                                                        env = env,
                                                                        n_episodes = n_episodes,
                                                                        gamma=0.98,
                                                                        visit_method="first_visit",
                                                                        averaging_method="moving",
                                                                        alpha=0.1,
                                                                        horizon=40,
                                                                        initial_state_values="random",
                                                                        typical_value = -5,
                                                                        exploring_starts=False,
                                                                        is_state_done=lambda state: state == 0,
                                                                        )
    # Create initial plot
    n_episodes = 10
    S = np.arange(0,11)
    y_low_lim = -20

    VS = [e.copy() if type(e) == np.ndarray else e for e in estimated_state_values_during_training]
                                                                         
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 11)
    ax.set_ylim(-13, 1)
    ax.set_xlabel("s")
    ax.set_ylabel("V(s)")

    points, = ax.plot(S, VS[0], ".b", label = "Estimated State Values")
    line, = ax.plot(S, -S, "-r", label="True State Values (-s)")
    ax.legend()

    # Function for updating plot
    def update(n):
        data = VS[n]
        if type(data) == str:
            ax.set_title(f"Policy : {data}")
        elif type(data) == np.ndarray:
            points.set_ydata(VS[n])

    # Create animation, save as .mp4 file and return path
    anim = FuncAnimation(   fig = fig,
                            func = update,
                            repeat = True,
                            frames = np.arange(0, len(VS)),
                            interval = 30)

    anim.save("video.mp4", writer = "ffmpeg", fps = 30)
    return "video.mp4"


if __name__ == "__main__":
    print('ratio gradio')
    iface = gr.Interface(fn=fn, inputs=gr.Number(), outputs=gr.Video())
    iface.launch()