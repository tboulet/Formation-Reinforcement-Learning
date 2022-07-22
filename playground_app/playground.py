import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from utils import *

def run_rl_algorithm(**config):
    """Run an algorithm and display result on streamlit.
    """
    st.header("Results of training:")
    if config["family"] == "MC":
        config["yield_frequency"] = st.selectbox("Display a frame each... (higher frequency slow down displaying)", ["step", "episode", "iteration"], index=1)
    elif config["family"] == "DP":
        config["yield_frequency"] = st.selectbox("Display a frame each... (higher frequency slow down displaying)", ["step", "iteration", "global_iteration"], index=1)
    elif config["family"] == "TD":
        config["yield_frequency"] = st.selectbox("Display a frame each... (higher frequency slow down displaying)", ["step", "episode"], index=1)
    else:
        raise ValueError("Unknown family: {}".format(config["family"]))

    # Generate for the good problem with the good algo and for the specified config
    algo = config["algo"]
    problem = config["problem"]
    try:
        if problem == "Prediction Problem":
            values_type = config["values_type"]
            if values_type == "State values V":
                datas = algo.find_state_values_yielding(**config)
            elif values_type == "Action values Q":
                datas = algo.find_action_values_yielding(**config)
        elif problem == "Control Problem":
            datas = algo.find_optimal_policy_yielding(**config)
    except AttributeError:
        raise ValueError(f"Algorithm {config['algo_name']} does not work for finding the specified values (if Prediction Problem) or finding the optimal policy. Please change problem or values kind.")

    #Treat this data
    title = "Algo starting"
    title_control = ""
    title_prediction = ""

    num_frame = 0
    frame_titles = dict()
    datas_list = list()
    env = config["env"]
    n_states, n_actions = env.observation_space.n, env.action_space.n
    
    greedy_actions = None
    a, b = config["range_values"]
    y_greedy_actions = 0.9 * b + 0.1 * a
    for data in datas:
        # If the data is a string, modify the title of the next frames.
        if type(data) == str:
            if "Prediction" in data:
                title_prediction = data
            elif "Control" in data:
                title_control = data
            title = title_control + " | " + title_prediction

        # If the data is an array, it can either be a Q(s,a), V(s) or greedy_actions(s). We are building a new frame.
        elif type(data) == np.ndarray:
            # Save the title of the frame. We will apply this title later.
            frame_titles[num_frame] = title
            # Add plot of actions.
            if greedy_actions is not None:
                for state in range(n_states):
                    datas_list.append([num_frame, state, greedy_actions[state], y_greedy_actions])
            # Add plot of Q values
            if len(data.shape) == 2:    # Q values
                for state in range(n_states):
                    for action in range(n_actions):
                        datas_list.append([num_frame, state, action, data[state, action]])
            # Add plot of V values or update greedy_actions depending of the nature of the problem which define the type of 1-dimensionnaly shaped data returned (V or actions).
            elif len(data.shape) == 1:  # 
                if problem == "Prediction Problem": #V values
                    for state in range(n_states):
                        datas_list.append([num_frame, state, -1, data[state]])
                elif problem == "Control Problem": # greedy actions
                    greedy_actions = data

            else: 
                raise ValueError("data must be either a string or a numpy array")
            num_frame += 1
    #Create df and plotly figure : we plot the value in function of the state, and the time-axis is defined as frame. We group data by action to distinguish Q(s,a) for different a.
    df = pd.DataFrame(datas_list, columns=["frame", "state", "action", "values"])
    fig = px.scatter(df,    x = "state", 
                            y = "values", 
                            color = "action", # if values_type == "Action values Q" else None, 
                            animation_frame="frame",
                            range_x=[-1, env.observation_space.n], range_y=config["range_values"])

    #This is for animated title for an animation (only way kekw)
    if len(fig.layout.updatemenus) == 0: raise ValueError("Likely cause of this error : The frequency for frame doesn't make sense for this algorithm, please change.")
    for button in fig.layout.updatemenus[0].buttons:
        button['args'][1]['frame']['redraw'] = True
    for k in range(len(fig.frames)):
        fig.frames[k]['layout'].update(title_text=frame_titles[k])
    
    #Display the figure
    if st.checkbox("Display training"):
        st.plotly_chart(fig)
        if greedy_actions is not None:
            st.write(f"The points that stays at y={y_greedy_actions} represents the greedy action. They are those chosen by the agent in the case of a greedy policy.")