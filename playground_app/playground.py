import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from utils import *

def run_rl_algorithm(**config):
    """Run an algorithm and display result on streamlit.
    """
    st.subheader("Results of training:")
    config["yield_frequency"] = st.selectbox("Display a frame each... (higher frequency slow down displaying)", ["step", "episode", "iteration"], index=1)

    # Generate for the good problem with the good algo and for the specified config
    algo = config["algo"]
    problem = config["problem"]
    if problem == "Prediction Problem":
        values_type = config["values_type"]
        if values_type == "State values V":
            datas = algo.find_state_values_yielding(**config)
        elif values_type == "Action values Q":
            datas = algo.find_action_values_yielding(**config)
        else:
            raise ValueError("values_type must be either 'State values V' or 'Action values Q'")
    elif problem == "Control Problem":
        datas = algo.find_optimal_policy_yielding(**config)
    else:
        raise ValueError("problem must be either 'Prediction Problem' or 'Control Problem'")

    #Treat this data
    title = "Algo starting"
    num_frame = 0
    frame_titles = dict()
    datas_list = list()
    env = config["env"]
    n_states, n_actions = env.observation_space.n, env.action_space.n
    for data in datas:
        if type(data) == str:
            title = data
        elif type(data) == np.ndarray:
            frame_titles[num_frame] = title

            if len(data.shape) == 2:    # Q values
                for state in range(n_states):
                    for action in range(n_actions):
                        datas_list.append([num_frame, state, action, data[state, action]])
            elif len(data.shape) == 1:  # 
                if problem == "Prediction Problem": #V values
                    for state in range(n_states):
                        datas_list.append([num_frame, state, -1, data[state]])
                elif problem == "Control Problem": # greedy actions
                    for state in range(n_states):
                        datas_list.append([num_frame, state, data[state],  data[state]])
            else: 
                raise ValueError("data must be either a string or a numpy array")
            num_frame += 1
    #Create df and plotly figure : we plot the value in function of the state, and the time-axis is defined as frame. We group data by action to distinguish Q(s,a) for different a.
    df = pd.DataFrame(datas_list, columns=["frame", "state", "action", "values"])
    fig = px.scatter(df,    x = "state", 
                            y = "values", 
                            color = "action" if values_type == "Action values Q" else None, 
                            animation_frame="frame",
                            range_x=[-1, 11], range_y=[-15, 5])
    #This is for animated title for an animation (only way kekw)
    for button in fig.layout.updatemenus[0].buttons:
        button['args'][1]['frame']['redraw'] = True
    for k in range(len(fig.frames)):
        fig.frames[k]['layout'].update(title_text=frame_titles[k])
    
    st.plotly_chart(fig)