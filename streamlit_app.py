import streamlit as st
import numpy as np

from utils import *
from policies import DiscretePolicyForDiscreteState

from playground_app.playground import *
from playground_app.mappings import map_name_to_algo, map_name_to_env

st.title("Reinforcement Learning Playground")
config = {}

# Input 1 : env and problem type
st.sidebar.header("Problem:")
env_name = st.sidebar.selectbox("Environment", ["RiverEnv"])
problem =  st.sidebar.selectbox("Problem", ["Prediction Problem", "Control Problem"])

Env = map_name_to_env[env_name]["Env"]
Pssa, Rsa = map_name_to_env[env_name]["model"]
config["env"] = Env()
config["transition_probability"] = Pssa
config["reward_probability"] = Rsa
config["problem"] = problem

if problem == "Prediction Problem":
    # Input 2 : policy to evaluate, value type and algo
    st.header("Algorithm used:")

    algo_name = st.selectbox("Algorithm", ["MonteCarlo", "IterativePolicyEvaluation"])
    Algo = map_name_to_algo[algo_name]["Algo"]

    values_type = st.selectbox("Values to estimate", ["State values V", "Action values Q"])

    env = Env()
    n_actions = env.action_space.n
    action_probs = list()
    st.caption("Policy to evaluate: (will be normalized)")
    for action in range(n_actions):
        action_probs.append(st.slider(f"Action {action}", 0, 100, value=50))
    action_probs = np.array(action_probs) / np.sum(action_probs)
    probs = np.array([action_probs for _ in range(env.observation_space.n)])
    policy = DiscretePolicyForDiscreteState(probs = probs)

    

    config["policy"] = policy
    config["algo"] = Algo()
    config["values_type"] = values_type
else:
    # Input 2 : algo
    pass


# Input 3 : Problem-related parameters
st.header("Parameters:")
col_problem, col_algo = st.columns(2)
with col_problem:
    if problem == "Prediction Problem":
        st.subheader("Prediction Problem:")
        if map_name_to_algo[algo_name]["family"] in ["MC", "TD"]:   # n_episode
            config["n_episodes"] = st.number_input("Number of episodes", value=20)
        elif map_name_to_algo[algo_name]["family"] == "DP":         # n_iterations
            config["n_iterations"] = st.number_input("Number of iterations", value=20)
        config["exploring_starts"] = st.checkbox("Exploring starts", value=False)   # exploring_starts
        if config["exploring_starts"]: config["is_state_done"] = map_name_to_env[env_name]["is_state_done"]

    elif problem == "Control Problem":
        config["n_iterations"] = st.number_input("Number of iterations", value=1)

# Input 4 : Algorithm-related parameters
with col_algo:
    if map_name_to_algo[algo_name]["family"] == "MC":
        st.subheader("Monte Carlo")
        config["visit_method"] = st.selectbox("Visit method", ["first_visit"])
        config["averaging_method"] = st.selectbox("Averaging method", ["cumulative", "moving"])
    if map_name_to_algo[algo_name]["family"] == "DP":
        pass

# Input 4 : Hyperparameters
with st.sidebar:
    st.header("Hyperparameters:")
    config["gamma"] = st.number_input("Discount factor", value=0.95)
    config["alpha"] = st.number_input("Learning rate", value=0.1)
    config["horizon"] = st.slider("Horizon", 0, 100, value=40)
    initial_values = st.selectbox("Initial values", ["zeros", "random", "optimistic"])
    config["initial_state_values"] = initial_values
    config["initial_action_values"] = initial_values
    config["typical_value (in magnitude order)"] = st.number_input("Typical value", value=1)

# Output : compute values and display
run_rl_algorithm(**config)