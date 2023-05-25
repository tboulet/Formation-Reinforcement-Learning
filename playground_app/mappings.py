from MC.monteCarlo import MonteCarlo
from DP.dynamicProgramming import IterativePolicyEvaluation, PolicyIteration, ValueIteration
from TD.TDLearning import TD, SARSA

from environnements.oceanEnv import OceanEnv, transition_probability_ocean, reward_probability_ocean
from environnements.nimEnv import NimEnv, transition_probability_nim, reward_probability_nim
from environnements.contextualBanditEnv import ContextualBanditEnv, transition_probability_CB, reward_probability_CB

map_name_to_algo = {"IterativePolicyEvaluation": {  "Algo": IterativePolicyEvaluation, 
                                                    "family": "DP"},
                    "PolicyIteration": {"Algo": PolicyIteration,
                                        "family": "DP"},
                    "ValueIteration":  {"Algo": ValueIteration,
                                        "family": "DP"},
                    "MonteCarlo": {     "Algo": MonteCarlo,
                                        "family": "MC"},
                    "TD(0)": {          "Algo": TD,
                                        "family": "TD"},
                    "SARSA" :           {"Algo" : SARSA,
                                         "family": "TD"},
                    
                        }

map_name_to_env = { "Ocean Env": { "Env" : OceanEnv, 
                                    "model" : (transition_probability_ocean, reward_probability_ocean),
                                    "is_state_done" : lambda state : state == 0,
                                    "range_values" : [-20, 5],
                                    "image_path" : "figure/ocean_env.jpeg",
                                    "description" : "In this environment you need to reach the beach as fast as possible. \
                                    You start in the ocean and you can only move in the 2 directions.  \
                                    The state consist of the distance with the beach and is represented by an integer between 0 and 10  \
                                    (you can't go more far than 10). The reward is -1 at each step and 0 when you reach the beach.  \
                                    The episode ends when you reach the beach. \
                                    ",
                                    },
                    
                    "Nim's Game" :  { "Env" : NimEnv, 
                                    "model" : (transition_probability_nim, reward_probability_nim),
                                    "is_state_done" : lambda state : state <= 0,
                                    "range_values" : [-2, 2],
                                    "image_path" : "figure/nim_env.png",
                                    "description" : "In this game you start with 10 matches and you can remove 1, 2 or 3 matches at each step (those are your actions). The player that removes the last match loses. You play against a random agent. The state consist of the number of matches left and is represented by an integer between 0 and n_matches=25. The reward is 1 if you win, -1 if you lose and 0 if the game is not finished. The episode ends when the game is finished."
                                    }, 

                    "n-Bandit Contextual" :  { "Env" : ContextualBanditEnv, 
                                    "model" : (transition_probability_CB, reward_probability_CB),
                                    "is_state_done" : lambda state : state == -1,
                                    "range_values" : [-1, 4],
                                    "image_path" : "figure/bandit_env.png",
                                    "description" : "In this famous environment, which is a foundation problem of theoretical RL, you have a slot machine with 4 arms. Each arm ill give you a reward following a random law that you don't now. This is contextual because which arm is better depends on the state. In particular here, the expected reward is r(s,a) = (s-a-1)%4 so the optimal action for each state is pi*(s)=s.",
                                    },        
                                
                        }

map_problem_to_algo_names = {   "Prediction Problem" : ["MonteCarlo", "IterativePolicyEvaluation", "TD(0)", "SARSA"],
                                "Control Problem" : ["MonteCarlo", "PolicyIteration", "ValueIteration", "SARSA"],
                                }