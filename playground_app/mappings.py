from MC.monteCarlo import MonteCarlo
from DP.dynamicProgramming import IterativePolicyEvaluation, PolicyIteration, ValueIteration
from TD.TDLearning import TD, SARSA

from env.riverEnv import RiverEnv, transition_probability_river, reward_probability_river
from env.nimEnv import NimEnv, transition_probability_nim, reward_probability_nim
from env.contextualBanditEnv import ContextualBanditEnv, transition_probability_CB, reward_probability_CB

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

map_name_to_env = { "River": { "Env" : RiverEnv, 
                                    "model" : (transition_probability_river, reward_probability_river),
                                    "is_state_done" : lambda state : state == 0,
                                    "range_values" : [-20, 5]
                                    },
                    
                    "Nim" :  { "Env" : NimEnv, 
                                    "model" : (transition_probability_nim, reward_probability_nim),
                                    "is_state_done" : lambda state : state <= 0,
                                    "range_values" : [-2, 2]
                                    }, 

                    "n-Bandit Contextual" :  { "Env" : ContextualBanditEnv, 
                                    "model" : (transition_probability_CB, reward_probability_CB),
                                    "is_state_done" : lambda state : state == -1,
                                    "range_values" : [-1, 4]
                                    },        
                                
                        }

map_problem_to_algo_names = {   "Prediction Problem" : ["MonteCarlo", "IterativePolicyEvaluation", "TD(0)", "SARSA"],
                                "Control Problem" : ["MonteCarlo", "PolicyIteration", "ValueIteration", "SARSA"],
                                }