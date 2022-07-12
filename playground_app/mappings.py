from MC.monteCarlo import MonteCarlo
from DP.dynamicProgramming import IterativePolicyEvaluation, PolicyIteration, ValueIteration
from riverEnv import RiverEnv, transition_probability, reward_probability

map_name_to_algo = {"IterativePolicyEvaluation": {  "Algo": IterativePolicyEvaluation, 
                                                    "family": "DP"},
                    "PolicyIteration": {"Algo": PolicyIteration,
                                        "family": "DP"},
                    "ValueIteration":  {"Algo": ValueIteration,
                                        "family": "DP"},
                    "MonteCarlo": {     "Algo": MonteCarlo,
                                        "family": "MC"}
                    
                        }

map_name_to_env = { "RiverEnv": { "Env" : RiverEnv, 
                                    "model" : (transition_probability, reward_probability),
                                    "is_state_done" : lambda state : state == 0,
                                    },
                    
                    "NimEnv" : None,        
                                
                        }

map_problem_to_algo_names = {   "Prediction Problem" : ["MonteCarlo", "IterativePolicyEvaluation"],
                                "Control Problem" : ["MonteCarlo", "PolicyIteration", "ValueIteration"],
                                }