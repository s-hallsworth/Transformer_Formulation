# External imports
import pyomo.environ as pyo
import numpy as np
import os
from gurobipy import Model, GRB
#from gurobi_ml import add_predictor_constr
from gurobi_machinelearning.src.gurobi_ml.add_predictor import add_predictor_constr

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

# Import from repo file
from cs_reactor_tnn_exp import reactor_problem, reactor_tnn
from helpers.GUROBI_ML_helper import get_inputs_gurobipy_FFN
from helpers.print_stats import save_gurobi_results
import helpers.convert_pyomo as convert_pyomo
import transformers, sys
sys.modules['transformers.src.transformers'] = transformers
from transformers.models.time_series_transformer.configuration_time_series_transformer import TimeSeriesTransformerConfig
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerForPrediction
# cloned transformers from: https://github.com/s-hallsworth/transformers.git
from gurobi_machinelearning.src.gurobi_ml.add_predictor import add_predictor_constr

"""
Run reactor experiments
"""

# Set Up
TESTING = False # fix TNN input for testing (faster solve)
if TESTING: 
    REP = 1
else:
    REP = 1 # number of repetitions of each scenario
r_offset = 0
NAME = "reactor"
SOLVER = "gurobi"
FRAMEWORK = "gurobipy"
train_tnn_path = ".\\trained_transformer\\case_study\\model_TimeSeriesTransformer_final.pth"
config = ".\\data\\reactor_config_huggingface.json"

# Define Transformer Constraint config:
ACTI_LIST_FULL = [ # Define which constraints and cut config to use
            "LN_var", "LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum",
                "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight",
            "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC", "MHA_attn_score", "MHA_output", 
            "RES_var", "MHA_softmax_env", "AVG_POOL_var", "embed_var"]
activation_dict = {}
for key in ACTI_LIST_FULL:
    activation_dict[key] = False
    
combinations = [ # define configuartions
    [1 , 0, 1, 1, 0], #2 -- fastest feasibile solution _/
    [1 , 0, 1, 0, 0], #3 -- good trade off speed and solve time _/
    #1 , 0, 0, 0, 0, #4 -- smallest opt. gap _/
    #1 , 0, 0, 1, 1, #5_/
    [1 , 0, 0, 1, 0], #6 --- fastest optimal solution _/
    
    [1 , 0, 1, 1, 1], #c4' 1 all
    #[1,  1, 1, 1, 1] #c4
    # 0 , 0, 0, 0, 0  #7 _/
]
combinations = [[bool(val) for val in sublist] for sublist in combinations]
ACTI = {}  
ACTI["LN_I"] = {"list": ["LN_var"]}
ACTI["LN_D"] = {"list": ["LN_num", "LN_num_squ", "LN_denom"]}
ACTI["MHA_I"] = {"list": ["MHA_attn_weight_sum", "MHA_attn_weight"]}
ACTI["MHA_D"] = {"list": ["MHA_Q", "MHA_K", "MHA_V", "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_attn_score", "MHA_output" , "RES_var"]}
ACTI["MHA_MC"] = {"list":[ "MHA_QK_MC", "MHA_WK_MC"]}

# Store TNN Architecture info (enc layer + other)
tnn_config = {}
tnn_config["Num Enc"] = 2
tnn_config["Num Dec"] = 2
tnn_config["Num Res"] = 10
tnn_config["Num LN"]  = 12
tnn_config["Num AVGP"] = 0
tnn_config["Num Dense"] = 15
tnn_config["Num ReLu"] = 0
tnn_config["Num SiLU"] = 4
tnn_config["Num Attn"] = 6


## RUN EXPERIMENTS:
# Set output directory
PATH =  f".\\Experiments\\Reactor__"
if not os.path.exists(PATH):
    os.makedirs(PATH)
    os.makedirs(PATH+"\\Logs")
PATH += "\\"

# Define Reactor Problem
opt_model, parameters,layer_outputs_dict, src, tgt = reactor_problem(train_tnn_path)

# for each experiment repetition
for r in range(REP):
    for c, combi in enumerate(combinations):
        print("C = ", c+1)    
        if c+1 == 2: #skip over I only config (good trade off)
            continue
        experiment_name = f"reactor_r{r+1+r_offset}_c{c+1}"
        # activate constraints
        ACTI["LN_I"]["act_val"], ACTI["LN_D"]["act_val"], ACTI["MHA_I"]["act_val"] , ACTI["MHA_D"]["act_val"], ACTI["MHA_MC"]["act_val"] = combi

        for k, val in ACTI.items():
            for elem in val["list"]:
                activation_dict[elem] = val["act_val"] # set activation dict to new combi
        tnn_config["Activated Bounds/Cuts"] = activation_dict # save act config

        # clone optimization model
        m = opt_model.clone()
        
        # create TNN
        m, ffn_parameter_dict, layer_outputs_dict, transformer = reactor_tnn(opt_model, parameters,layer_outputs_dict,activation_dict, config, src, tgt)
        # tnn_config["TNN Out Expected"] = None
        
        # Convert to gurobipy
        gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
        
        # Add FNNs to gurobi model using GurobiML
        for key, value in ffn_parameter_dict.items():
            nn, input_nn, output_nn = value
            input, output = get_inputs_gurobipy_FFN(input_nn, output_nn, map_var)
            pred_constr = add_predictor_constr(gurobi_model, nn, input, output)

        gurobi_model.update() # update gurobi model with FFN constraints

        ## Optimise
        gurobi_model.setParam('LogToConsole', 0)
        gurobi_model.setParam('OutputFlag', 1)
        gurobi_model.setParam('LogFile', PATH+f'Logs\\{experiment_name}.log')
        gurobi_model.setParam('TimeLimit', 43200) #12h
        gurobi_model.optimize()
        
        if gurobi_model.status == GRB.OPTIMAL:
            optimal_parameters = {}
            for v in gurobi_model.getVars():
                #print(f'var name: {v.varName}, var type {type(v)}')
                if "[" in v.varName:
                    name = v.varname.split("[")[0]
                    if name in optimal_parameters.keys():
                        optimal_parameters[name] += [v.x]
                    else:
                        optimal_parameters[name] = [v.x]
                else:    
                    optimal_parameters[v.varName] = v.x
            
            # save results
            tnn_config["TNN Out"] = np.array(optimal_parameters["x"])  
            print("x", np.array(optimal_parameters["x"]))
            print("x_enc", np.array(optimal_parameters["x"]))
        elif gurobi_model.status == GRB.INFEASIBLE:
            gurobi_model.computeIIS()
            gurobi_model.write("pytorch_model.ilp")
            tnn_config["TNN Out"] = None
            
        # save results
        tnn_config["Enc Seq Len"] = transformer.N
        tnn_config["TNN Model Dims"] = transformer.d_model
        tnn_config["TNN Head Dims"] = transformer.d_k
        tnn_config["TNN Head Size"] = transformer.d_H
        tnn_config["TNN Input Dim"] = transformer.input_dim


        if not TESTING:
            save_gurobi_results(gurobi_model, PATH+experiment_name, experiment_name, r+1+r_offset, tnn_config)
