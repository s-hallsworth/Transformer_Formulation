# External imports
import numpy as np
import os
from gurobipy import GRB
#from gurobi_ml import add_predictor_constr
from gurobi_machinelearning.src.gurobi_ml.add_predictor import add_predictor_constr

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

# Import from repo file
from Examples.reactor.reactor_tnn_exp import reactor_problem, reactor_tnn
from MINLP_tnn.helpers.GUROBI_ML_helper import get_inputs_gurobipy_FFN
from MINLP_tnn.helpers.print_stats import save_gurobi_results
import MINLP_tnn.helpers.convert_pyomo as convert_pyomo
import transformers
import sys
sys.modules['transformers.src.transformers'] = transformers
# cloned transformers from: https://github.com/s-hallsworth/transformers.git

"""
Module for running reactor optimization experiments with Transformer Neural Networks (TNNs).

This script defines and executes experiments for the reactor case study, integrating 
pretrained Transformer models into a Mixed-Integer Nonlinear Programming (MINLP) framework 
using Pyomo and Gurobi.

The main functionalities include:
1. Setting up the reactor optimization problem with constraints and bounds for state variables.
2. Configuring Transformer-based Neural Network (TNN) constraints, including Layer 
   Normalization (LN) and Multi-Head Attention (MHA).
3. Solving the optimization problem using Gurobi and logging the results for further analysis.

Main Steps:
1. **Setup**: Define experimental parameters such as solver framework, repetitions, and 
   TNN configuration.
2. **Constraint Configuration**: Specify active constraints and cuts for the TNN formulation.
3. **Model Definition**: Define the reactor optimization problem and incorporate TNN constraints 
   for both encoder and decoder layers.
4. **Optimization**: Convert the Pyomo model to the Gurobi framework and solve using 
   predefined configurations.
5. **Results Logging**: Log optimization results, including decision variables and TNN hyperparameters.

Key Variables:
- `TESTING`: Enables faster testing by reducing the number of repetitions.
- `ACTI`: Defines groups of constraints for Layer Normalization (LN) and Multi-Head Attention (MHA).
- `combinations`: Specifies combinations of active constraints.
- `tnn_config`: Stores experiment results and hyperparameters.

Usage:
Run the script to perform the experiments as defined in the configurations. Adjust the 
parameters in the "Setup", "Constraint Configuration" and "Solve Settings" sections to customize the experiment.
For ease you can search for the TO DO sections and adjust these parameters.
"""

## Define Set Up
### ------------------ TO DO: CONFIG EXPERIMENTS ----------------------###
TESTING = False # fix TNN input for testing (faster solve)
if TESTING: 
    REP = 1
else:
    REP = 1 # number of repetitions of each scenario
r_offset = 0
NAME = "reactor"
SOLVER = "gurobi"
FRAMEWORK = "gurobipy"
train_tnn_path = ".\\training\\models\\model_TimeSeriesTransformer_final.pth"
config = ".\\training\\models\\config_minlp_tnn.json"
### ---------------------------------------------------------------------###


## Define Transformer Constraint config (# which bounds and cuts to apply to the MINLP TNN formulation)
ACTI_LIST_FULL = [
            "LN_var", "LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum",
                "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight",
            "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC", "MHA_attn_score", "MHA_output", 
            "RES_var", "MHA_softmax_env", "AVG_POOL_var", "embed_var"]
activation_dict = {}
for key in ACTI_LIST_FULL:
    activation_dict[key] = False
### ------------------ TO DO: SET BOUND+CUT CONFIGS ----------------------### 
# define configuartions  
combinations = [ 
    [1 , 0, 1, 1, 0], #2 -- fastest feasibile solution on trajectory problem _/  No Mc
    [1 , 0, 1, 0, 0], #3 -- I only
    [1 , 0, 0, 1, 0], #6 -- fastest optimal solution on trajectory problem _/ LNprop
    [1 , 0, 1, 1, 1], #1 -- all
]
### ----------------------------------------------------------------------###

combinations = [[bool(val) for val in sublist] for sublist in combinations]
ACTI = {}  
ACTI["LN_I"] = {"list": ["LN_var"]}
ACTI["LN_D"] = {"list": ["LN_num", "LN_num_squ", "LN_denom"]}
ACTI["MHA_I"] = {"list": ["MHA_attn_weight_sum", "MHA_attn_weight"]}
ACTI["MHA_D"] = {"list": ["MHA_Q", "MHA_K", "MHA_V", "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_attn_score", "MHA_output" , "RES_var"]}
ACTI["MHA_MC"] = {"list":[ "MHA_QK_MC", "MHA_WK_MC"]}


## RUN EXPERIMENTS:
# Set output directory
PATH =  ".\\experiments\\Reactor__"
if not os.path.exists(PATH):
    os.makedirs(PATH)
    os.makedirs(PATH+"\\Logs")
PATH += "\\"


## Define Reactor Problem
opt_model, parameters,layer_outputs_dict, src, tgt = reactor_problem(train_tnn_path)


## RUN EXPERIMENTS:
tnn_config={} # dict storing additional log information which is later converted to csv

# For each experiment repetition
for r in range(REP):
    for c, combi in enumerate(combinations):
        print("C = ", c+1)    
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

        
        ### ------------------ TO DO: SET UP SOLVE SETTINGS  ----------------------###
        gurobi_model.setParam('LogToConsole', 0)
        gurobi_model.setParam('OutputFlag', 1)
        gurobi_model.setParam('LogFile', PATH+f'Logs\\{experiment_name}.log')
        gurobi_model.setParam('TimeLimit', 43200) #12h
        ### -----------------------------------------------------------------------###
        
        ## Optimise
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
