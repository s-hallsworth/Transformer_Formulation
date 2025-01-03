import pyomo.environ as pyo
import numpy as np
import os
from gurobipy import GRB
from gurobi_ml import add_predictor_constr

# Import from repo files
from MINLP_tnn import transformer as TNN
from MINLP_tnn.helpers import extract_from_pretrained
from MINLP_tnn.helpers.print_stats import save_gurobi_results
import MINLP_tnn.helpers.convert_pyomo as convert_pyomo
from MINLP_tnn.helpers.GUROBI_ML_helper import get_inputs_gurobipy_FFN
from combine_csv import combine

# turn off floating-point round-off
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' 

"""
Module for solving a toy trajectory optimization problem using Transformer Neural Networks (TNNs).

This script defines and executes experiments on a toy trajectory problem, integrating 
pretrained TNNs into a Mixed-Integer Nonlinear Programming (MINLP) framework using Pyomo 
and Gurobi. The experiments explore the effects of varying configurations of active 
constraints and hyperparameters on optimization performance.

Key Functionalities:
1. **Setup**: Define experimental parameters such as sequence length and prediction length for trajectory optimization.
2. **TNN Integration**: Incorporate a pretrained Transformer-based model into the 
   Pyomo optimization framework, including constraints for attention and feed-forward layers.
3. **Constraint Configuration**: Experiment with different combinations of active bounds 
   and cuts for TNN formulation.
4. **Optimization**: Solve the optimization problem using Gurobi and evaluate the results.
5. **Results Analysis**: Log the results for further analysis and optionally combine results 
   into a single CSV file.

Main Steps:
1. **Setup**: Initialize experimental parameters and configure TNN settings, including the 
   hyperparameters and learned parameters of the Transformer model.
2. **Define the Problem**: Formulate the trajectory problem with variables and constraints, 
   and set the objective to minimize the deviation from the target trajectory.
3. **TNN Constraints**: Define input/output relationships for the TNN layers and add 
   Transformer components (attention, normalization, feed-forward layers) as constraints.
4. **Optimization**: Convert the Pyomo model to Gurobi format, solve the optimization problem, 
   and log results such as trajectory variables and TNN outputs.
5. **Result Saving**: Save results for each configuration and optionally combine results into a CSV file.

Key Variables:
- `TESTING`: Enables faster testing by fixing TNN inputs and limiting repetitions.
- `ACTI`: Groups of constraints for Layer Normalization (LN) and Multi-Head Attention (MHA).
- `combinations`: Configurations of active constraints.
- `tnn_config`: Stores experiment metadata, including model parameters and results.
- `PATH`: Directory for saving logs and results.

Usage:
Run the script to solve the trajectory problem with different TNN configurations. Adjust 
the parameters in the "Setup" section and modify the constraint combinations for further experimentation.
"""

# Set Up
TESTING = False # fix TNN input for testing (faster solve)
combine_files = not TESTING
REP = 1 # number of repetitions of each scenario
NAME = "traj_k_e"
SOLVER = "gurobi"
FRAMEWORK = "gurobipy"
exp_name = "traj_k_e"
script_dir = os.path.dirname(os.path.abspath(__file__))
rel_PATH =  f".\\experiments\\{exp_name}"
PATH = os.path.join(script_dir, rel_PATH)
tnn_config = {}



# Define Toy Problem:
# optimal trajectory of a projectile launched with x velocity v1 and y velocity v2
model = pyo.ConcreteModel(name="(TRAJ_TOY)")
rel_dir = 'data\\toy_traj_k_enc_config_2.json' 
hyper_params = os.path.join(script_dir, rel_dir)
tnn_config["Config File"] = hyper_params

# define constants
T_end = 0.5
steps = 19
time = np.linspace(0, T_end, num=steps)

seq_len = 2 # sequence size
time_history = time[0:seq_len]
pred_len = 1

time = time[:seq_len+pred_len]
steps = len(time)

g = 9.81
v_l1 = 0.2
v_l2 = 1.5
dt = time[-1] - time[0]
overlap = 1 # overlap of input data to TNN and output data

# define sets
model.time = pyo.Set(initialize=time)
model.time_history = pyo.Set(initialize=time_history)

# define parameters
def target_location_rule(M, t):
    return v_l1 * t 
model.loc1 = pyo.Param(model.time, rule=target_location_rule) 


def target_location2_rule(M, t):
    np.random.seed(int(v_l2*t*100))
    return (v_l2*t) - (0.5 * g * (t**2)) + ( np.random.uniform(-1,1)/30 )
model.loc2 = pyo.Param(model.time, rule=target_location2_rule) 

bounds_target = (-3,3)

# define variables
model.x1 = pyo.Var(model.time, bounds = bounds_target ) # distance path
model.x2 = pyo.Var(model.time, bounds = bounds_target ) # height path

# define initial conditions
model.x1_constr = pyo.Constraint(expr= model.x1[0] == 0) 
model.x2_constr = pyo.Constraint(expr= model.x2[0] == 0) 


# transformer inputs to get expected TNN output
input_x1 = [0.0, 0.00555569, 0.01111138] # from solution traj_toy.py
input_x2 = [0.0, 0.05100613, 0.09444281]

# Set objective
model.obj = pyo.Objective(
    expr= sum((model.x1[t] - model.loc1[t])**2 + (model.x2[t] - model.loc2[t])**2 for t in model.time), sense=pyo.minimize
)  # -1: maximize, +1: minimize (default)

# load trained transformer
model_path = "training\\models\\TNN_traj_enc_2.keras" # dmodel 4, num heads 1, n ence 1, n dec 1, head dim 4, pred_len 2+1 
model_PATH = os.path.join(script_dir, model_path)
layer_names, parameters , tnn_model = extract_from_pretrained.get_learned_parameters(model_PATH)


tnn_config["Model"] = model_path

# get intermediate results dictionary for optimal input values
input = np.array([[ [x1,x2] for x1,x2 in zip(input_x1, input_x2)]], dtype=np.float32)
layer_outputs_dict = extract_from_pretrained.get_intermediate_values(model_PATH, input[:, 0:seq_len, :])
FFN_out = np.array(layer_outputs_dict["dense_4"])[0].transpose(1,0)

# ##------ Fix model solution for TESTING ------##
if TESTING:
    REP = 1
    model.fixed_loc_constraints = pyo.ConstraintList()
    for i,t in enumerate(model.time):
        if t <= model.time_history.last():
            model.fixed_loc_constraints.add(expr= input_x1[i] == model.x1[t])
            model.fixed_loc_constraints.add(expr= input_x2[i]  == model.x2[t])

# ## --------------------------------##

# Fix ffn params: add layer not in architecture which is between the two ffns
ffn_1_params = parameters['ffn_1']
parameters['ffn_2']  = {'input_shape':ffn_1_params['input_shape']}
parameters['ffn_2'] |= {'dense_3':  ffn_1_params['dense_3']}
parameters['ffn_2'] |= {'dense_4':ffn_1_params['dense_4']}

parameters['ffn_1']  = {'input_shape': ffn_1_params['input_shape']}
parameters['ffn_1'] |= {'dense_1': ffn_1_params['dense_1']}
parameters['ffn_1'] |= {'dense_2': ffn_1_params['dense_2']}

# Gather TNN info
tnn_config["TNN Out Expected"] = layer_outputs_dict["dense_4"]

# Get learned parameters
layer = 'multi_head_attention_1'
W_q = parameters[layer,'W_q']
W_k = parameters[layer,'W_k']
W_v = parameters[layer,'W_v']
W_o = parameters[layer,'W_o']
try:
    b_q = parameters[layer,'b_q']
    b_k = parameters[layer,'b_k']
    b_v = parameters[layer,'b_v']
    b_o = parameters[layer,'b_o']
except: # no bias values found
        b_q = 0
        b_k = 0
        b_v = 0
        b_o = 0

gamma1 = parameters['layer_normalization_1', 'gamma']
beta1  = parameters['layer_normalization_1', 'beta']

gamma2 = parameters['layer_normalization_2', 'gamma']
beta2  = parameters['layer_normalization_2', 'beta']

# initially all constraints deactivated
ACTI_LIST = [
            "LN_var", "LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum",
             "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight",
            "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC", "MHA_attn_score", "MHA_output", 
            "RES_var"] #names of bounds and cuts to activate
                # "MHA_softmax_env"<- removed from list: should be dynamic
                # "AVG_POOL_var" <- no avg pool
                #  "embed_var" <- no embed
ACTI_LIST_FULL = [
            "LN_var", "LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum",
             "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight",
            "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC", "MHA_attn_score", "MHA_output", 
            "RES_var", "MHA_softmax_env", "AVG_POOL_var", "embed_var"]

activation_dict = {}
for key in ACTI_LIST_FULL:
    activation_dict[key] = False

ACTI = {}  
ACTI["LN_I"] = {"list": ["LN_var"]}
ACTI["LN_D"] = {"list": ["LN_num", "LN_num_squ", "LN_denom"]}
ACTI["MHA_I"] = {"list": ["MHA_attn_weight_sum", "MHA_attn_weight"]}
ACTI["MHA_D"] = {"list": ["MHA_Q", "MHA_K", "MHA_V", "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_attn_score", "MHA_output" , "RES_var"]}
ACTI["MHA_MC"] = {"list":[ "MHA_QK_MC", "MHA_WK_MC"]}

combinations = [
    [1 , 0, 1, 1, 1],
    [1 , 0, 1, 1, 0],
    [1 , 0, 1, 0, 0],
    [1 , 0, 0, 0, 0],
    [1 , 0, 0, 1, 1],
    [1 , 0, 0, 1, 0], 
    [0 , 0, 0, 0, 0],
]
combinations = [[bool(val) for val in sublist] for sublist in combinations]


# for each experiment repetition
for r in range(REP):
        
    for c, combi in enumerate(combinations):# for each combination of constraints/bounds
        experiment_name = f"{exp_name}_r{r+1}_c{c+1}"
        # activate constraints
        ACTI["LN_I"]["act_val"], ACTI["LN_D"]["act_val"], ACTI["MHA_I"]["act_val"] , ACTI["MHA_D"]["act_val"], ACTI["MHA_MC"]["act_val"] = combi

        for k, val in ACTI.items():
            for elem in val["list"]:
                activation_dict[elem] = val["act_val"] # set activation dict to new combi
        tnn_config["Activated Bounds/Cuts"] = activation_dict # save act config

        # clone optimization model
        m = model.clone()
        
    
        #init and activate constraints
        transformer = TNN.Transformer(hyper_params, m, activation_dict)  
        
        # Define tranformer
        transformer.add_input_var("input_embed", dims=(seq_len, transformer.input_dim), bounds=(None,None))
        transformer.add_layer_norm( "input_embed", "layer_norm", gamma1, beta1)
        transformer.add_attention( "layer_norm","attention_output", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        transformer.add_residual_connection("input_embed", "attention_output", "residual_1")
        transformer.add_layer_norm( "residual_1", "layer_norm_2", gamma2, beta2)
        nn, input_nn, output_nn = transformer.get_ffn("layer_norm_2", "ffn_1", "ffn_1", (seq_len,2), parameters)
        transformer.add_residual_connection("residual_1", "ffn_1", "residual_2")  
        nn2, input_nn2, output_nn2 = transformer.get_ffn( "residual_2", "ffn_2", "ffn_2", (pred_len+1, 2), parameters)
            
        # add constraints to trained TNN input
        m.tnn_constraints = pyo.ConstraintList()
        indices = []
        for set in str(transformer.M.input_embed.index_set()).split("*"):
            indices.append( getattr(m, set) )
        for tnn_index, index in zip(indices[0], m.time_history):
            m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].first()]== m.x1[index])
            m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].last()] == m.x2[index]) 
            
        # add constraints to trained TNN output
        indices = []
        for set in str(output_nn2.index_set()).split("*"): 
            indices.append( getattr(m, set) )
        out_index = 0
        for t_index, t in enumerate(m.time):
            index = t_index + 1 # 1 indexing
            
            if t > m.time_history.last(): 
                out_index += 2
                m.tnn_constraints.add(expr= output_nn2[indices[0].at(out_index), indices[1].first()] == m.x1[t])
                m.tnn_constraints.add(expr= output_nn2[indices[0].at(out_index), indices[1].last()]  == m.x2[t])
               
        # # Convert to gurobipy
        gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
        
        ## Add FNN1 to gurobi model
        input_1, output_1 = get_inputs_gurobipy_FFN(input_nn, output_nn, map_var)
        pred_constr1 = add_predictor_constr(gurobi_model, nn, input_1, output_1)
        
        inputs_2, outputs_2 = get_inputs_gurobipy_FFN(input_nn2, output_nn2, map_var)
        pred_constr2 = add_predictor_constr(gurobi_model, nn2, inputs_2, outputs_2)
        gurobi_model.update()

        # Set output directory
        print(PATH)
        if not os.path.exists(PATH): # Create directory if does not exist
            os.makedirs(PATH)
            os.makedirs(PATH+"\\logs")
        PATH += "\\"
        
        ## Optimize
        gurobi_model.setParam('LogFile', PATH+f'logs\\{experiment_name}.log')
        gurobi_model.setParam('TimeLimit', 21600) # 6h
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
            tnn_config["TNN Out"] = np.array(layer_outputs_dict["dense_4"])[0]   
            
            x1 = np.array(optimal_parameters['x1'])
            x2 = np.array(optimal_parameters['x2'])
            loc1 = np.array([v for k,v in model.loc1.items()])
            loc2 = np.array([v for k,v in model.loc2.items()])       
        else:
            tnn_config["TNN Out"] = None
        # save results
        tnn_config["Enc Seq Len"] = transformer.N
        tnn_config["Pred Len"] = pred_len
        tnn_config["Overlap"] = overlap
        tnn_config["TNN Model Dims"] = transformer.d_model
        tnn_config["TNN Head Dims"] = transformer.d_k
        tnn_config["TNN Head Size"] = transformer.d_H
        tnn_config["TNN Input Dim"] = transformer.input_dim
        
        if c==1:
            tnn_config["Config"] = "fixed tnn inputs"
        elif c==2:
            tnn_config["Config"] = "fixed tnn inputs with all (1)"
        elif c == 0:
            tnn_config["Config"] = "LN_prop (6)"
            
        elif c == 3:
            tnn_config["Config"] = "fixed tnn inputs 2"
        else:
            tnn_config["Config"] = "only initial tnn input"

        if not TESTING:
            save_gurobi_results(gurobi_model, PATH+experiment_name, experiment_name, r+1+1, tnn_config)

if combine_files:            
    output_filename = f'{exp_name}.csv'
    combine(PATH, output_filename)