import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import os
from omlt import OmltBlock
import torch
import itertools
from gurobipy import Model, GRB
from gurobi_ml import add_predictor_constr

# Import from repo files
import transformer_b_flag_cuts as TNN
from trained_transformer.Tmodel import TransformerModel
import helpers.extract_from_pretrained as extract_from_pretrained
from helpers.print_stats import solve_pyomo, solve_gurobipy, save_gurobi_results
import helpers.convert_pyomo as convert_pyomo
from helpers.GUROBI_ML_helper import get_inputs_gurobipy_FNN

# turn off floating-point round-off
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' 

"""
    Solve toy problem with varying setups:
        - change active constraints (automated)
        - change framework + solver (manual)
"""

# Set Up
TESTING = False # fix TNN input for testing (faster solve)
REP = 3 # number of repetitions of each scenario
NAME = "track_traj_keras_enc"
SOLVER = "gurobi"
FRAMEWORK = "gurobipy"
PATH =  ".\\Experiments\Track_k_e\\"

# Store TNN Architecture info
tnn_config = {}
tnn_config["Num Enc"] = 1
tnn_config["Num Dec"] = 0
tnn_config["Num Res"] = 2
tnn_config["Num LN"]  = 2
tnn_config["Num AVGP"] = 0
tnn_config["Num Dense"] = 4
tnn_config["Num ReLu"] = 2

# Define Toy Problem:
# optimal trajectory of a projectile launched with x velocity v1 and y velocity v2

model = pyo.ConcreteModel(name="(TRAJ_TOY)")
config_file = '.\\data\\toy_track_k_enc_config_2.json' 
tnn_config["Config File"] = config_file

# define constants
T_end = 0.5
steps = 19
time = np.linspace(0, T_end, num=steps)

seq_len = 2 # sequence size
time_history = time[0:seq_len]
pred_len = 1

time = time[:seq_len+pred_len]
steps = len(time)
print(steps)

g = 9.81
v_l1 = 0.2
v_l2 = 1.5
dt = time[-1] - time[0]
overlap = 1 # overlap of input data to TNN and output data
bounds_target = (-3,3)

# define sets
model.time = pyo.Set(initialize=time)
model.time_history = pyo.Set(initialize=time_history)

# define parameters
def target_location_rule(M, t):
    return v_l1 * t #+ (np.random.rand(1)/30)
model.loc1 = pyo.Param(model.time, rule=target_location_rule) 

def target_location2_rule(M, t):
    return (v_l2*t) - (0.5 * g * (t**2)) + (np.random.rand(1)/30)
model.loc2 = pyo.Param(model.time, rule=target_location2_rule) 

bounds_target = (-3,3)

# define variables
model.x1 = pyo.Var(model.time, bounds = bounds_target ) # distance path
model.x2 = pyo.Var(model.time, bounds = bounds_target ) # height path

# define initial conditions
model.x1_constr = pyo.Constraint(expr= model.x1[0] == 0) 
model.x2_constr = pyo.Constraint(expr= model.x2[0] == 0) 


# transformer inputs to get expected TNN output
input_x1 =   v_l1 * time  
input_x2 =  (v_l2*time) - (0.5 * g * (time*time))

# ##------ Fix model solution for TESTING ------##
if TESTING:
    model.fixed_loc_constraints = pyo.ConstraintList()
    for i,t in enumerate(model.time_history):
        model.fixed_loc_constraints.add(expr= input_x1[i] == model.x1[t])
        model.fixed_loc_constraints.add(expr= input_x2[i]  == model.x2[t])

# ## --------------------------------##

# Set objective
model.obj = pyo.Objective(
    expr= sum((model.x1[t] - model.loc1[t])**2 + (model.x2[t] - model.loc2[t])**2 for t in model.time), sense=pyo.minimize
)  # -1: maximize, +1: minimize (default)

# load trained transformer
model_path = ".\\trained_transformer\\TNN_traj_enc_2.keras" # dmodel 4, num heads 1, n ence 1, n dec 1, head dim 4, pred_len 2+1 
layer_names, parameters ,_ = extract_from_pretrained.get_learned_parameters(model_path)
tnn_config["Model"] = model_path

# get intermediate results dictionary for optimal input values
input = np.array([[ [x1,x2] for x1,x2 in zip(input_x1, input_x2)]], dtype=np.float32)
layer_outputs_dict = extract_from_pretrained.get_intermediate_values(model_path, input[:, 0:seq_len, :])

# Fix ffn params: add layer not in architecture which is between the two ffns
ffn_1_params = parameters['ffn_1']
parameters['ffn_2']  = {'input_shape':ffn_1_params['input_shape']}#, "input": ffn_1_params['input']}
parameters['ffn_2'] |= {'dense_3':  ffn_1_params['dense_3']}
parameters['ffn_2'] |= {'dense_4':ffn_1_params['dense_4']}

parameters['ffn_1']  = {'input_shape': ffn_1_params['input_shape']}
parameters['ffn_1'] |= {'dense_1': ffn_1_params['dense_1']}
parameters['ffn_1'] |= {'dense_2': ffn_1_params['dense_2']}

# Gather TNN info
tnn_config["TNN Out Expected"] = layer_outputs_dict["dense_4"]

# Get learned parameters
layer = 'mutli_head_attention_1'
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
ACTI["LN_D"] = {"list": ["LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum"]}

ACTI["MHA_I"] = {"list": ["MHA_attn_weight_sum", "MHA_attn_weight"]}
ACTI["MHA_D"] = {"list": [ "MHA_Q", "MHA_K", "MHA_V", "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_attn_score", "MHA_output" ]}
ACTI["MHA_MC"] = {"list": [ "MHA_QK_MC", "MHA_WK_MC"]}

ACTI["RES_ALL"] = {"list":[ "RES_var"]}

ACTI_Groups = 6
LN_ALL = ACTI["LN_I"]["list"] + ACTI["LN_D"]["list"]
MHA_ALL = ACTI["MHA_I"]["list"]  + ACTI["MHA_D"]["list"] + ACTI["MHA_MC"]["list"]
#check all bounds and constraints in the lists
assert(len(ACTI["RES_ALL"]["list"])+ len(MHA_ALL)+ len(LN_ALL) == len(ACTI_LIST)) 

# for each experiment repetition
for r in range(REP):
    # generate all combinations of activating constraint groups
    combinations = list(itertools.product([False, True], repeat=6))
    if TESTING:
        combinations = combinations[-3:]
    else:
        comninations = combinations.reverse()
        
    for c, combi in enumerate(combinations):# for each combination of constraints/bounds
        experiment_name = f"track_k_e_r{r+1}_c{c+1}"
        
        # activate constraints
        ACTI["LN_I"]["act_val"], ACTI["LN_D"]["act_val"], ACTI["MHA_I"]["act_val"] , ACTI["MHA_D"]["act_val"], ACTI["MHA_MC"]["act_val"] , ACTI["RES_ALL"]["act_val"] = combi

        for k, val in ACTI.items():
            for elem in val["list"]:
                activation_dict[elem] = val["act_val"] # set activation dict to new combi
        tnn_config["Activated Bounds/Cuts"] = activation_dict # save act config

        # clone optimization model
        m = model.clone()
    
        #init and activate constraints
        transformer = TNN.Transformer(config_file, m, activation_dict)  
        
        # Define tranformer
        transformer.add_input_var("input_embed", dims=(seq_len, transformer.input_dim), bounds=(-3,3))
        transformer.add_layer_norm( "input_embed", "layer_norm", gamma1, beta1)
        transformer.add_attention( "layer_norm","attention_output", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        transformer.add_residual_connection("input_embed", "attention_output", "residual_1")
        transformer.add_layer_norm( "residual_1", "layer_norm_2", gamma2, beta2)
        nn, input_nn, output_nn = transformer.get_fnn("layer_norm_2", "ffn_1", "ffn_1", (seq_len,2), parameters)
        transformer.add_residual_connection("residual_1", "ffn_1", "residual_2")  
        nn2, input_nn2, output_nn2 = transformer.get_fnn( "residual_2", "ffn_2", "ffn_2", (pred_len+1, 2), parameters)
            
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
            
            if t > m.time_history.last(): # since overlap is 1
                out_index += 1
                print(out_index, t )
                m.tnn_constraints.add(expr= output_nn2[indices[0].at(out_index), indices[1].first()] == m.x1[t])
                m.tnn_constraints.add(expr= output_nn2[indices[0].at(out_index), indices[1].last()]  == m.x2[t])
        

        # # Convert to gurobipy
        gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
        
        ## Add FNN1 to gurobi model
        input_1, output_1 = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
        pred_constr1 = add_predictor_constr(gurobi_model, nn, input_1, output_1)
        
        inputs_2, outputs_2 = get_inputs_gurobipy_FNN(input_nn2, output_nn2, map_var)
        pred_constr2 = add_predictor_constr(gurobi_model, nn2, inputs_2, outputs_2)
        gurobi_model.update()
        #pred_constr.print_stats()
        
        ## Optimizes
        # gurobi_model.setParam('DualReductions',0)
        # gurobi_model.setParam('MIPFocus',1)
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

            plt.figure(figsize=(6, 4))
            #plt.plot(time, input[0,:,:], label= ["1", "2"])
            plt.plot(time, loc2, 'o', label = f'y data')
            plt.plot(time, x2, '--x', label = f'y predicted')
            plt.plot(time, loc1, 'o', label = f'x data')
            plt.plot(time, x1, '--x', label = f'x predicted')
            plt.xlabel("time")
            plt.ylabel("magnitude")
            plt.title(experiment_name)
            plt.legend()
            plt.savefig(PATH+f'\images\{experiment_name}_time.png')
            
            plt.figure(figsize=(6, 4))
            plt.plot(loc1, loc2, 'o', label = f'target trajectory')
            plt.plot(x1, x2, '--x', label = f'cannon ball trajectory')
            plt.title(f'Trajectory')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.savefig(PATH+f'\images\{experiment_name}_traj.png')    
                        
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

        save_gurobi_results(gurobi_model, PATH+experiment_name, experiment_name, r+1, tnn_config)