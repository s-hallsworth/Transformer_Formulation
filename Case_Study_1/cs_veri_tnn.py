"""
This code an adaptation of code from: https://github.com/cog-imperial/PartitionedFormulations_NN/blob/main/src/optimalAdversary.py
created by:
    Calvin Tsay (tsaycal) - Imperial College London
    Jan Kronqvist (jkronqvi) - KTH Royal Institute of Technology
    Alexander Thebelt (ThebTron) - Imperial College London
    Ruth Misener (rmisener) - Imperial College London

For further details see the git repo or refer to the related article "Partition-based formulations for mixed-integer optimization of trained ReLU neural networks" Tsay et al. 2021

This file contains implementations of the Optimal Adversary problem described in
Section 4.1 of the manuscript. The file can be run directly, with the following parameters.
Parameters can be changed on lines 46-60.
    
    Parameters:
        problemNo (int): index of problem to be solved, can be in range(100)
        epsilon (real): maximum l1-norm defining perturbations
"""

import numpy as np
import gurobipy as gb
import pyomo.environ as pyo
import numpy as np
from pyomo import dae
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import unittest
import os
from omlt import OmltBlock
import torch
from helpers.print_stats import solve_pyomo, solve_gurobipy
import helpers.convert_pyomo as convert_pyomo
from gurobipy import Model, GRB
from gurobi_ml import add_predictor_constr
from helpers.GUROBI_ML_helper import get_inputs_gurobipy_FNN

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

# Import from repo file
import transformer_b_flag as TNN
from trained_transformer.Tmodel import TransformerModel
import helpers.extract_from_pretrained as extract_from_pretrained

TESTING = True # TESTING

problemNo = 0 # image to select from MNIST dataset
epsilon = 0 
nLayers = 3
instances = np.load(r'.\data\mnist2x50instances.npz')
inputimage = instances['images'][problemNo] # flattened image
labels = instances['labels'][problemNo] # [true label, adversary label]
image_size_flat = instances['w1'].shape[1]
classification_labels = 10
channels = 1
image_size=28
patch_size=4 
num_classes=10
channels=1
input = torch.as_tensor(inputimage).float().reshape(1, channels, image_size, image_size) #image_size * image_size image

tgt_dict = {}
for p1 in range(image_size):
    for p2 in range(image_size):
        tgt_dict[p1, p2] = input[0,0,p1,p2].tolist()

# Create pyomo model
model = pyo.ConcreteModel(name="(TOY_TRANFORMER)")

# Define parameters and sets
model.labels = pyo.Set(initialize=labels)
model.channel_dim = pyo.Set(initialize=range(channels))
model.image_dim = pyo.Set(initialize=range(image_size))
model.out_labels_dim = pyo.Set(initialize=range(classification_labels))

model.eps = pyo.Param(initialize=epsilon)
model.target_image = pyo.Param(model.image_dim, model.image_dim, initialize=tgt_dict)

# Define variables
model.purturb_image = pyo.Var(model.image_dim, model.image_dim, bounds=(0,1))
model.purturb = pyo.Var(model.image_dim, model.image_dim, bounds=(0, model.eps))

# Add constraints to purturbed image:
model.purturb_constraints = pyo.ConstraintList()
for i in model.purturb_image.index_set():
    model.purturb_constraints.add(expr= model.purturb_image[i] <= model.target_image[i] + model.eps) # # purturb image <= min(max purturb,1)
    model.purturb_constraints.add(expr= model.purturb_image[i] >= model.target_image[i] - model.eps) # # purturb image >=  max(min purturb,0)
    
    #purturb >= to the abs different between x and x'
    model.purturb_constraints.add(expr= model.purturb[i] >= model.purturb_image[i] - model.target_image[i])
    model.purturb_constraints.add(expr= model.purturb[i] >= model.target_image[i] - model.purturb_image[i])

# total purturb at each pixel <= epsilon   
model.purturb_constraints.add(expr= sum(model.purturb[i] for i in model.purturb.index_set()) <= model.eps) 
   
   
# Load transformer
from vit_TNN import *
file_name = "vit_6_1_6_12"
tnn_path = f".\\trained_transformer\\verification\\{file_name}.pt" 
device = 'cpu'
config_params = file_name.split('_')
dim= int(config_params[1])
depth= int(config_params[2])
heads= int(config_params[3])
mlp_dim= int(config_params[4])
head_size = int(dim/heads)
config_list = [channels, dim, head_size , heads, image_size*image_size, 1e-5]
tnn_model = torch.load(tnn_path, map_location=device)


# Define which constraints and cut config to use
ACTI_LIST_FULL = [
            "LN_var", "LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum",
             "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight",
            "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC", "MHA_attn_score", "MHA_output", 
            "RES_var", "MHA_softmax_env", "AVG_POOL_var", "embed_var"]
activation_dict = {}
for key in ACTI_LIST_FULL:
    activation_dict[key] = False

combinations = [
# 1 , 0, 1, 1, 1, #1 all
# 1 , 0, 1, 1, 0 #2 -- fastest feasibile solution
#1 , 0, 1, 0, 0, #3 -- good trade off speed and solve time
# [1 , 0, 0, 0, 0], #4 -- smallest opt. gap
# [1 , 0, 0, 1, 1], #5
 1 , 0, 0, 1, 0, #6 --- fastest optimal solution
# [0 , 0, 0, 0, 0]  #7
]
combinations = [bool(val) for val in combinations]

ACTI = {}  
ACTI["LN_I"] = {"list": ["LN_var"]}
ACTI["LN_D"] = {"list": ["LN_num", "LN_num_squ", "LN_denom"]}
ACTI["MHA_I"] = {"list": ["MHA_attn_weight_sum", "MHA_attn_weight"]}
ACTI["MHA_D"] = {"list": ["MHA_Q", "MHA_K", "MHA_V", "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_attn_score", "MHA_output" , "RES_var"]}
ACTI["MHA_MC"] = {"list":[ "MHA_QK_MC", "MHA_WK_MC"]}
ACTI["LN_I"]["act_val"], ACTI["LN_D"]["act_val"], ACTI["MHA_I"]["act_val"] , ACTI["MHA_D"]["act_val"], ACTI["MHA_MC"]["act_val"] = combinations

for k, val in ACTI.items():
    for elem in val["list"]:
        activation_dict[elem] = val["act_val"] # set activation dict to new combi
 
# TESTING ----   
# Define formulated transformer
transformer = TNN.Transformer( config_list, model, activation_dict)
layer_names, parameters, _, layer_outputs_dict = extract_from_pretrained.get_torchViT_learned_parameters(tnn_model, input, heads)
if TESTING:
    plt.imshow(input.squeeze(0).squeeze(0), cmap='gray')
    plt.show()
    
# Add Sequential 1 x28 x 18 mult 18 x patch size
layer = "linear_1"
num_patch_dim = int(image_size_flat/(patch_size*patch_size))
model.num_patch_dim = pyo.Set(initialize=range(num_patch_dim ))
model.patch_dim = pyo.Set(initialize=range(patch_size*patch_size))
model.embed_dim = pyo.Set(initialize=range(dim))

W_emb = parameters[layer,'W']
b_emb = parameters[layer,'b']
model.patch_input= pyo.Var(model.num_patch_dim, model.patch_dim)

#--------
model.rearrange_constraints = pyo.ConstraintList()
for i in range(0, image_size, patch_size):  
    for j in range(0, image_size, patch_size):  
        patch_i = i // patch_size  
        patch_j = j // patch_size  
        patch_index = patch_i * (image_size // patch_size) + patch_j  
        
        for pi in range(patch_size):
            for pj in range(patch_size):
                pos_i = patch_index  
                pos_j = pi * patch_size + pj 

                model.rearrange_constraints.add(expr= model.patch_input[pos_i, pos_j] == model.purturb_image[i + pi, j + pj])
out = transformer.embed_input( "patch_input" , "embed_input", model.embed_dim, W_emb, b_emb)

#  CLS tokens
cls_token = parameters['cls_token']
model.cls_dim= pyo.Set(initialize=range(num_patch_dim + 1))
model.cls = pyo.Var(model.cls_dim, model.embed_dim)
model.cls_constraints = pyo.ConstraintList()
for c, c_dim in enumerate(model.cls_dim):
    for e, e_dim in enumerate(model.embed_dim):
        if c < 1:
            model.cls_constraints.add(expr= model.cls[c_dim, e_dim] == cls_token[e])
        else:
           model.cls_constraints.add(expr= model.cls[c_dim, e_dim] == model.embed_input[model.num_patch_dim.at(c), model.embed_dim.at(e+1)] )
            
# Add Positional Embedding
b_pe= parameters['pos_embedding']
out = transformer.add_pos_encoding(model.cls, "pe", b_pe )

# Layer Norm
gamma1 = parameters['layer_normalization_1', 'gamma']
beta1  = parameters['layer_normalization_1', 'beta']
out = transformer.add_layer_norm( out, "LN_1", gamma1, beta1)
res = out
       
Attention
layer = 'self_attention_1'
W_q = parameters[layer,'W_q']
W_k = parameters[layer,'W_k']
W_v = parameters[layer,'W_v']
W_o = parameters[layer,'W_o']

b_q = parameters[layer,'b_q']
b_k = parameters[layer,'b_k']
b_v = parameters[layer,'b_v']
b_o = parameters[layer,'b_o']
out = transformer.add_attention( out,"attention_output", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)

# # Residual 
# out = transformer.add_residual_connection(res, out, "residual_1")
# res = out
     
# # Layer Norm2
# gamma = parameters['layer_normalization_2', 'gamma']
# beta  = parameters['layer_normalization_2', 'beta']
# out = transformer.add_layer_norm( out, "LN_2", gamma, beta)

       
# # FFN
# nn, input_nn, output_nn = transformer.get_fnn(out, "ffn_1", "ffn_1", (num_patch_dim + 1, dim), parameters)
        
# # Residual 
# out = transformer.add_residual_connection(res, output_nn, "residual_2")

# # print("TNN output shape")
# # for i in out.index_set():
# #     print(i)
     
# # cls pool
# model.pool= pyo.Var(model.channel_dim, model.embed_dim)
# def pool_rule(model, d):
#     return model.pool[0, d] == out[0,d]
# model.pool_constr = pyo.Constraint(model.embed_dim, rule=pool_rule)

# # Norm
# gamma = parameters['layer_normalization_3', 'gamma']
# beta  = parameters['layer_normalization_3', 'beta']
# out = transformer.add_layer_norm( model.pool, "LN_3", gamma, beta)

# # Linear
# W_emb = parameters['linear_3', 'W']
# b_emb  = parameters['linear_3', 'b']
# out = transformer.embed_input( out, "output", model.out_labels_dim, W_emb, b_emb)

# Set objective
# model.obj = pyo.Objective(
#     expr= out[0, model.labels.last()] - out[0, model.labels.first()] , sense=pyo.maximize
# )  # -1: maximize, +1: minimize (default)

# TESTING
model.obj = pyo.Objective(
    expr= sum(model.purturb_image[i] - model.target_image[i] for i in model.purturb_image.index_set()), sense=pyo.minimize
)  # -1: maximize, +1: minimize (default)
# -------

# Convert & Solve 
# # Convert to gurobipy
gurobi_model, map_var, _ = convert_pyomo.to_gurobi(model)


# TESTING ---
# ## Add FNN1 to gurobi model
# input_1, output_1 = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
# pred_constr1 = add_predictor_constr(gurobi_model, nn, input_1, output_1)
#----------

# gurobi_model.update()

## Optimizes
# gurobi_model.setParam('DualReductions',0)
#gurobi_model.setParam('MIPFocus',1)
PATH = r".\Experiments\Verification"
experiment_name = "testing_veri"
gurobi_model.setParam('LogFile', PATH+f'\\Logs\\{experiment_name}_6.log')
gurobi_model.setParam('TimeLimit', 43200) #12h
gurobi_model.optimize()
    
# Results
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
            
if gurobi_model.status == GRB.INFEASIBLE:
        gurobi_model.computeIIS()
        gurobi_model.write("vit_model.ilp")
        
purturb_image = np.array(optimal_parameters['purturb_image'])
target_image = np.array(optimal_parameters['target_image'])

if TESTING:
    # check input to tnn
    patch = np.array(optimal_parameters['patch_input']).flatten()
    patch_exp = np.array(list(layer_outputs_dict['to_patch_embedding.0'])[0].tolist()).flatten() # convert from (image size * image size) to (patch_num * patch size)
    assert np.isclose(patch, patch_exp , atol=1e-6).all()
    
    # check linear layer:
    embed = np.array(optimal_parameters['embed_input'])
    embed_exp = np.array(list(layer_outputs_dict['to_patch_embedding'])[0].tolist()).flatten()
    assert np.isclose(embed, embed_exp , atol=1e-6).all()
    

    # check layer norm:
    val = np.array(optimal_parameters["LN_1"])
    val_exp = np.array(list(layer_outputs_dict['transformer.layers.0.0.norm'])[0].tolist()).flatten()
    assert np.isclose(val, val_exp , atol=1e-6).all()
    print("mean, min, max, diff images: ",np.mean(val - val_exp), max(val - val_exp), min(val - val_exp))

    
    # check self attention:
    val = np.array(optimal_parameters["attention_output"])
    val_exp = np.array(list(layer_outputs_dict['transformer.layers.0.0.fn.to_out.0'])[0].tolist()).flatten()
    assert np.isclose(val, val_exp , atol=1e-6).all()
    
print("---------------------------------------------------")
# print("purturbed image: ",purturb_image)
# print()
# print("target image: ",target_image)
print()
print("mean, min, max, diff images: ",np.mean(purturb_image - target_image), max(purturb_image - target_image), min(purturb_image - target_image))
print()
print("layer outputs trained tnn keys: \n", layer_outputs_dict.keys())
print("---------------------------------------------------")

        
    
