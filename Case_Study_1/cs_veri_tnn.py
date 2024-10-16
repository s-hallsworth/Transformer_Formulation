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
Parameters can be changed on lines 45-51.
    
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


problemNo = 0 # image to select from MNIST dataset
epsilon = 5
nLayers = 3
instances = np.load(r'.\data\mnist2x50instances.npz')
inputimage = instances['images'][problemNo]
labels = instances['labels'][problemNo] # [true label, adversary label]
image_size = instances['w1'].shape[1]
classification_labels = 10
tgt_dict = {}
print(inputimage.shape)
for key, value in zip(range(image_size), inputimage):
    tgt_dict[key] = value

# Create pyomo model
model = pyo.ConcreteModel(name="(TOY_TRANFORMER)")

# Define parameters and sets
model.labels = pyo.Set(initialize=labels)
model.image_dim = pyo.Set(initialize=range(image_size))
model.out_labels_dim = pyo.Set(initialize=range(classification_labels))

model.eps = pyo.Param(initialize=epsilon)
model.NN_output = pyo.Param(model.out_labels_dim)
model.purturb_M = pyo.Param(initialize= max(abs(inputimage))+epsilon-1)
model.target_image = pyo.Param(model.image_dim, initialize=tgt_dict)

# Define variables
model.purturb_image = pyo.Var(model.image_dim)
model.purturb = pyo.Var(model.image_dim, bounds=(0, model.eps))
model.purturb_s_min = pyo.Var(model.image_dim, within=pyo.Binary)
model.purturb_s_max = pyo.Var(model.image_dim, within=pyo.Binary)

# Add constraints to purturbed image:
model.purturb_constraints = pyo.ConstraintList()
for i in model.purturb_image.index_set():
    # purturb image: less than min(max purturb,1)
    model.purturb_constraints.add(expr= model.purturb_image[i] <= model.purturb_s_min[i] + ((model.target_image[i] + model.eps)*(model.purturb_s_min[i]-1)) ) # less than min(max purturb,1)
    model.purturb_constraints.add(expr= model.target_image[i] + model.eps - 1 <= model.purturb_s_min[i] * model.purturb_M ) 
    model.purturb_constraints.add(expr= model.target_image[i] + model.eps - 1 >= (model.purturb_s_min[i]-1) * model.purturb_M ) 
    
    # purturb image: greater than max(min purturb,0)
    model.purturb_constraints.add(expr= model.purturb_image[i] >= (model.target_image[i] - model.eps)*model.purturb_s_max[i] )
    model.purturb_constraints.add(expr= model.target_image[i] - model.eps <= model.purturb_s_max[i] * model.purturb_M )
    model.purturb_constraints.add(expr= model.target_image[i] - model.eps >= (model.purturb_s_max[i] - 1 )* model.purturb_M )
    
    #purturb: greater than or equal to the abs different between x and x'
    model.purturb_constraints.add(expr= model.purturb[i] >= model.purturb_image[i] - model.target_image[i])
    model.purturb_constraints.add(expr= model.purturb[i] >= model.target_image[i] - model.purturb_image[i])

# total purturb at each pixel <= epsilon   
model.purturb_constraints.add(expr= sum(model.purturb[i] for i in model.image_dim) <= model.eps) 
   
   
# Load transformer
from vit_TNN import *
file_name = "vit_18_1_6_12"
tnn_path = f".\\trained_transformer\\verification\\{file_name}.pt" 
device = 'cpu'
config_params = file_name.split('_')
image_size=28
patch_size=4 
num_classes=10
channels=1
dim= int(config_params[1])
depth= int(config_params[2])
heads= int(config_params[3])
mlp_dim= int(config_params[4])
# tnn_model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, channels=channels, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim)
# tnn_model.load_state_dict(torch.load(tnn_path, map_location=device))
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
# [1 , 0, 1, 0, 0], #3 -- good trade off speed and solve time
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
    
# Define formulated transformer
transformer = TNN.Transformer( ".\\data\\verification_config.json", model, activation_dict)
input = torch.as_tensor(inputimage).float().reshape(image_size, image_size)
input = input.unsqueeze(0).unsqueeze(0) # b c h w
layer_names, parameters, _, layer_outputs_dict = extract_from_pretrained.get_torchViT_learned_parameters(tnn_model, input, heads)
    
# Transformer input var:
tnn_input = model.purturb_image

# Add Sequential
layer = "linear_1"
W_linear = parameters[layer,'W']
b_linear = parameters[layer,'b']
transformer.embed_input(tnn_input, "embed", dim, W_linear, b_linear)
# Add Transfromer Layers

# Add Output transforms


# # Set objectivve
# model.setObjective(-(x[ind+1][labels[1]] - x[ind+1][labels[0]])) #default is min. Thus equiv to max: x[ind+1][labels[1]] - x[ind+1][labels[0]]

# model.setParam('MIPFocus',3) # 3: focus on improving  the dual bound. https://www.gurobi.com/documentation/current/refman/mipfocus.html
# model.setParam('Cuts',1)     # 1: Moderate cut generation. https://www.gurobi.com/documentation/current/refman/cuts.html 
# model.setParam('Method', 1)  # 1: dual simplex used to solve continuous model or root node relaxation
# model.setParam('TimeLimit',3600)
# #model.setParam('DisplayInterval', 50)
           
# # model.optimize()
    



        
        
    
