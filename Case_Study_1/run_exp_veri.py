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
Parameters can be changed on lines 57-69.
    
    Parameters:
        problemNo (int): index of problem to be solved, can be in range(100)
        epsilon (real): maximum l1-norm defining perturbations
"""

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
import torchvision

# Import from repo files
from vit_TNN import *
from cs_veri_tnn_exp import *
import transformer_b_flag as TNN
import helpers.extract_from_pretrained as extract_from_pretrained
from helpers.print_stats import solve_pyomo, solve_gurobipy, save_gurobi_results
import helpers.convert_pyomo as convert_pyomo
from helpers.GUROBI_ML_helper import get_inputs_gurobipy_FNN

# turn off floating-point round-off
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' 

"""
    Solve verification problem with varying setups:
        - change active constraints, tnn file (automated)
        - change framework + solver (manual)
"""

# Set Up
TESTING = False # fix TNN input for testing (faster solve)
if TESTING: 
    REP = 1
else:
    REP = 1 # number of repetitions of each scenario
r_offset = 0
NAME = "verification"
SOLVER = "gurobi"
FRAMEWORK = "gurobipy"
im_sz=[4]  # Define image pixels (and folder to select tnn models from)
#file_names = ["vit_6_1_6_12", "vit_6_2_6_12", "vit_6_4_6_12"] # changing depth
file_names = ["vit_6_1_6_12", "vit_12_1_6_12", "vit_6_2_6_12", "vit_12_2_6_12" ]
#file_names = ["vit_12_1_6_12", "vit_18_1_6_12", "vit_24_1_6_12"] # changing embed dim 12, 18, 24
#file_names = ["vit_12_1_6_12", "vit_12_2_6_12", "vit_12_4_6_12"] # changing embed dim 12, for each depth
#file_names = ["vit_18_1_6_12", "vit_18_2_6_12", "vit_18_4_6_12"] # changing embed dim 18, for each depth
#file_names = ["vit_24_1_6_12", "vit_24_2_6_12", "vit_24_4_6_12"] # changing embed dim 24, for each depth

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
    [1 , 0, 1, 1, 0], #2 -- fastest feasibile solution _/  No Mc
    [1 , 0, 1, 0, 0], #3 -- good trade off speed and solve time _/ I only
    #1 , 0, 0, 0, 0, #4 -- smallest opt. gap _/
    #1 , 0, 0, 1, 1, #5_/
    [1 , 0, 0, 1, 0], #6 --- fastest optimal solution _/ LNprop
    
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
tnn_config["Num Enc"] = "1"
tnn_config["Num Dec"] =" 0"
tnn_config["Num Res"] = "2"
tnn_config["Num LN"]  = "2 + 1"
tnn_config["Num AVGP"] = "0"
tnn_config["Num Dense"] = "2 + 3"
tnn_config["Num ReLu"] = "1"
tnn_config["Num Pool"] = " 0 + 1"

# Define Optimisation Problem:
problemNo = 0 # image to select from MNIST dataset ( test: 0 --> the number 7)
epsilon = 0.0001 ##
channels = 1
patch_size=2 
classification_labels = 10
adv_label = 1
    

## RUN EXPERIMENTS:
# For varied pixel sized images
for image_size in im_sz:
    # Set output directory
    PATH =  f".\\Experiments\\Verification_{image_size*image_size}_eps{epsilon}"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        os.makedirs(PATH+"\\Logs")
    PATH += "\\"
    
    # Load Data Set
    torch.manual_seed(42)
    DOWNLOAD_PATH = '/data/mnist'
    transform_mnist = torchvision.transforms.Compose([ torchvision.transforms.Resize((image_size, image_size)), ##
                                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0,), (1,))])
    mnist_testset = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True, transform=transform_mnist)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=1, shuffle=False)
    images, labels = next(iter(test_loader))
    
    inputimage = images[problemNo] # flattened image
    labels = [labels[problemNo].item(), adv_label] # [true label, adversary label]
    model, input = verification_problem(inputimage, epsilon, channels, image_size, labels, classification_labels)

    # For each trained TNN
    for file_name in file_names:
        # for each experiment repetition
        for r in range(REP):
            for c, combi in enumerate(combinations):
                print("C = ", c+1)    
                if c+1 != 4: ### REMOVE
                    continue
                experiment_name = f"{file_name}_i{image_size}_r{r+1+r_offset}_c{c+1}"
                # activate constraints
                ACTI["LN_I"]["act_val"], ACTI["LN_D"]["act_val"], ACTI["MHA_I"]["act_val"] , ACTI["MHA_D"]["act_val"], ACTI["MHA_MC"]["act_val"] = combi

                for k, val in ACTI.items():
                    for elem in val["list"]:
                        activation_dict[elem] = val["act_val"] # set activation dict to new combi
                tnn_config["Activated Bounds/Cuts"] = activation_dict # save act config

                # clone optimization model
                m = model.clone()
                
                # Define ViT:
                tnn_path = f".\\trained_transformer\\verification_{image_size*image_size}\\{file_name}.pt" 
                device = 'cpu'
                
                print(file_name, tnn_path)
                m, ffn_parameter_dict, layer_outputs_dict, transformer = verification_tnn(m, inputimage, image_size, patch_size, channels, file_name, tnn_path, activation_dict, device)
                tnn_config["TNN Out Expected"] = np.array(list(layer_outputs_dict['mlp_head'])[0].tolist()).flatten()

                # Convert to gurobipy
                gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
                
                # Add FNNs to gurobi model using GurobiML
                for key, value in ffn_parameter_dict.items():
                    nn, input_nn, output_nn = value
                    input, output = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
                    pred_constr = add_predictor_constr(gurobi_model, nn, input, output)

                gurobi_model.update() # update gurobi model with FFN constraints

                ## Optimizes
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
                    tnn_config["max purturb"] = np.max(np.array(optimal_parameters["purturb"]))
                    tnn_config["min purturb"] = np.min(np.array(optimal_parameters["purturb"]))
                    tnn_config["TNN Out"] = np.array(optimal_parameters["output"])  
                else:
                    tnn_config["TNN Out"] = None
                    
                # save results
                tnn_config["Enc Seq Len"] = transformer.N
                tnn_config["TNN Model Dims"] = transformer.d_model
                tnn_config["TNN Head Dims"] = transformer.d_k
                tnn_config["TNN Head Size"] = transformer.d_H
                tnn_config["TNN Input Dim"] = transformer.input_dim


                if not TESTING:
                    save_gurobi_results(gurobi_model, PATH+experiment_name, experiment_name, r+1+r_offset, tnn_config)