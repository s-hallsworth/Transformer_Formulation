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
import unittest
import os
from omlt import OmltBlock
import torch
from gurobipy import Model, GRB
from gurobi_ml import add_predictor_constr
import torchvision

# Import from repo file
from helpers.print_stats import solve_pyomo, solve_gurobipy
import helpers.convert_pyomo as convert_pyomo
from helpers.GUROBI_ML_helper import get_inputs_gurobipy_FNN
import transformer_b_flag as TNN
import helpers.extract_from_pretrained as extract_from_pretrained
from vit_TNN import *

TESTING = False # TESTING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

def verification_problem(inputimage, epsilon, channels, image_size, labels, classification_labels):
    max_input = np.max(inputimage.numpy())
    min_input = np.min(inputimage.numpy())
    input = torch.as_tensor(inputimage).float().reshape(1, channels, image_size, image_size) #image_size * image_size image
    tgt_dict = {}
    for p1 in range(image_size):
        for p2 in range(image_size):
            tgt_dict[p1, p2] = input[0,0,p1,p2].tolist()
            
    # Create pyomo model
    model = pyo.ConcreteModel(name="(ViT)")

    # Define parameters and sets
    model.labels = pyo.Set(initialize=labels)
    model.channel_dim = pyo.Set(initialize=range(channels))
    model.image_dim = pyo.Set(initialize=range(image_size))
    model.out_labels_dim = pyo.Set(initialize=range(classification_labels))

    model.eps = pyo.Param(initialize=epsilon)
    model.target_image = pyo.Param(model.image_dim, model.image_dim, initialize=tgt_dict)

    # Define variables
    model.purturb_image = pyo.Var(model.image_dim, model.image_dim)
    model.purturb = pyo.Var(model.image_dim, model.image_dim, bounds=(0, model.eps))
    model.out = pyo.Var(model.channel_dim, model.out_labels_dim)
    
    # Add constraints to purturbed image:
    model.purturb_constraints = pyo.ConstraintList()
    for i in model.purturb_image.index_set():
        model.purturb_image[i].lb = max( model.target_image[i] - epsilon, min_input) # cap min value of pixels
        model.purturb_image[i].ub = min( model.target_image[i] + epsilon, max_input) # cap max value of pixels
        
        #purturb >= to the abs different between x and x'
        model.purturb_constraints.add(expr= model.purturb[i] >= model.purturb_image[i] - model.target_image[i])
        model.purturb_constraints.add(expr= model.purturb[i] >= model.target_image[i] - model.purturb_image[i])
        
    # total purturb at each pixel <= epsilon   
    model.purturb_constraints.add(expr= sum(model.purturb[i] for i in model.purturb.index_set()) <= model.eps) 
    
    # Set objective:
    model.obj = pyo.Objective(
        expr= -(model.out[0, model.labels.last()] - model.out[0, model.labels.first()]) , sense=pyo.minimize
    )  # -1: maximize, +1: minimize (default); last-->incorrect label, first-->correct label

    return model, input
   
def verification_tnn(model, inputimage, image_size, patch_size, channels, file_name, tnn_path, activation_dict, device, eps=1e-6):
    # Load transformer
    config_params = file_name.split('_')
    image_size_flat = image_size * image_size
    dim= int(config_params[1])
    depth= int(config_params[2])
    heads= int(config_params[3])
    mlp_dim= int(config_params[4])
    head_size = int(dim/heads)
    config_list = [channels, dim, head_size , heads, image_size*image_size, eps]
    tnn_model = torch.load(tnn_path, map_location=device)
    input = torch.as_tensor(inputimage).float().reshape(1, channels, image_size, image_size) #image_size * image_size image
    
    # Define formulated transformer
    transformer = TNN.Transformer( config_list, model, activation_dict)
    layer_names, parameters, _, layer_outputs_dict = extract_from_pretrained.get_torchViT_learned_parameters(tnn_model, input, heads)
        
    # Add Sequential 
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
    transformer.add_pos_encoding("cls", "pe", b_pe )

    # Layer Norm
    gamma1 = parameters['layer_normalization_1', 'gamma']
    beta1  = parameters['layer_normalization_1', 'beta']
    transformer.add_layer_norm( "pe", "LN_1", gamma1, beta1)
    res = "pe"
        
    ffn_parameter_dict = {}
    for l in range(depth):
        # Layer Norm
        gamma1 = parameters['layer_normalization_1', 'gamma']
        beta1  = parameters['layer_normalization_1', 'beta']
        if l < 1:
            res = "pe"
        transformer.add_layer_norm( res, f"LN_1_{l}", gamma1, beta1)
        
            
        # # Attention
        layer = 'self_attention_1'
        W_q = parameters[layer,'W_q']
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']

        b_q = parameters[layer,'b_q']
        b_k = parameters[layer,'b_k']
        b_v = parameters[layer,'b_v']
        b_o = parameters[layer,'b_o']
        transformer.add_attention( f"LN_1_{l}",f"attention_output_{l}", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, tnn_from="pytorch")

        # Residual 
        transformer.add_residual_connection(res, f"attention_output_{l}", f"residual_1_{l}")
        res = f"residual_1_{l}"
            
        # Layer Norm2
        gamma = parameters['layer_normalization_2', 'gamma']
        beta  = parameters['layer_normalization_2', 'beta']
        transformer.add_layer_norm( f"residual_1_{l}", f"LN_2_{l}", gamma, beta)

            
        # # # FFN
        ffn_params =  transformer.get_fnn(f"LN_2_{l}", f"ffn_1_{l}", "ffn_1", (num_patch_dim + 1, dim), parameters)
        ffn_parameter_dict[f"ffn_1_{l}"] = ffn_params # ffn_params: nn, input_nn, output_nn
                
        # Residual 
        out = transformer.add_residual_connection(res, f"ffn_1_{l}", f"residual_2_{l}")
        res = f"residual_2_{l}"

        
    # cls pool
    model.pool= pyo.Var(model.channel_dim, model.embed_dim)
    def pool_rule(model, d):
        return model.pool[0, d] == out[0,d]
    model.pool_constr = pyo.Constraint(model.embed_dim, rule=pool_rule)

    # Norm
    gamma = parameters['layer_normalization_3', 'gamma']
    beta  = parameters['layer_normalization_3', 'beta']
    out = transformer.add_layer_norm( model.pool, "LN_3", gamma, beta)

    # Linear
    W_emb = parameters['linear_3', 'W']
    b_emb  = parameters['linear_3', 'b']
    out = transformer.embed_input( out, "output", model.out_labels_dim, W_emb, b_emb)
    
    # output constraints
    model.out_constraints = pyo.ConstraintList()
    for i in model.out.index_set():
        model.out_constraints.add(expr= model.out[i] == out[i])
        
    return model, ffn_parameter_dict, layer_outputs_dict, transformer