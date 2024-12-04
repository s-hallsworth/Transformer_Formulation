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
from gurobipy import Model, GRB
from gurobi_ml import add_predictor_constr
import torchvision

# Import from repo file
from MINLP_tnn.helpers.print_stats import solve_pyomo, solve_gurobipy
import MINLP_tnn.helpers.convert_pyomo as convert_pyomo
from MINLP_tnn.helpers.GUROBI_ML_helper import get_inputs_gurobipy_FFN
import transformer_b_flag as TNN
from Examples.optimal_trajectory.training.Tmodel import TransformerModel
import MINLP_tnn.helpers.extract_from_pretrained as extract_from_pretrained

TESTING = True # TESTING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off
#Load MNIST data
torch.manual_seed(42)
DOWNLOAD_PATH = '/data/mnist'
transform_mnist = torchvision.transforms.Compose([ torchvision.transforms.Resize((4, 4)), ##
                                                  torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0,), (1,))])
                               #torchvision.transforms.Normalize((0.1307,), (0.3081,))]) #transform to match training data scale
mnist_testset = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True, transform=transform_mnist)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=False)
images, labels = next(iter(test_loader))

# Set parameters
problemNo = 0 # image to select from MNIST dataset ( test: 0 --> the number 7)
epsilon = 0.0001
inputimage = torch.round(images[problemNo], decimals=4) # flattened image
max_input = np.max(inputimage.numpy())
min_input = np.min(inputimage.numpy())
labels = [labels[problemNo].item(), 1] # [true label, adversary label]
classification_labels = 10
channels = 1
image_size=4 ##
image_size_flat = image_size * image_size
patch_size=2 ##
num_classes=10
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
model.purturb_image = pyo.Var(model.image_dim, model.image_dim)
model.purturb = pyo.Var(model.image_dim, model.image_dim, bounds=(0, model.eps))

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
   
   
# Load transformer
from vit_TNN import *
file_name = "vit_6_2_6_12"
tnn_path = f".\\trained_transformer\\verification_16\\{file_name}.pt" 
device = 'cpu'
config_params = file_name.split('_')
dim= int(config_params[1])
depth= int(config_params[2])
heads= int(config_params[3])
mlp_dim= int(config_params[4])
head_size = int(dim/heads)
config_list = [channels, dim, head_size , heads, image_size*image_size, 1e-6]
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
#1, 1, 0, 0, 0
1 , 0, 1, 1, 1, #1 all
#1 , 0, 1, 1, 0 #2 -- fastest feasibile solution _/
#1 , 0, 1, 0, 0, #3 -- good trade off speed and solve time _/
#1 , 0, 0, 0, 0, #4 -- smallest opt. gap _/
#1 , 0, 0, 1, 1, #5_/
#1 , 0, 0, 1, 0, #6 --- fastest optimal solution _/
# 0 , 0, 0, 0, 0  #7 _/
]
combinations = [bool(val) for val in combinations]

ACTI = {}  
ACTI["LN_I"] = {"list": ["LN_var"]}
ACTI["LN_D"] = {"list": ["LN_num", "LN_num_squ", "LN_denom"]}
ACTI["MHA_I"] = {"list": ["MHA_attn_weight_sum", "MHA_attn_weight"]}
#ACTI["MHA_D"] = {"list": ["MHA_Q", "MHA_K", "MHA_V", "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_attn_score", "MHA_output" , "RES_var"]}
ACTI["MHA_D"] = {"list": ["MHA_Q", "MHA_K", "MHA_V", "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_attn_score", "MHA_output" , "RES_var"]} # ]}# 
ACTI["MHA_MC"] = {"list":[ "MHA_QK_MC", "MHA_WK_MC"]}

ACTI["LN_I"]["act_val"], ACTI["LN_D"]["act_val"], ACTI["MHA_I"]["act_val"] , ACTI["MHA_D"]["act_val"], ACTI["MHA_MC"]["act_val"] = combinations

for k, val in ACTI.items():
    for elem in val["list"]:
        activation_dict[elem] = val["act_val"] # set activation dict to new combi
 
# TESTING ----   
# Define formulated transformer
transformer = TNN.Transformer( config_list, model, activation_dict)
layer_names, parameters, _, layer_outputs_dict = extract_from_pretrained.get_torchViT_learned_parameters(tnn_model, input, heads)
# if TESTING:
#     plt.imshow(input.squeeze(0).squeeze(0), cmap='gray')
#     plt.show()
  
# list to help create new variable names for each layer  
count_list = []
def count_layer_name(layer_name, count_list):
    count = count_list.count(layer_name) + 1
    count_list.append(layer_name)
    return count

# Add Sequential 1 x28 x 18 mult 18 x patch size
num_patch_dim = int(image_size_flat/(patch_size*patch_size))
model.num_patch_dim = pyo.Set(initialize=range(num_patch_dim ))
model.patch_dim = pyo.Set(initialize=range(patch_size*patch_size))
model.embed_dim = pyo.Set(initialize=range(dim))

layer_name = "linear"
count = count_layer_name(layer_name, count_list)
W_emb = parameters[f"{layer_name}_{count}",'W']
b_emb = parameters[f"{layer_name}_{count}",'b']
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
transformer.add_layer_norm( "pe", "LN_1_1", gamma1, beta1)
res = "pe"

ffn_parameter_dict = {}

for l in range(depth):
    # Layer Norm
    layer_name = "layer_normalization"
    count = count_layer_name(layer_name, count_list)
    gamma1 = parameters[f'{layer_name}_{count}', 'gamma']
    beta1  = parameters[f'{layer_name}_{count}', 'beta']
    if l < 1:
        res = "pe"

    transformer.add_layer_norm( res, f"LN_{count}", gamma1, beta1)
    prev = f"LN_{count}"
        
    # # Attention
    layer_name= 'self_attention'
    count = count_layer_name(layer_name, count_list)
    W_q = parameters[f"{layer_name}_{count}",'W_q']
    W_k = parameters[f"{layer_name}_{count}",'W_k']
    W_v = parameters[f"{layer_name}_{count}",'W_v']
    W_o = parameters[f"{layer_name}_{count}",'W_o']

    b_q = parameters[f"{layer_name}_{count}",'b_q']
    b_k = parameters[f"{layer_name}_{count}",'b_k']
    b_v = parameters[f"{layer_name}_{count}",'b_v']
    b_o = parameters[f"{layer_name}_{count}",'b_o']
    transformer.add_attention( prev, f"attention_output_{count}", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, norm_softmax=True)
    prev = f"attention_output_{count}"
    
    # Residual 
    layer_name= 'residual'
    count = count_layer_name(layer_name, count_list)
    transformer.add_residual_connection(res, prev, f"residual_{count}")
    res = f"residual_{count}"
    
    # Layer Norm2
    layer_name = "layer_normalization"
    count = count_layer_name(layer_name, count_list)
    gamma = parameters[f'{layer_name}_{count}', 'gamma']
    beta  = parameters[f'{layer_name}_{count}', 'beta']
    out = transformer.add_layer_norm( res, f"LN_{count}", gamma, beta)
    prev = f"LN_{count}"
        
    # # # FFN
    layer_name = "ffn"
    count = count_layer_name(layer_name, count_list)
    ffn_params =  transformer.get_ffn(out, f"{layer_name}_{count}", f"{layer_name}_{count}", (num_patch_dim + 1, dim), parameters)
    ffn_parameter_dict[f"{layer_name}_{count}"] = ffn_params # ffn_params: nn, input_nn, output_nn
    prev = f"{layer_name}_{count}"
          
    # Residual 
    layer_name= 'residual'
    count = count_layer_name(layer_name, count_list)
    out = transformer.add_residual_connection(res, prev, f"residual_{count}")
    res = out

     
# cls pool
model.pool= pyo.Var(model.channel_dim, model.embed_dim)
def pool_rule(model, d):
    return model.pool[0, d] == out[0,d]
model.pool_constr = pyo.Constraint(model.embed_dim, rule=pool_rule)

# Norm
layer_name = "layer_normalization"
count = count_layer_name(layer_name, count_list)
gamma = parameters[f'{layer_name}_{count}', 'gamma']
beta  = parameters[f'{layer_name}_{count}', 'beta']
out = transformer.add_layer_norm( model.pool, f"LN_{count}", gamma, beta)

# Linear
layer_name = "linear"
count = count_layer_name(layer_name, count_list)
W_emb = parameters[f"{layer_name}_{count}",'W']
b_emb = parameters[f"{layer_name}_{count}",'b']
out = transformer.embed_input( out, "output", model.out_labels_dim, W_emb, b_emb)

# #Set objective
# model.obj = pyo.Objective(
#     expr= (out[0, model.labels.last()] - out[0, model.labels.first()]) , sense=pyo.maximize
# )  # -1: maximize, +1: minimize (default); last-->incorrect label, first-->correct label

##
# model.obj = pyo.Objective(
#     expr= -(out[0, model.labels.last()] - out[0, model.labels.first()]) , sense=pyo.minimize
# )  # -1: maximize, +1: minimize (default); last-->incorrect label, first-->correct label


# # TESTING
model.obj = pyo.Objective(
    expr= sum(model.purturb_image[i] - model.target_image[i] for i in model.purturb_image.index_set()), sense=pyo.minimize
)  # -1: maximize, +1: minimize (default)
# -------

# Convert & Solve 
# # Convert to gurobipy
gurobi_model, map_var, _ = convert_pyomo.to_gurobi(model)

# Add FNNs to gurobi model using GurobiML
for key, value in ffn_parameter_dict.items():
    nn, input_nn, output_nn = value
    input, output = get_inputs_gurobipy_FFN(input_nn, output_nn, map_var)
    pred_constr = add_predictor_constr(gurobi_model, nn, input, output)

gurobi_model.update() # update gurobi model with FFN constraints

## Optimizes
# gurobi_model.setParam('DualReductions',0)
# gurobi_model.setParam('MIPFocus',1)
PATH = r".\Experiments\Verification"
experiment_name = "testing_veri"
gurobi_model.setParam('LogFile', PATH+f'\\Logs\\{experiment_name}_{file_name}.log')
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
    
    # check linear layer:
    embed = np.array(optimal_parameters['embed_input'])
    embed_exp = np.array(list(layer_outputs_dict['to_patch_embedding'])[0].tolist()).flatten()
    

    # check layer norm:
    val = np.array(optimal_parameters["LN_1"])
    val_exp = np.array(list(layer_outputs_dict['transformer.layers.0.0.norm'])[0].tolist()).flatten()
    #assert np.isclose(val, val_exp , atol=1e-5).all()
    print("ln1: mean, min, max ",np.mean(val), np.max(val ), np.min(val))
    print("ln1: mean, min, max, diff images: ",np.mean(val - val_exp), np.max(val - val_exp), np.min(val - val_exp))
    
    # check self attention:
    val = np.array(optimal_parameters["attention_output_1"])
    val_exp = np.array(list(layer_outputs_dict['transformer.layers.0.0.fn.to_out'])[0].tolist()).flatten()
    print("attn_out: mean, min, max, diff images: ",np.mean(val - val_exp), max(val - val_exp), min(val - val_exp))
    
    ########################## dubugging code
    output_name = "attention_output_1"
    Q_form = torch.tensor(optimal_parameters[f"Block_{output_name}.Q"])
    K_form = torch.tensor(optimal_parameters[f"Block_{output_name}.K"])
    V_form = torch.tensor(optimal_parameters[f"Block_{output_name}.V"])
    
    attn_weight = torch.tensor(optimal_parameters[f"Block_{output_name}.attention_weight"])
    
    from einops import rearrange
    Q_form = rearrange(Q_form, '(h d k) -> d (k h)', d=num_patch_dim+1, h=heads)
    K_form = rearrange(K_form, '(h d k) -> d (k h)', d=num_patch_dim+1, h=heads)
    V_form = rearrange(V_form, '(h d k) -> d (k h)', d=num_patch_dim+1, h=heads)
            
    print(Q_form.shape)
    val = np.array(torch.stack((Q_form, K_form, V_form), dim = -1).permute(0,2,1).tolist()).flatten()
    val_exp = np.array(list(layer_outputs_dict['transformer.layers.0.0.fn.to_qkv'])[0].tolist()).flatten()
    print("qkv: mean, min, max, diff images: ",np.mean(val - val_exp), max(val - val_exp), min(val - val_exp))

    val = np.array(attn_weight.tolist()).flatten()
    val_exp = np.array(list(layer_outputs_dict['transformer.layers.0.0.fn.attend'])[0].tolist()).flatten()
    print("attn: mean, min, max ",np.mean(val), max(val ), min(val))
    print("attn weight: mean, min, max, diff images: ",np.mean(val - val_exp), max(val - val_exp), min(val - val_exp))
    
    # check layer norm:
    val = np.array(optimal_parameters["LN_2"])
    val_exp = np.array(list(layer_outputs_dict['transformer.layers.0.1.norm'])[0].tolist()).flatten()
    print("ln2: mean, min, max ",np.mean(val), np.max(val ), np.min(val))
    print("ln2: mean, min, max, diff images: ",np.mean(val - val_exp), np.max(val - val_exp), np.min(val - val_exp))
    
    # check ffn:
    val = np.array(optimal_parameters["ffn_1"])
    val_exp = np.array(list(layer_outputs_dict['transformer.layers.0.1.fn.net'])[0].tolist()).flatten()
    print("ffn: mean, min, max, diff images: ",np.mean(val - val_exp), max(val - val_exp), min(val - val_exp))
    
    # check pool:
    val = np.array(optimal_parameters["pool"])
    val_exp = np.array(list(layer_outputs_dict['to_latent'])[0].tolist()).flatten()
    print("pool: mean, min, max, diff images: ",np.mean(val - val_exp), max(val - val_exp), min(val - val_exp))

    # # check layer norm3:
    layer_name = "layer_normalization"
    count = count_layer_name(layer_name, count_list)-1
    val = np.array(optimal_parameters[f"LN_{count}"])
    val_exp = np.array(list(layer_outputs_dict['mlp_head.0'])[0].tolist()).flatten()
    print("ln3: mean, min, max, diff images: ",np.mean(val - val_exp), max(val - val_exp), min(val - val_exp))


    # check output:
    val = np.array(optimal_parameters["output"])
    val_exp = np.array(list(layer_outputs_dict['mlp_head'])[0].tolist()).flatten()
    print("form out: ",val)
    print("trained tnn out: ",val_exp)
    print("out: mean, min, max, diff images: ",np.mean(val - val_exp), max(val - val_exp), min(val - val_exp))
    #assert np.isclose(val, val_exp , atol=1e-5).all()

# print("---------------------------------------------------")
# # print("purturbed image: ",purturb_image)
# # print()
# # print("target image: ",target_image)
# print()
# print("mean, min, max, diff images: ",np.mean(purturb_image - target_image), max(purturb_image - target_image), min(purturb_image - target_image))
# print()
# print("layer outputs trained tnn keys: \n", layer_outputs_dict.keys())
# print("---------------------------------------------------")

        
    