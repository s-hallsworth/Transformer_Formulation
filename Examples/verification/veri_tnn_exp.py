import pyomo.environ as pyo
import numpy as np
import os
import torch

# Import from repo file
from MINLP_tnn.transformer import Transformer as TNN
import MINLP_tnn.helpers.extract_from_pretrained as extract_from_pretrained
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off


"""
Verification of Transformer Neural Networks using Pyomo and Gurobi.

This script defines functions and utilities for verifying Transformer Neural Networks (TNNs) 
on the MNIST dataset. The TNN is modelled as a Mixed Integer Nonlinear 
Programming (MINLP) problem and solved using Pyomo and Gurobi frameworks.

Functions:
- `verification_problem`: 
    Constructs a Pyomo model for verifying adversarial robustness of a TNN with input image 
    perturbations constrained by an epsilon value.
    
- `count_layer_name`: 
    Tracks and counts occurrences of layer names for creating unique identifiers in the 
    Transformer formulation.
    
- `verification_tnn`: 
    Formulates a Transformer architecture as constraints in the optimization model, 
    including embedding, layer normalization, multi-head attention, feed-forward layers, 
    and positional encoding.

Dependencies:
- Pyomo for optimization model construction.
- PyTorch for loading pretrained TNN models.
- Gurobi for solving the formulated optimization problem.
- torchvision for loading and preprocessing the MNIST dataset.
- omlt and custom helper functions for integrating TNN layers with Pyomo models.

Usage:
1. Define the `verification_problem` to initialize the Pyomo model with perturbation constraints.
2. Use `verification_tnn` to incorporate TNN architecture as constraints.
3. Solve the resulting MINLP using Gurobi or Pyomo solvers in a separate script (e.g. run_exp_veri.py).
4. Analyze the results to verify adversarial robustness.
"""

def verification_problem(inputimage, epsilon, channels, image_size, labels, classification_labels):
    """
    Constructs a Pyomo model for adversarial robustness verification of a vision Transformer Neural Network (TNN).

    Args:
        inputimage (torch.Tensor): Input image tensor with shape (C, H, W) representing the MNIST image.
        epsilon (float): Maximum allowed perturbation per pixel.
        channels (int): Number of image channels (e.g., 1 for grayscale images).
        image_size (int): Dimension of the input image (e.g., 28 for 28x28 images).
        labels (list): List of two integers `[true_label, adversarial_label]`.
        classification_labels (int): Total number of classification labels in the dataset.

    Returns:
        tuple:
            - model (pyo.ConcreteModel): Pyomo model with constraints for image perturbation and TNN outputs.
            - input (torch.Tensor): Preprocessed input image reshaped into the required format.

    Notes:
        - The objective function maximizes the adversarial output difference between true and adversarial labels.
        - Perturbation constraints ensure that pixel modifications are bounded by `epsilon`.
    """
    
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
    
    # Set objective 
    model.obj = pyo.Objective(
        expr= (model.out[0, model.labels.last()] - model.out[0, model.labels.first()]) , sense=pyo.maximize
    )  # -1: maximize, +1: minimize (default); last-->incorrect label, first-->correct label


    return model, input

def count_layer_name(layer_name, count_list):
        """
        Tracks and counts occurrences of layer names to create unique identifiers for Transformer layers.

        Args:
            layer_name (str): Name of the layer being added to the Transformer model.
            count_list (list): List that tracks occurrences of layer names to ensure unique identifiers.

        Returns:
            int: Updated count of the layer in the `count_list`.

        Notes:
            - This function is used to maintain unique naming for layers when added as Pyomo constraints.
        """
        count = count_list.count(layer_name) + 1
        count_list.append(layer_name)
        return count
   
def verification_tnn(model, inputimage, image_size, patch_size, channels, file_name, tnn_path, activation_dict, device, eps=1e-6):
    """
    Incorporates a pretrained Transformer Neural Network (TNN) into a Pyomo model as mathematical constraints using the MINLP TNN.

    Args:
        model (pyo.ConcreteModel): Pyomo model instance to which the TNN constraints will be added.
        inputimage (torch.Tensor): Input image tensor reshaped for processing by the Transformer model.
        image_size (int): Dimension of the input image (e.g., 28 for 28x28 images).
        patch_size (int): Size of each patch in the image for the patch embedding layer.
        channels (int): Number of image channels (e.g., 1 for grayscale images).
        file_name (str): Name of the configuration file describing the TNN architecture.
        tnn_path (str): Path to the pretrained TNN model file (.pt format).
        activation_dict (dict): Dictionary specifying which constraints are active for each Transformer layer.
        device (str): Computational device to use ('cpu' or 'cuda').
        eps (float, optional): Small value to prevent division by zero during normalization. Defaults to 1e-6.

    Returns:
        tuple:
            - model (pyo.ConcreteModel): Updated Pyomo model with Transformer architecture constraints.
            - ffn_parameter_dict (dict): Dictionary containing feed-forward network (FFN) parameters.
            - layer_outputs_dict (dict): Outputs of layers from the pretrained Transformer model.
            - transformer (TNN): Transformer object encapsulating the formulated constraints.

    Notes:
        - Converts the pretrained Vision Transformer (ViT) model into constraints, including embedding, 
          positional encoding, layer normalization, self-attention, and feed-forward layers.
        - Ensures compatibility with Pyomo optimization models for adversarial robustness verification.
    """
    
    # Load vision transformer
    config_params = file_name.split('_')
    image_size_flat = image_size * image_size
    dim= int(config_params[1])
    depth= int(config_params[2])
    heads= int(config_params[3])
    mlp_dim= int(config_params[4])
    head_size = int(dim/heads)
    num_classes = 10
    config_list = [channels, dim, head_size , heads, image_size*image_size, eps]
    
    tnn_model = torch.load(tnn_path, map_location=device, weights_only=False)
    input = torch.as_tensor(inputimage).float().reshape(1, channels, image_size, image_size) #image_size * image_size image
    
    # Define formulated transformer
    transformer = TNN( config_list, model, activation_dict)
    layer_names, parameters, _, layer_outputs_dict = extract_from_pretrained.get_torchViT_learned_parameters(tnn_model, input, heads)
        
    # list to help create new variable names for each layer  
    count_list = []

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

    # Perform patching
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
    
    # Linearly transforme patched image
    out = transformer.embed_input( "patch_input" , "embed_input", model.embed_dim, W_emb, b_emb)

    #  Add CLS tokens
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

    # output constraints
    model.out_constraints = pyo.ConstraintList()
    for i in model.out.index_set():
        model.out_constraints.add(expr= model.out[i] == out[i])
        
    return model, ffn_parameter_dict, layer_outputs_dict, transformer
