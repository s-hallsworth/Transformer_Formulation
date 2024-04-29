import json
import numpy as np

# read model parameters
with open("model_weights.json", "r") as read_file:
    transformer_weights = json.load(read_file)

# create dictionary with parameters
dict_transformer_params = {}
layer_names = []
for layer_name in transformer_weights:
    layer_names += [layer_name]
    parameters =  transformer_weights[layer_name]
    try:
        print(layer_name, np.array(parameters).shape)
    except:
        print(layer_name, len(parameters))
        
    if 'LAYER_NORM' in layer_name.upper():
        if len(parameters) == 2:
            dict_transformer_params[(layer_name, 'gamma')] = parameters[0]
            dict_transformer_params[(layer_name, 'beta')] = parameters[1] 
            
        # else may contain gamma and beta parameters
    
    if 'MULTI_HEAD_ATTENTION' in layer_name.upper():  
        for i in range( len(parameters)):
            print(layer_name, np.array(parameters[i]).shape)
            
        if len(parameters) > 4: # has bias
            dict_transformer_params[(layer_name, 'W_q')] = parameters[0]
            dict_transformer_params[(layer_name, 'W_k')] = parameters[2]
            dict_transformer_params[(layer_name, 'W_v')] = parameters[4]
            dict_transformer_params[(layer_name, 'W_o')] = parameters[6]
            
            dict_transformer_params[(layer_name, 'b_q')] = parameters[1]
            dict_transformer_params[(layer_name, 'b_k')] = parameters[3]
            dict_transformer_params[(layer_name, 'b_v')] = parameters[5]
            dict_transformer_params[(layer_name, 'b_o')] = parameters[7]
            
        else: # no bias
            dict_transformer_params[(layer_name, 'W_q')] = parameters[0]
            dict_transformer_params[(layer_name, 'W_k')] = parameters[1]
            dict_transformer_params[(layer_name, 'W_v')] = parameters[2]
            dict_transformer_params[(layer_name, 'W_o')] = parameters[3]
            
    if 'CONV2D' in layer_name.upper():  
        dict_transformer_params[(layer_name, 'W')] = parameters[0]
        dict_transformer_params[(layer_name, 'b')] = parameters[1] 
        
    if 'DENSE' in layer_name.upper():  
        dict_transformer_params[(layer_name, 'W')] = parameters[0]
        dict_transformer_params[(layer_name, 'b')] = parameters[1] 
        
        
print(layer_names)
#print(dict_transformer_params)   

# get learned params
gamma1 = dict_transformer_params['layer_normalization_130','gamma']
beta1 = dict_transformer_params['layer_normalization_130','beta']