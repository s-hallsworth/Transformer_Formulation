import json
import numpy as np
import keras
import os

def save_weights_json(model_path, file_name):
    """
    Save weights of pre-trained keras model to json file with layer name appended
    """
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off
    
    # load pre-trained model
    model = keras.models.load_model(model_path)

    # print model summary
    # print("--- Model Summary ---")
    # model.summary()

    # extract weights
    model_weights = {}

    for layer in model.layers:
        weights = layer.get_weights()
        if weights:  
            model_weights[layer.name] = [w.tolist() for w in weights]  

    #save weights    
    with open(file_name, 'w') as json_file:
        json.dump(model_weights, json_file)
        
    print(f"Weights of the model have been saved to {file_name}")
        

def get_learned_parameters(path):
    """
    Read model parameters and store in dict with associated name
    """
    with open(path, "r") as read_file:
        transformer_weights = json.load(read_file)

    # create dictionary with parameters
    dict_transformer_params = {}
    layer_names = []
    for layer_name in transformer_weights:
        layer_names += [layer_name]
        parameters =  transformer_weights[layer_name]
        # try:
        #     print(layer_name, np.array(parameters).shape)
        # except:
        #     print(layer_name, len(parameters))
            
        if 'LAYER_NORM' in layer_name.upper():
            if len(parameters) == 2:
                dict_transformer_params[(layer_name, 'gamma')] = parameters[0]
                dict_transformer_params[(layer_name, 'beta')] = parameters[1] 
                
            # else may contain gamma and beta parameters
        
        if 'MULTI_HEAD_ATTENTION' in layer_name.upper():  
                
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
            
    #print(layer_names) # ['layer_normalization_130', 'multi_head_attention_65', 'layer_normalization_131', 'conv2d_42', 'conv2d_43', 'dense_70', 'dense_71']
    #print(dict_transformer_params)   
    return layer_names, dict_transformer_params

def get_intermediate_values(model_path, sample_input, file_name=None):
    
    # load pre-trained model
    model = keras.models.load_model(model_path)

    # Create a new model that outputs every layer's output
    layer_outputs = [layer.output for layer in model.layers]
    model_multi_output = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Make predictions
    outputs = model_multi_output.predict(sample_input)

    # format and save
    outputs_list = [output.tolist() for output in outputs]
    layer_outputs_dict = {model.layers[i].name: outputs_list[i] for i in range(len(model.layers))}
    
    if file_name:
        with open(file_name, 'w') as file:
            json.dump(layer_outputs_dict, file, indent=4)
            
        print(f"intermedite outputs of the model have been saved to {file_name}")
    
    return layer_outputs_dict
