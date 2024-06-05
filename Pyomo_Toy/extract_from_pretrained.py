import json
import numpy as np
import keras
import os
    
def get_weights(model_path, save_json=True, file_name="model_weights.json"):
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
    if save_json:   
        with open(file_name, 'w') as json_file:
            json.dump(model_weights, json_file)
            
        print(f"Weights of the model have been saved to {file_name}")
    
    else: 
        return model_weights, model       

def get_learned_parameters(model_path):
    """
    Read model parameters and store in dict with associated name
    """
    
    transformer_weights, model = get_weights(model_path, save_json=False)
    model_layers = [x.name for x in model.layers if "dropout" not in x.name]
    model_outputs = [layer.output for layer in model.layers if "dropout" not in layer.name]

    # create dictionary with parameters
    dict_transformer_params = {}
    layer_names = []
    count_LN = 0
    count_MHA = 0
    count_Conv2D = 0
    count_Dense = 0
    count_Layers = 0
    
    for i in range(len(model_layers)):
        layer_name = model_layers[i]
        
        try:
            parameters =  transformer_weights[layer_name]
        except:
            continue
        
        if 'LAYER_NORM' in layer_name.upper():
            count_LN += 1
            new_layer_name = 'layer_normalization_'+str(count_LN)
            if len(parameters) == 2:
                dict_transformer_params[(new_layer_name , 'gamma')] = parameters[0]
                dict_transformer_params[(new_layer_name , 'beta')] = parameters[1] 
                
            # else may contain gamma and beta parameters
        
        if 'MULTI_HEAD_ATTENTION' in layer_name.upper():  
            count_MHA += 1
            new_layer_name = 'multi_head_attention_'+str(count_MHA)
            if len(parameters) > 4: # has bias
                dict_transformer_params[(new_layer_name, 'W_q')] = parameters[0]
                dict_transformer_params[(new_layer_name, 'W_k')] = parameters[2]
                dict_transformer_params[(new_layer_name, 'W_v')] = parameters[4]
                dict_transformer_params[(new_layer_name, 'W_o')] = parameters[6]
                
                dict_transformer_params[(new_layer_name, 'b_q')] = parameters[1]
                dict_transformer_params[(new_layer_name, 'b_k')] = parameters[3]
                dict_transformer_params[(new_layer_name, 'b_v')] = parameters[5]
                dict_transformer_params[(new_layer_name, 'b_o')] = parameters[7]
                
            else: # no bias
                dict_transformer_params[(new_layer_name, 'W_q')] = parameters[0]
                dict_transformer_params[(new_layer_name, 'W_k')] = parameters[1]
                dict_transformer_params[(new_layer_name, 'W_v')] = parameters[2]
                dict_transformer_params[(new_layer_name, 'W_o')] = parameters[3]
                
        if 'CONV2D' in layer_name.upper():  
            count_Conv2D += 1
            new_layer_name = 'conv2D_'+str(count_Conv2D)
            dict_transformer_params[(new_layer_name, 'W')] = parameters[0]
            dict_transformer_params[(new_layer_name, 'b')] = parameters[1] 
            
        if 'DENSE' in layer_name.upper():  

            if 'DENSE' in model_layers[i-1].upper():
                count_Layers += 1
                #new_layer_name = 'dense_'+str(count_Layers)
                
                dict_transformer_params[NN_name] |= { layer_name: {'W': parameters[0], 'b': parameters[1]}}
               
            else:
                count_Layers = 1
                count_Dense += 1
                NN_name = 'ffn_'+str(count_Dense)
                new_layer_name = NN_name
                #new_layer_name = 'dense_'+str(count_Layers)
               
                dict_transformer_params[NN_name] = {'input_shape': model_outputs[i-1].shape, 
                                                    'input': model_outputs[i-1],
                                                     layer_name: {'W': parameters[0], 'b': parameters[1]}}  
            
        layer_names += [new_layer_name]
            
    #print(layer_names) # ['layer_normalization_130', 'multi_head_attention_65', 'layer_normalization_131', 'conv2d_42', 'conv2d_43', 'dense_70', 'dense_71']
    #print(dict_transformer_params['multi_head_attention_1','W_q'])   
    return layer_names, dict_transformer_params, model

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
    layer_outputs_dict = {}
    layer_names = []
    for i in range(len(model.layers)):
        if "dropout" in model.layers[i].name: # drop out does nothing during inference
            continue

        if model.layers[i].name[-1].isnumeric():
            layer_name = model.layers[i].name.rsplit('_', maxsplit=1)[0]
        else:
            layer_name = model.layers[i].name
        count = layer_names.count(layer_name) + 1
        layer_outputs_dict[layer_name+'_'+str(count)] = outputs_list[i]
        layer_names += [layer_name]
    
    if file_name:
        with open(file_name, 'w') as file:
            json.dump(layer_outputs_dict, file, indent=4)
            
        print(f"intermedite outputs of the model have been saved to {file_name}")
    
    return layer_outputs_dict
