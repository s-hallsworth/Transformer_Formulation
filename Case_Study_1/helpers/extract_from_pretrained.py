import json
import numpy as np
import keras
import torch
import os
from torch import nn
import collections

    
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
    model_activations = []
    for layer in model.layers:
        if "dropout" in layer.name:
            continue
        
        config = layer.get_config()
        if 'activation' in config:
            model_activations += [config['activation']]
        else:
            model_activations += [None]

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
            new_layer_name = 'mutli_head_attention_'+str(count_MHA)
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
            
            # if previous layer also dense, count as part of previous FFN
            if 'DENSE' in model_layers[i-1].upper() and model_activations[i-1] == model_activations[i]: 
                count_Layers += 1
                #new_layer_name = 'dense_'+str(count_Layers)
                
                dict_transformer_params[NN_name] |= { layer_name: {'W': parameters[0], 'b': parameters[1], 'activation': model_activations[i]}}
            
            # else create new ffn in dict 
            else: 
                count_Layers = 1
                count_Dense += 1
                NN_name = 'ffn_'+str(count_Dense)
                new_layer_name = NN_name
                #new_layer_name = 'dense_'+str(count_Layers)
               
                dict_transformer_params[NN_name] = {'input_shape': model_outputs[i-1].shape, 
                                                    'input': model_outputs[i-1],
                                                     layer_name: {'W': parameters[0], 'b': parameters[1], 'activation': model_activations[i]}}  
            
        layer_names += [new_layer_name]
            
    #print(layer_names) # ['layer_normalization_130', 'mutlihead_attention_65', 'layer_normalization_131', 'conv2d_42', 'conv2d_43', 'dense_70', 'dense_71']
    #print(dict_transformer_params['mutlihead_attention_1','W_q'])   
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

def get_pytorch_model_weights(model, save_json=True, file_name='.\weights.json'):
    """
    Save weights of pre-trained PyTorch model to json file with layer name appended.
    """
    
    model.eval()

    # Extract weights
    model_weights = {}
    model_bias = {}
    for name, param in model.named_parameters():
            
        if "weight" in name:
            new_name = name.split('weight')[0]
            if not new_name[-1].isalnum():
                new_name = new_name[:-1]
            model_weights[new_name] = param.detach().cpu().numpy().tolist()    
            
        if "bias" in name:
            new_name = name.split('bias')[0]
            if not new_name[-1].isalnum():
                new_name = new_name[:-1]
            model_bias[new_name] = param.detach().cpu().numpy().tolist()
            
            
        
    # Save weights
    if save_json:
        with open(file_name, 'w') as json_file:
            json.dump(model_weights, json_file)
            
        print(f"Weights of the model have been saved to {file_name}")
    
    else:
        return model_weights, model_bias
    
def get_pytorch_learned_parameters(model, enc_input, dec_input, head_size, sequence_size):
    """
    Read model parameters and store in dict with associated name. 
    ** NB: Assumes ReLU Activation function **
    """
    # src = torch.rand(enc_input.shape)
    # tgt = torch.rand(dec_input.shape)
    src = torch.as_tensor(enc_input).float()
    tgt = torch.as_tensor(dec_input).float()
    
    input_shapes = collections.OrderedDict()
    output_shapes = collections.OrderedDict()
    dict_outputs = {}

    ## Get layer input shapes
    # Function to capture the input shape
    def hook_fn(module, input, output, name):
        input_shapes[name]  = input[0].shape
        output_shapes[name] = output[0].shape
        if name in dict_outputs.keys():
            dict_outputs[name] |= {output}
        else:
            dict_outputs[name] = {output}
        
    
    # Register hooks to all layers
    for name, layer in model.named_modules():
        if "dropout" not in name:
            layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name))
    
    
    model.eval()
    with torch.no_grad():
        _ = model(src, tgt, sequence_size)
    
    layers = [i for i in list(input_shapes.keys()) if i ]

    # # Print the input shapes
    # for layer_name, shape in input_shapes.items():
    #     print(f"Layer: {layer_name}, Input shape: {shape}")
    # for layer_name, shape in output_shapes.items():
    #     print(f"Layer: {layer_name}, Input shape: {shape}")
    
    # Get weights and biases
    transformer_weights, transformer_bias = get_pytorch_model_weights(model, save_json=False)
    # layers = [val for  val in model.named_modules() if "dropout" not in val[0]]
   
    # Create dictionary with parameters
    dict_transformer_params = {}
    layer_names = []
    count_layer_names = []
    
    count_encoder_layers = 0
    count_decoder_layers = 0
    next = None
    
    # for each layer
    for i, layer in enumerate(layers):
        layer_name = layer #[0]
        new_layer_name = None
        
        if "encoder" in layer_name:
            prefix = "enc_"
            
            if "layer" in layer_name:
                prefix += "_"
            
            if layer_name.lower().endswith('self_attn'): #only parse 1 encoder/decoder layer since parameters are repeated
                count_encoder_layers += 1
                
                if count_encoder_layers > 1:
                    continue

        elif "decoder" in layer_name:
            prefix = "dec_"
            
            if "layer" in layer_name:
                prefix += "_"
                
            if layer_name.lower().endswith('self_attn'): #only parse 1 decoder layer since parameters are repeated
                count_decoder_layers += 1
                
                if count_decoder_layers > 1:
                    continue
        else:
            prefix = ""

        # store learned parameters for layers  in dict
        W_parameters = transformer_weights.get(layer_name, None)
        b_parameters = transformer_bias.get(layer_name, None)

        if 'norm' in layer_name.lower():
            name = 'layer_normalization'
            count = count_layer_names.count(prefix+name) + 1
            new_layer_name = f"{prefix+name}_{count}"
            count_layer_names.append(prefix+name)
            
            layer_names.append(new_layer_name)
            
            dict_transformer_params[(new_layer_name, 'gamma')] = W_parameters
            dict_transformer_params[(new_layer_name, 'beta')] = b_parameters
            
        if layer_name.lower().endswith('attn'):
            try: 
                # if embed dim = k, v dim --> weights concatinated in in_proj (see pytorch doc)
                W_parameters = transformer_weights.get(layer_name + ".in_proj")
                b_parameters = transformer_bias.get(layer_name + ".in_proj", None)
                
                
                if "self" in layer_name.lower():
                    emb_shape = [int(np.array(W_parameters).shape[0]/3), int(np.array(W_parameters).shape[0]/3), int(np.array(W_parameters).shape[0]/3)]
                elif "multihead" in layer_name.lower():
                    size_input_mha = input_shapes[layer_name][1]
                    size_output_encoder = input_shapes['transformer.encoder'][1]
                    
                    #cross attention calculates Q from dec inut but K, V from encoder output
                    emb_shape = [size_input_mha, size_output_encoder, size_output_encoder]
                W_q, W_k, W_v = torch.split(torch.tensor(W_parameters), emb_shape)
                if b_parameters:
                    b_q, b_k, b_v = torch.split(torch.tensor(b_parameters), emb_shape)
                else:
                    b_q = None
                    
                
            except:
                W_q = transformer_weights.get(layer_name + ".q_proj")
                W_k = transformer_weights.get(layer_name + ".k_proj")
                W_v = transformer_weights.get(layer_name + ".v_proj")
                
                b_q = transformer_bias.get(layer_name + ".q_proj", None)
                b_k = transformer_bias.get(layer_name + ".k_proj", None)
                W_v = transformer_bias.get(layer_name + ".v_proj", None)
                
            
            # set name of type of attention
            if 'self_attn' in layer_name.lower():    
                name = 'self_attention'
                count = count_layer_names.count(prefix+name) + 1
                new_layer_name = f"{prefix+name}_{count}"
                count_layer_names.append(prefix+name)
                
                layer_names.append(new_layer_name)
                
            elif 'multihead_attn' in layer_name.lower():
                name = 'mutli_head_attention'
                count = count_layer_names.count(prefix+name) + 1
                new_layer_name = f"{prefix+name}_{count}"
                count_layer_names.append(prefix+name)
                
                layer_names.append(new_layer_name)
            
            # Save in dict Q, K, V weights and biases
            dict_transformer_params[(new_layer_name, 'W_q')] =  arrange_qkv(W_q, head_size).detach().cpu().numpy().tolist()
            dict_transformer_params[(new_layer_name, 'W_k')] =  arrange_qkv(W_k, head_size).detach().cpu().numpy().tolist()
            dict_transformer_params[(new_layer_name, 'W_v')] =  arrange_qkv(W_v, head_size).detach().cpu().numpy().tolist()
            
            if not b_q is None:
                dict_transformer_params[(new_layer_name, 'b_q')] = torch.reshape(b_q, (head_size, int(b_q.shape[0]/head_size))).detach().cpu().numpy().tolist()
                dict_transformer_params[(new_layer_name, 'b_k')] = torch.reshape(b_k, (head_size, int(b_k.shape[0]/head_size))).detach().cpu().numpy().tolist()
                dict_transformer_params[(new_layer_name, 'b_v')] = torch.reshape(b_k, (head_size, int(b_v.shape[0]/head_size))).detach().cpu().numpy().tolist()
            
            out_proj_name = layer_name + ".out_proj"
            W_o = transformer_weights.get(out_proj_name)
            dict_transformer_params[(new_layer_name, 'W_o')] =  arrange_o(W_o, head_size).detach().cpu().numpy().tolist()
            
            b_o = transformer_bias.get(out_proj_name , None)
            if not b_o  is None:
                dict_transformer_params[(new_layer_name, 'b_o')] = b_o
                
            
        if 'linear' in layer_name.lower():
            # if next layer also dense, count as part of previous FFN

            if layer_name == next:
                name = 'ffn'
                count = count_layer_names.count(prefix+name)
                new_layer_name = f"{prefix+name}_{count}"
                
                dict_transformer_params[new_layer_name] |= {layer_name: {'W': W_parameters, 'b': b_parameters, 'activation': "relu"}}  
                
            elif i <  len(layers) - 1 and i > 0:
                if "linear" in layers[i+1]:
                    next = layers[i+1]
                    
                    name = 'ffn'
                    count = count_layer_names.count(prefix+name) + 1
                    new_layer_name = f"{prefix+name}_{count}"
                    count_layer_names.append(prefix+name)
                    layer_names.append(new_layer_name)
                    
                    dict_transformer_params[new_layer_name] = {'input_shape': input_shapes[layer_name], 
                                                        'input': layers[-1],
                                                        layer_name: {'W': W_parameters, 'b': b_parameters, 'activation': "relu"}} 
                
            else:
                name = 'linear'
                count = count_layer_names.count(prefix+name) + 1
                new_layer_name = f"{prefix+name}_{count}"
                count_layer_names.append(prefix+name)
                
                layer_names.append(new_layer_name)

                dict_transformer_params[(new_layer_name, 'W')] =  W_parameters
                dict_transformer_params[(new_layer_name, 'b')] =  b_parameters
                

    return layer_names, dict_transformer_params, model, [count_encoder_layers, count_decoder_layers], dict_outputs

def arrange_qkv(W, head_size):
    " reshape W to match expected shape [model_dims, headsize, qkv_dim]"
    W = torch.reshape(W, (head_size, int(W.shape[-1]/head_size), W.shape[-1]))
    W = torch.transpose(W, 1,2)
    W = torch.transpose(W, 0,1)
    return W

def arrange_o(W, head_size):
    " reshape W to match expected shape [ headsize, qkv_dim, model_dims]"
    W = torch.tensor(W)
    W = torch.reshape(W, (head_size, int(W.shape[-1]/head_size), W.shape[-1]))
    return W

def get_pytorch_intermediate_values(model, sample_input1, sample_input2, sequence_size):
    model.eval()
    sample_input1 = torch.as_tensor(sample_input1)
    sample_input2 = torch.as_tensor(sample_input2)
    
    print(sample_input1.dtype, sample_input2.dtype, type(sequence_size))
    
    # Dictionary to store the output of each layer
    layer_outputs_dict = {}
    layer_names = []

    # Function to get the output of each layer
    def output_hook_fn(module, input, output, name):
        layer_outputs_dict[name] = output

    # Register hooks for all layers
    for name, layer in model.named_modules():
        if "dropout" not in name:
            layer.register_forward_hook(lambda module, input, output, name=name: output_hook_fn(module, input, output, name))

    # Make a forward pass with the sample input
    with torch.no_grad():
        model(sample_input1, sample_input2, sequence_size)

    # # Format the layer names
    # formatted_layer_outputs = {}
    # for name, output in layer_outputs_dict.items():

    #     if name == 'encoder':
    #         layer_name = name
            
    #     elif name.endswith('self_attn'):
    #         layer_name = 'self_attention'
            
    #     elif name.endswith('mutlihead_attn'):
    #         layer_name = 'mutli_head_attention'
            
    #     elif 'linear' in name:
    #         layer_name = 'dense'
            
    #     elif 'norm' in name:
    #         layer_name = 'layer_normalization'

    #     else:
    #         continue
                
    #     count = layer_names.count(layer_name) + 1
    #     formatted_layer_outputs[f"{layer_name}_{count}"] = output
    #     layer_names.append(layer_name)

    # if file_name:
    #     with open(file_name, 'w') as file:
    #         json.dump(formatted_layer_outputs, file, indent=4)
    #     print(f"Intermediate outputs of the model have been saved to {file_name}")

    return layer_outputs_dict 