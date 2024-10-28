import json
import numpy as np
import keras
import torch
import os
from torch import nn
import collections
from torch.nn import ReLU
from transformers.activations import SiLUActivation as SiLU # clone transformers from: https://github.com/s-hallsworth/transformers.git -- > pip install -e .
from transformers.models.time_series_transformer.configuration_time_series_transformer import TimeSeriesTransformerConfig
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerForPrediction
from vit_TNN import *

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
            # if previous layer also dense, count as part of previous FFN
            if 'DENSE' in model_layers[i-1].upper(): # and model_activations[i-1] == model_activations[i]: 
                count_Layers += 1
                #new_layer_name = 'dense_'+str(count_Layers)
                
                dict_transformer_params[NN_name] |= { "dense_"+str(count_Layers): {'W': parameters[0], 'b': parameters[1], 'activation': model_activations[i]}}
            
            # else create new ffn in dict 
            else: 
                count_Layers = 1
                count_Dense += 1
                NN_name = 'ffn_'+str(count_Dense)
                new_layer_name = NN_name
                #new_layer_name = 'dense_'+str(count_Layers)
               
                dict_transformer_params[NN_name] = {'input_shape': model_outputs[i-1].shape, 
                                                    'input': model_outputs[i-1],
                                                     "dense_"+str(count_Layers): {'W': parameters[0], 'b': parameters[1], 'activation': model_activations[i]}}  
            
        layer_names += [new_layer_name]
            
    #print(layer_names) # ['layer_normalization_130', 'multihead_attention_65', 'layer_normalization_131', 'conv2d_42', 'conv2d_43', 'dense_70', 'dense_71']
    #print(dict_transformer_params['multihead_attention_1','W_q'])   
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
    
def get_pytorch_learned_parameters(model, enc_input, dec_input, num_heads, sequence_size=None):
    """
    Read model parameters and store in dict with associated name. 
    """

    src = torch.as_tensor(enc_input).float()
    enc_prefix = "enc"
    dec_prefix = "dec"
    
    input_shapes = collections.OrderedDict()
    output_shapes = collections.OrderedDict()
    dict_outputs = {}
    activations_dict = {}
    
    # Get layer input shapes
    def hook_fn(module, input, output, name):

        input_shapes[name]  = input[0].shape
        output_shapes[name]  = output[0].shape

        if name in dict_outputs.keys():
            dict_outputs[name] |= {output}
        else:
            dict_outputs[name] = {output}
            
        #Get activation functions
        if isinstance(module, nn.TransformerEncoderLayer):
            if "relu" in str(module.activation):
                activations_dict[enc_prefix] = "relu"
            else:
                activations_dict[enc_prefix] = "UNKNOWN"
                raise ValueError("Error parsing transformer model. Unrecognised activation function used in encoder")
        elif isinstance(module, nn.TransformerDecoderLayer):
            if "relu" in str(module.activation):
                activations_dict[dec_prefix] = "relu"
            else:
                activations_dict[dec_prefix] = "UNKNOWN"
                raise ValueError("Error parsing transformer model. Unrecognised activation function used in decoder")
             
    # Register hooks to all layers
    for name, layer in model.named_modules():
        if "dropout" not in name:
            layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name))
        
    # Forward pass through model
    model.eval()
    if not dec_input  is None:
        tgt = torch.as_tensor(dec_input).float()
        with torch.no_grad():
            _ = model(src, tgt, tgt.shape[0])
    else:
        with torch.no_grad():
            _ = model(src)

    # Get model layers  
    layers = [i for i in list(input_shapes.keys()) if i ]
    
    # Get weights and biases
    transformer_weights, transformer_bias = get_pytorch_model_weights(model, save_json=False)
 

    # # Print the input shapes
    # for layer_name, shape in input_shapes.items():
    #     print(f"Layer: {layer_name}, Input shape: {shape}")
    # for layer_name, shape in output_shapes.items():
    #     print(f"Layer: {layer_name}, Input shape: {shape}")
    
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
        suffix = "_"
        
        if "encoder" in layer_name:
            prefix = enc_prefix
            
            if "layer" in layer_name:
                suffix += "_"
            
            if layer_name.lower().endswith('self_attn'): #only parse 1 encoder/decoder layer since parameters are repeated
                count_encoder_layers += 1

        elif "decoder" in layer_name:
            prefix = dec_prefix
            
            if "layer" in layer_name:
                suffix += "_"
                
            if layer_name.lower().endswith('self_attn'): #only parse 1 decoder layer since parameters are repeated
                count_decoder_layers += 1
                
        else:
            prefix = ""
            suffix = ""

        # store learned parameters for layers  in dict
        W_parameters = transformer_weights.get(layer_name, None)
        b_parameters = transformer_bias.get(layer_name, None)

        if 'norm' in layer_name.lower():
            name = 'layer_normalization'
            count = count_layer_names.count(prefix+suffix+name) + 1
            new_layer_name = f"{prefix+suffix+name}_{count}"
            count_layer_names.append(prefix+suffix+name)
            
            layer_names.append(new_layer_name)
            
            dict_transformer_params[(new_layer_name, 'gamma')] = W_parameters
            dict_transformer_params[(new_layer_name, 'beta')] = b_parameters
            
        if layer_name.lower().endswith('attn'):
            try: 
                # if embed dim = k, v dim --> weights concatinated in in_proj (see pytorch doc)
                W_parameters = transformer_weights.get(layer_name + ".in_proj")
                b_parameters = transformer_bias.get(layer_name + ".in_proj", None)
                
                # print("weight shape: ", np.array(W_parameters).shape)
                # print("bias shape: ",np.array(b_parameters).shape)
                
                if "self" in layer_name.lower():
                    emb_shape = [int(np.array(W_parameters).shape[0]/3), int(np.array(W_parameters).shape[0]/3), int(np.array(W_parameters).shape[0]/3)]
                elif "multihead" in layer_name.lower():
                    size_input_mha = input_shapes[layer_name][1]
                    size_output_encoder = input_shapes['transformer.encoder'][1]
                    
                    #cross attention calculates Q from dec inut but K, V from encoder output
                    emb_shape = [size_input_mha, size_output_encoder, size_output_encoder]
                W_q, W_k, W_v = torch.split(torch.tensor(W_parameters), emb_shape)
                # print("weight shape: ", W_q.shape)
                if b_parameters:
                    b_q, b_k, b_v = torch.split(torch.tensor(b_parameters), emb_shape)
                else:
                    b_q = None
                # print("bias shape: ",  b_q.shape)
                    
                
            except:
                W_q = torch.tensor(transformer_weights.get(layer_name + ".q_proj"))
                W_k = torch.tensor(transformer_weights.get(layer_name + ".k_proj"))
                W_v = torch.tensor(transformer_weights.get(layer_name + ".v_proj"))
                
                b_q = torch.tensor(transformer_bias.get(layer_name + ".q_proj", None))
                b_k = torch.tensor(transformer_bias.get(layer_name + ".k_proj", None))
                b_v = torch.tensor(transformer_bias.get(layer_name + ".v_proj", None))
                
            
            # set name of type of attention
            if 'self_attn' in layer_name.lower():    
                name = 'self_attention'
                count = count_layer_names.count(prefix+suffix+name) + 1
                new_layer_name = f"{prefix+suffix+name}_{count}"
                count_layer_names.append(prefix+suffix+name)
                
                layer_names.append(new_layer_name)
                
            elif 'multihead_attn' in layer_name.lower():
                name = 'multi_head_attention'
                count = count_layer_names.count(prefix+suffix+name) + 1
                new_layer_name = f"{prefix+suffix+name}_{count}"
                count_layer_names.append(prefix+suffix+name)
                
                layer_names.append(new_layer_name)
            
            # Save in dict Q, K, V weights and biases
            dict_transformer_params[(new_layer_name, 'W_q')] =  arrange_qkv(W_q, num_heads).detach().cpu().numpy().tolist()
            dict_transformer_params[(new_layer_name, 'W_k')] =  arrange_qkv(W_k, num_heads).detach().cpu().numpy().tolist()
            dict_transformer_params[(new_layer_name, 'W_v')] =  arrange_qkv(W_v, num_heads).detach().cpu().numpy().tolist()
            
            if not b_q is None:
                dict_transformer_params[(new_layer_name, 'b_q')] = torch.reshape(b_q, (num_heads, int(b_q.shape[0]/num_heads))).detach().cpu().numpy().tolist()
                dict_transformer_params[(new_layer_name, 'b_k')] = torch.reshape(b_k, (num_heads, int(b_k.shape[0]/num_heads))).detach().cpu().numpy().tolist()
                dict_transformer_params[(new_layer_name, 'b_v')] = torch.reshape(b_v, (num_heads, int(b_v.shape[0]/num_heads))).detach().cpu().numpy().tolist()
            
            out_proj_name = layer_name + ".out_proj"
            W_o = transformer_weights.get(out_proj_name)
            dict_transformer_params[(new_layer_name, 'W_o')] =  arrange_o(W_o, num_heads).detach().cpu().numpy().tolist()
            
            b_o = transformer_bias.get(out_proj_name , None)
            if not b_o  is None:
                dict_transformer_params[(new_layer_name, 'b_o')] = b_o
                
            
        if 'linear' in layer_name.lower():
            # if next layer also dense, count as part of previous FFN
            
            if layer_name == next:
                # set activation function
                activation = None #second linear layer of enc/dec has no activation
                
                # set name
                name = 'ffn'
                count = count_layer_names.count(prefix+suffix+name)
                new_layer_name = f"{prefix+suffix+name}_{count}"
                
                # store layer info
                dict_transformer_params[new_layer_name] |= {layer_name: {'W': W_parameters, 'b': b_parameters, 'activation': activation}}  
                
            elif i <  len(layers) - 1 and i > 0:
                # get activation function
                try:
                    activation = activations_dict[prefix]
                except:
                    raise ValueError("Activation function for feed forward neural network not found.")
                
                # save layer information
                if "linear" in layers[i+1]:
                    next = layers[i+1]
                    
                    name = 'ffn'
                    count = count_layer_names.count(prefix+suffix+name) + 1
                    new_layer_name = f"{prefix+suffix+name}_{count}"
                    count_layer_names.append(prefix+suffix+name)
                    layer_names.append(new_layer_name)
                    
                    dict_transformer_params[new_layer_name] = {'input_shape': np.array(input_shapes[layer_name]), 
                                                        'input': layers[-1],
                                                        layer_name: {'W': W_parameters, 'b': b_parameters, 'activation': activation} }
                
            else:
                name = 'linear'
                count = count_layer_names.count(prefix+suffix+name) + 1
                new_layer_name = f"{prefix+suffix+name}_{count}"
                count_layer_names.append(prefix+name)
                
                layer_names.append(new_layer_name)

                dict_transformer_params[(new_layer_name, 'W')] =  W_parameters
                dict_transformer_params[(new_layer_name, 'b')] =  b_parameters
                
        print(layer, new_layer_name)
    return layer_names, dict_transformer_params, model, [count_encoder_layers, count_decoder_layers], dict_outputs

def get_ViT_model_weights(model, save_json=True, file_name='.\weights.json'):
    """
    Save weights of pre-trained PyTorch model to json file with layer name appended.
    """
    
    model.eval()

    # Extract weights
    model_weights = {}
    model_bias = {}
    cls_token = []
    pos_embedding = []
    
    for name, param in model.named_parameters():
        
        if "cls" in name:
            cls_token = param.squeeze(0).squeeze(0).tolist()
        
        elif "pos_embedding" in name:
            pos_embedding  = param.squeeze(0).tolist()
            
        else:
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
        return model_weights, model_bias, cls_token, pos_embedding

def get_torchViT_learned_parameters(model, enc_input, num_heads):
    """
    Read model parameters and store in dict with associated name. 
    """
    if not torch.is_tensor(enc_input):
        src = torch.as_tensor(enc_input).float()
    else:
        src = enc_input
    input_shapes = collections.OrderedDict()
    output_shapes = collections.OrderedDict()
    dict_outputs = {}
    activations_dict = {}
    count_layer_names = []
    map = collections.OrderedDict()
    
    
    # Get layer input shapes
    def hook_fn(module, input, output, name):

        input_shapes[name]  = input[0].shape
        output_shapes[name]  = output[0].shape

        if name in dict_outputs.keys():
            dict_outputs[name] |= {output}
        else:
            dict_outputs[name] = {output}
            
        # Get mappings
        new_name = None
        if isinstance(module, nn.LayerNorm):
            new_name = "layer_normalization"
        elif "qkv" in name:
            new_name = "self_attention"
        elif isinstance(module, nn.ReLU):
            new_name = "relu"
        elif isinstance(module, nn.Linear):
            new_name = "linear"
        elif "pos_embedding" in name:
            new_name = "pos_embedding"
        if not new_name is None:
            count = count_layer_names.count(new_name) + 1
            map[f"{new_name}_{count}"]= name
            count_layer_names.append(new_name)

    # Register hooks to all layers
    for name, layer in model.named_modules():
        if "dropout" not in name:
            layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name))
        
    # Forward pass through model
    model.eval()
    with torch.no_grad():
        _ = model(src)

    # Get model layers  
    layers = [i for i in list(input_shapes.keys()) if i ]
    
    # Get weights and biases
    transformer_weights, transformer_bias, cls_token, pos_embedding = get_ViT_model_weights(model, save_json=False)
    
    # print(transformer_weights.keys())
    # print(transformer_bias.keys())
    
    # for k,v in map.items():
    #     print(f"{k}: {v}")

    # # Print the input shapes
    # for layer_name, shape in input_shapes.items():
    #     print(f"Layer: {layer_name}, Input shape: {shape}")
    # for layer_name, shape in output_shapes.items():
    #     print(f"Layer: {layer_name}, Input shape: {shape}")
    
    # Create dictionary with parameters
    dict_transformer_params = {}
    layer_names = []
    count_layer_names = []
    next = None
    
    #save cls token and pos embedding learned params:
    dict_transformer_params['cls_token'] = cls_token
    dict_transformer_params['pos_embedding'] = pos_embedding
    
    # for each layer
    for i, layer in enumerate(map.keys()):
        layer_name = map[layer]
        new_layer_name = None
        prefix = ""
        suffix = ""

        # store learned parameters for layers  in dict
        W_parameters = transformer_weights.get(layer_name, None)
        b_parameters = transformer_bias.get(layer_name, None)

        if 'norm' in layer.lower():
            name = 'layer_normalization'
            count = count_layer_names.count(prefix+suffix+name) + 1
            new_layer_name = f"{prefix+suffix+name}_{count}"
            count_layer_names.append(prefix+suffix+name)

            dict_transformer_params[(new_layer_name, 'gamma')] = W_parameters
            dict_transformer_params[(new_layer_name, 'beta')] = b_parameters
            
        elif "attention" in layer.lower():

            # if embed dim = k, v dim --> weights concatinated in in_proj (see pytorch doc)
            W_parameters = transformer_weights.get(layer_name )
            b_parameters = transformer_bias.get(layer_name, None)
            
            print("weight shape: ", np.array(W_parameters).shape)
            # print("bias shape: ",np.array(b_parameters).shape)
            
            emb_shape = [int(np.array(W_parameters).shape[0]/3), int(np.array(W_parameters).shape[0]/3), int(np.array(W_parameters).shape[0]/3)]
            W_parameters = torch.tensor(W_parameters)
            W_q, W_k, W_v = torch.split(W_parameters, emb_shape, dim=0)
            
            from einops import rearrange
            W_q = rearrange(W_q, '(h k) d-> d h k', h=num_heads)
            W_k = rearrange(W_k, '(h k) d -> d h k', h=num_heads)
            W_v = rearrange(W_v, '(h k) d -> d h k', h=num_heads)
            
            print("weight shape: ", W_q.shape,  W_k.shape,  W_v.shape)
            if not b_parameters is None:
                b_q, b_k, b_v = torch.split(torch.tensor(b_parameters), emb_shape)

                print("bias shape: ",  b_q.shape)
                
            
            # set name of type of attention   
            name = 'self_attention'
            count = count_layer_names.count(prefix+suffix+name) + 1
            new_layer_name = f"{prefix+suffix+name}_{count}"
            count_layer_names.append(prefix+suffix+name)
            
            layer_names.append(new_layer_name)

            # Save in dict Q, K, V weights and biases
            dict_transformer_params[(new_layer_name, 'W_q')] =  W_q.detach().cpu().numpy().tolist()
            dict_transformer_params[(new_layer_name, 'W_k')] =  W_k.detach().cpu().numpy().tolist()
            dict_transformer_params[(new_layer_name, 'W_v')] =  W_v.detach().cpu().numpy().tolist()
            if not b_parameters is None:
                dict_transformer_params[(new_layer_name, 'b_q')] = torch.reshape(b_q, (num_heads, int(b_q.shape[0]/num_heads))).detach().cpu().numpy().tolist()
                dict_transformer_params[(new_layer_name, 'b_k')] = torch.reshape(b_k, (num_heads, int(b_k.shape[0]/num_heads))).detach().cpu().numpy().tolist()
                dict_transformer_params[(new_layer_name, 'b_v')] = torch.reshape(b_v, (num_heads, int(b_v.shape[0]/num_heads))).detach().cpu().numpy().tolist()
            else:
                dict_transformer_params[(new_layer_name, 'b_q')] = None
                dict_transformer_params[(new_layer_name, 'b_k')] = None
                dict_transformer_params[(new_layer_name, 'b_v')] = None
            
            out_proj_name = list(map.values())[i+1]
            print(list(map.values())[i+1])
            W_o = transformer_weights.get(out_proj_name, None) 
            W_o = torch.tensor(W_o) #.view(model_dims,  qkv_dim , num_heads).permute(2,1,0) #(d, k*h)  -->(h,k,d)
            W_o = rearrange(W_o, 'd (h k) -> h k d', h=num_heads)
            print("W_o shape", np.array(W_o).shape)
            dict_transformer_params[(new_layer_name, 'W_o')] =   W_o.detach().cpu().numpy().tolist()
            
            b_o = transformer_bias.get(out_proj_name , None)
            print("b_o shape", np.array(b_o).shape)
            if not b_o  is None:
                dict_transformer_params[(new_layer_name, 'b_o')] = b_o
                
            
        elif 'linear' in layer.lower():
            # if next layer also dense, count as part of previous FFN
            if 'relu' in list(map.keys())[i-1]:
                # set activation function
                activation = None
                
                # set name
                name = 'ffn'
                count = count_layer_names.count(prefix+suffix+name)
                new_layer_name = f"{prefix+suffix+name}_{count}"
                
                # store layer info
                dict_transformer_params[new_layer_name] |= {layer_name: {'W': W_parameters, 'b': b_parameters, 'activation': activation}}  
                
            elif list(map.keys())[-1] != layer:
                if "relu" in list(map.keys())[i+1]:
                    # get activation function
                    activation = 'relu'
                    
                    # # save layer information
                    # if "linear" in layers[i+1]:
                    #     next = layers[i+1]
                        
                    name = 'ffn'
                    count = count_layer_names.count(prefix+suffix+name) + 1
                    new_layer_name = f"{prefix+suffix+name}_{count}"
                    count_layer_names.append(prefix+suffix+name)
                    layer_names.append(new_layer_name)
                    
                    dict_transformer_params[new_layer_name] = {'input_shape': np.array(input_shapes[layer_name]), 
                                                        'input': layers[-1],
                                                        layer_name: {'W': W_parameters, 'b': b_parameters, 'activation': activation} }
                else:
                    name = 'linear'
                    count = count_layer_names.count(prefix+suffix+name) + 1
                    new_layer_name = f"{prefix+suffix+name}_{count}"
                    count_layer_names.append(prefix+name)
                    
                    layer_names.append(new_layer_name)

                    dict_transformer_params[(new_layer_name, 'W')] =  W_parameters
                    dict_transformer_params[(new_layer_name, 'b')] =  b_parameters
            else:
                name = 'linear'
                count = count_layer_names.count(prefix+suffix+name) + 1
                new_layer_name = f"{prefix+suffix+name}_{count}"
                count_layer_names.append(prefix+name)
                
                layer_names.append(new_layer_name)

                dict_transformer_params[(new_layer_name, 'W')] =  W_parameters
                dict_transformer_params[(new_layer_name, 'b')] =  b_parameters
                
        print(layer, layer_name, new_layer_name)
    return layer_names, dict_transformer_params, model, dict_outputs


def get_hugging_learned_parameters(model, enc_input, dec_input, num_heads, hugging_face_dict):
    """
    Read model parameters and store in dict with associated name. 
    """

    src = torch.as_tensor(enc_input).float()
    tgt = torch.as_tensor(dec_input).float()
    enc_prefix = "enc"
    dec_prefix = "dec"
    
    input_shapes = collections.OrderedDict()
    output_shapes = collections.OrderedDict()
    dict_outputs = {}
    activations_dict = {}
    
    # Get layer input shapes
    def hook_fn(module, input, output, name, layer):

        if len(list(input)) > 0:
            if not isinstance(input[0], torch.Size):
                input_shapes[name]  = input[0].shape
            else:
                input_shapes[name]  = list(input[0])
            
        if len(list(output)) > 0 and isinstance(output, torch.Tensor):
            if not isinstance(output[0], torch.Size):
                output_shapes[name]  = output[0].shape
            else:
                output_shapes[name]  = list(output[0])
                
            if name in dict_outputs.keys():
                dict_outputs[name] |= {output}
            else:
                dict_outputs[name] = {output}
            
        #Get activation functions
        if "activation" in name:
            if isinstance(module, ReLU):
                activations_dict[enc_prefix] = "relu"
                activations_dict[dec_prefix] = "relu"
            elif isinstance(module, SiLU):
                activations_dict[enc_prefix] = "silu"
                activations_dict[dec_prefix] = "silu"
            else:
                activations_dict[enc_prefix] = "UNKNOWN"
                activations_dict[dec_prefix] = "UNKNOWN"
                raise ValueError("Error parsing transformer model. Unrecognised activation function used in decoder")
            
    # Register hooks to all layers
    hook_names = ["linear", "encoder", "decoder", "layer", "attention", "scaler", "embedding","proj","regression"]
    for name, layer in model.named_modules():
        if any(value in name.lower() for value in hook_names):
            layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name, layer))
        
    # Forward pass through model
    model.eval()
    with torch.no_grad():
        _ = model.generate(past_values = hugging_face_dict["past_values"] , 
                past_time_features = hugging_face_dict["past_time_features"],
                past_observed_mask = hugging_face_dict["past_observed_mask"],
                future_time_features = hugging_face_dict["future_time_features"])
        
    # Get model layers  
    layers = [i for i in list(input_shapes.keys()) if i ]
    
    # Get weights and biases
    transformer_weights, transformer_bias = get_pytorch_model_weights(model, save_json=False)

    # # Print the input shapes
    # for layer_name, shape in input_shapes.items():
    #     print(f"Layer: {layer_name}, Input shape: {shape}")
    # for layer_name, shape in output_shapes.items():
    #     print(f"Layer: {layer_name}, Input shape: {shape}")
    
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
        suffix = "_"
        
        if "activation" in layer_name:
            continue
        
        if "decoder" in layer_name:
            prefix = dec_prefix
            
            if "layer" in layer_name:
                suffix += "_"
                
            if layer_name.lower().endswith('self_attn'): #only parse 1 decoder layer since parameters are repeated
                count_decoder_layers += 1

        elif "encoder" in layer_name:
            prefix = enc_prefix
            
            if "layer" in layer_name:
                suffix += "_"
            
            if layer_name.lower().endswith('self_attn'): #only parse 1 encoder/decoder layer since parameters are repeated
                count_encoder_layers += 1

        else:
            prefix = ""
            suffix = ""

        # store learned parameters for layers  in dict
        W_parameters = transformer_weights.get(layer_name, None)
        b_parameters = transformer_bias.get(layer_name, None)

        if 'norm' in layer_name.lower():
            name = 'layer_normalization'
            count = count_layer_names.count(prefix+suffix+name) + 1
            new_layer_name = f"{prefix+suffix+name}_{count}"
            count_layer_names.append(prefix+suffix+name)
            
            layer_names.append(new_layer_name)
            
            dict_transformer_params[(new_layer_name, 'gamma')] = W_parameters
            dict_transformer_params[(new_layer_name, 'beta')] = b_parameters
            
        elif layer_name.endswith("attn.out_proj"):
            layer_name = layer_name.split('.out_proj')[0]
            
            W_q = torch.tensor(transformer_weights.get(layer_name + ".q_proj"))
            W_k = torch.tensor(transformer_weights.get(layer_name + ".k_proj"))
            W_v = torch.tensor(transformer_weights.get(layer_name + ".v_proj"))
            
            b_q = torch.tensor(transformer_bias.get(layer_name + ".q_proj", None))
            b_k = torch.tensor(transformer_bias.get(layer_name + ".k_proj", None))
            b_v = torch.tensor(transformer_bias.get(layer_name + ".v_proj", None))
                
            # set name of type of attention
            if 'self_attn' in layer_name.lower():    
                name = 'self_attention'
                count = count_layer_names.count(prefix+suffix+name) + 1
                new_layer_name = f"{prefix+suffix+name}_{count}"
                count_layer_names.append(prefix+suffix+name)
                
                layer_names.append(new_layer_name)
                
            elif "encoder_attn" in layer_name.lower():
                name = 'multi_head_attention'
                count = count_layer_names.count(prefix+suffix+name) + 1
                new_layer_name = f"{prefix+suffix+name}_{count}"
                count_layer_names.append(prefix+suffix+name)
                
                layer_names.append(new_layer_name)
            
            # Save in dict Q, K, V weights and biases
            dict_transformer_params[(new_layer_name, 'W_q')] =  arrange_qkv(W_q, num_heads).detach().cpu().numpy().tolist()
            dict_transformer_params[(new_layer_name, 'W_k')] =  arrange_qkv(W_k, num_heads).detach().cpu().numpy().tolist()
            dict_transformer_params[(new_layer_name, 'W_v')] =  arrange_qkv(W_v, num_heads).detach().cpu().numpy().tolist()
            
            if not b_q is None:
                dict_transformer_params[(new_layer_name, 'b_q')] = torch.reshape(b_q, (num_heads, int(b_q.shape[0]/num_heads))).detach().cpu().numpy().tolist()
                dict_transformer_params[(new_layer_name, 'b_k')] = torch.reshape(b_k, (num_heads, int(b_k.shape[0]/num_heads))).detach().cpu().numpy().tolist()
                dict_transformer_params[(new_layer_name, 'b_v')] = torch.reshape(b_v, (num_heads, int(b_v.shape[0]/num_heads))).detach().cpu().numpy().tolist()
            
            out_proj_name = layer_name + ".out_proj"
            W_o = transformer_weights.get(out_proj_name)
            dict_transformer_params[(new_layer_name, 'W_o')] =  arrange_o(W_o, num_heads).detach().cpu().numpy().tolist()
            
            b_o = transformer_bias.get(out_proj_name , None)
            if not b_o  is None:
                dict_transformer_params[(new_layer_name, 'b_o')] = b_o
                
                
        elif "positions" in  layer_name.lower():
            name = 'pos_encoding'
            count = count_layer_names.count(prefix+suffix+name) + 1
            new_layer_name = f"{prefix+suffix+name}_{count}"
            count_layer_names.append(prefix+suffix+name)
            
            layer_names.append(new_layer_name)
            dict_transformer_params[(new_layer_name, 'b')] = W_parameters  
            
            
        elif 'fc' in layer_name.lower():
            if i > 1:
                # get activation function
                if "activation" in layers[i+1]:
                    activation = activations_dict[prefix]
                else:
                    activation = None #second linear layer of enc/dec has no activation
                    
                if not ("fc" in layers[i-1]) and not ("activation" in layers[i-1] and "fc" in layers[i-2]) : # create new ffn
                    name = 'ffn'
                    count = count_layer_names.count(prefix+suffix+name) + 1
                    new_layer_name = f"{prefix+suffix+name}_{count}"
                    count_layer_names.append(prefix+suffix+name)
                    layer_names.append(new_layer_name)
                    
                    dict_transformer_params[new_layer_name] = {'input_shape': np.array(input_shapes[layer_name]), 
                                                        'input': layers[-1],
                                                        layer_name: {'W': W_parameters, 'b': b_parameters, 'activation': activation} }
                else: #add to previous ffn
                    # set name
                    name = 'ffn'
                    count = count_layer_names.count(prefix+suffix+name)
                    new_layer_name = f"{prefix+suffix+name}_{count}"
                    
                    # store layer info
                    dict_transformer_params[new_layer_name] |= {layer_name: {'W': W_parameters, 'b': b_parameters, 'activation': activation}}  
                    
            else: # create new ffn 
                name = 'ffn'
                count = count_layer_names.count(prefix+suffix+name) + 1
                new_layer_name = f"{prefix+suffix+name}_{count}"
                count_layer_names.append(prefix+suffix+name)
                layer_names.append(new_layer_name)
                
                dict_transformer_params[new_layer_name] = {'input_shape': np.array(input_shapes[layer_name]), 
                                                    'input': layers[-1],
                                                    layer_name: {'W': W_parameters, 'b': b_parameters, 'activation': activation} }
                    
        elif "embedding." in layer_name.lower() or "regression" in layer_name.lower() or "projection.proj." in layer_name.lower():
            name = 'linear'
            count = count_layer_names.count(prefix+suffix+name) + 1
            new_layer_name = f"{prefix+suffix+name}_{count}"
            count_layer_names.append(prefix+name)
            
            layer_names.append(new_layer_name)

            dict_transformer_params[(new_layer_name, 'W')] =  W_parameters
            dict_transformer_params[(new_layer_name, 'b')] =  b_parameters
                
        print(layer, new_layer_name)
    #print(layer_names)
    return layer_names, dict_transformer_params, model, [count_encoder_layers, count_decoder_layers], dict_outputs

def arrange_qkv(W, num_heads):
    " reshape W to match expected shape when reading weights [model_dims, num heads * qkv_dim] --> [model_dims, num heads, qkv_dim]"
    model_dims = W.shape[-1]
    qkv_dim = int(model_dims/num_heads)
    W = W.view(num_heads, qkv_dim, model_dims) #[h,k,d]
    return W.permute(2,0,1) #[d,h,k]

def arrange_o(W, num_heads):
    " reshape W to match expected shape [ num heads, qkv_dim, model_dims]"
    W = torch.tensor(W) # [d, h*k]
    model_dims = W.shape[-1]
    qkv_dim = int(model_dims/num_heads)
    
    W = W.view(model_dims, num_heads, qkv_dim) #[d,h,k]
    return W.permute(1,2,0) #[h, k, d]

def get_pytorch_intermediate_values(model, sample_input1, sample_input2, sequence_size):
    model.eval()
    sample_input1 = torch.as_tensor(sample_input1)
    sample_input2 = torch.as_tensor(sample_input2)
    
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
            
    #     elif name.endswith('multihead_attn'):
    #         layer_name = 'multi_head_attention'
            
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