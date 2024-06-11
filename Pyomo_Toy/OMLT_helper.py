
from omlt.neuralnet.layer import DenseLayer, InputLayer
from omlt.neuralnet.network_definition import NetworkDefinition
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from omlt.io.keras import keras_reader

def weights_to_NetworkDefinition(NN_name, model_parameters,
    scaling_object=None, scaled_input_bounds=None, unscaled_input_bounds=None
):
    ### Function to create an OMLT NetworkDefinition based on a weights of a MLP
    ### RELU activation only
    
    n_inputs = model_parameters[NN_name]['input_shape'] #num input weights
    
    # init NetworkDefinition
    net = NetworkDefinition(
        scaling_object=scaling_object,
        scaled_input_bounds=scaled_input_bounds,
        unscaled_input_bounds=unscaled_input_bounds,
    )
   
    # add input layer
    prev_layer = InputLayer([1])#([n_inputs[1]], n_inputs[2]])
    net.add_layer(prev_layer)

    # get weights, biases, num inputs, num outputs for each layer of FFN
    for layer_name, val in model_parameters[NN_name].items():

        if "dense" in layer_name:
            weights = val['W']
            biases = model_parameters[NN_name][layer_name]['b']
            n_layer_inputs, n_layer_nodes = np.shape(weights)
            
            # create omlt dense layer
            dense_layer = DenseLayer(
                [n_layer_inputs],
                [1],#[n_layer_nodes],
                activation="relu",
                weights=weights,
                biases=biases,
            )
            
            # add dense layer to network definition
            net.add_layer(dense_layer)
            net.add_edge(prev_layer, dense_layer)
            prev_layer = dense_layer

    return net

def weights_to_NetDef(NN_name, input_value, model_parameters,input_bounds
):
    input_seq_len = np.array(input_value).squeeze(0).shape[0]
    input_dims = np.array(input_value).squeeze(0).shape[1]
    net = []
    
    for dim in range(input_dims):
        print("DIM ", dim)
        nn = Sequential(name=NN_name+"_dim_"+str(dim))
        nn.add(Input(np.array([1,input_seq_len])))
    
        weights_list = [ ]
        first_layer = True
        count_layers = len(model_parameters[NN_name])
        
        # get weights, biases, num inputs, num outputs for each layer of FFN
        for layer_name, val in model_parameters[NN_name].items():
            count_layers -= 1
            print(layer_name, count_layers)
            
            if "dense" in layer_name:
                if first_layer:
                    weights = np.expand_dims(np.array(val['W'])[dim,:],axis=0)
                    bias = np.array(val['b'])
                    n_layer_inputs, n_layer_nodes = np.shape(weights)
                    first_layer = False
                    
                elif count_layers == 0:
                    weights = np.expand_dims(np.array(val['W'])[:,dim],axis=1)
                    bias = np.expand_dims(np.array(val['b'][dim]), axis=0)
                    n_layer_inputs, n_layer_nodes = np.shape(weights)
                    
                else:
                    weights = np.array(val['W'])
                    bias = np.array(val['b'])
                    n_layer_inputs, n_layer_nodes = np.shape(weights)
                   
                print(np.array(val['W']).shape, np.shape(weights), np.shape(bias)) 
                
                weights_list += [[weights, bias]]
                
                print(n_layer_nodes)
                nn.add(Dense(n_layer_nodes, activation='relu',name=layer_name))
    
        for i, w in enumerate(weights_list):
            nn.layers[i].set_weights(w)

        net += [keras_reader.load_keras_sequential(nn,unscaled_input_bounds=input_bounds)]

    return net
    
    #######
    # weights_list = [ ]
    # # get weights, biases, num inputs, num outputs for each layer of FFN
    # for layer_name, val in model_parameters[NN_name].items():
    #     print(layer_name)
    #     if "dense" in layer_name:
    #         weights = np.array(val['W'])
    #         bias = np.array(val['b'])
    #         weights_list += [[weights, bias]]
    #         n_layer_inputs, n_layer_nodes = np.shape(weights)
            
    #         print("weight shape",np.shape(weights))
    #         nn.add(Dense(n_layer_nodes, activation='relu',name=layer_name))
  
    # for i, w in enumerate(weights_list):
    #     nn.layers[i].set_weights(w)

    # net = keras_reader.load_keras_sequential(nn,unscaled_input_bounds=input_bounds)
    # return net