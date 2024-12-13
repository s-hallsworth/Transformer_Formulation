
from omlt.neuralnet.layer import DenseLayer, InputLayer
from omlt.neuralnet.network_definition import NetworkDefinition
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
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
    nn = Sequential(name=NN_name)
    nn.add(Input(np.array(input_value).squeeze(0).shape))

    weights_list = [ ]
    # get weights, biases, num inputs, num outputs for each layer of FFN
    for layer_name, val in model_parameters[NN_name].items():

        if "dense" in layer_name:
            weights = np.array(val['W'])
            bias = np.array(val['b'])
            weights_list += [[weights, bias]]
            n_layer_inputs, n_layer_nodes = np.shape(weights)
            
            nn.add(Dense(n_layer_nodes, activation='relu',name=layer_name))
  
    for i, w in enumerate(weights_list):
        nn.layers[i].set_weights(w)

    net = keras_reader.load_keras_sequential(nn,unscaled_input_bounds=input_bounds)
    return net