
from omlt.neuralnet.layer import DenseLayer, InputLayer
from omlt.neuralnet.network_definition import NetworkDefinition
import numpy as np
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from omlt.io.keras import keras_reader

def weights_to_NetDef(new_name, NN_name, input_shape, model_parameters,input_bounds=None):
    """
    Converts model weights and structure into an OMLT NetworkDefinition using a Keras Sequential model.

    Args:
        new_name (str): Name for the Keras model.
        NN_name (str): Identifier for the neural network in the `model_parameters` dictionary.
        input_shape (tuple or list): Shape of the input data.
        model_parameters (dict): Dictionary containing the weights, biases, and activation functions for the layers.
        input_bounds (optional): Bounds for the input variables.

    Returns:
        NetworkDefinition: An OMLT-compatible network definition object based on the input Keras model.

    Notes:
        - Supports only dense layers with ReLU activation.
        - Transposes weight matrices where necessary to align with Keras layer expectations.
        - Uses `keras_reader.load_keras_sequential` to convert the Keras model into an OMLT NetworkDefinition.
    """

    input_shape = np.array(input_shape)
    nn = Sequential(name=new_name)
    nn.add(Input(input_shape))
    weights_list = [ ]
    
    # get weights, biases, num inputs, num outputs for each layer of FFN
    for layer_name, val in model_parameters[NN_name].items():

        if "dense" in layer_name:
            weights = np.array(val['W'])
            bias = np.array(val['b'])
            n_layer_inputs, n_layer_nodes = np.shape(weights)
                
            # Determine activation function
            if val['activation'] =='relu':
                weights_list += [[weights, bias]]
                nn.add(Dense(n_layer_nodes, activation='relu',name=layer_name))

            else:
                raise TypeError(f'Error in layer: {layer_name}. Activation function not currently supported for ', val['activation'])

    # set weights for each layer
    for i, w in enumerate(weights_list):
        nn.layers[i].set_weights(w)
        
    net = keras_reader.load_keras_sequential(nn, unscaled_input_bounds=input_bounds)
    return net

