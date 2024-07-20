
from omlt.neuralnet.layer import DenseLayer, InputLayer
from omlt.neuralnet.network_definition import NetworkDefinition
import numpy as np
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def weights_to_NetDef(NN_name, input_shape, model_parameters
):
    input_shape = np.array(input_shape)
    nn = Sequential(name=NN_name)
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

    #print("model summary",nn.summary())

    # set weights for each layer
    for i, w in enumerate(weights_list):
        nn.layers[i].set_weights(w)
        
    return nn

def get_inputs_gurobipy_FNN(input_nn, output_nn, map_var):
    inputs = []
    outputs = []
    prev_input = {}
    prev_output = {}
    for index, value in map_var.items():
        
        if str(index).split('[')[0] == input_nn.name:
            
            if "," in str(index):
                try: 
                    inputs += [ [prev_input[str(index).split(',')[0]], value]]
                except:
                    prev_input[str(index).split(',')[0]] = value
            else:
                inputs += [value]
            
            
        elif str(index).split('[')[0] == output_nn.name:
            if "," in str(index):
                try: 
                    outputs += [ [prev_output[str(index).split(',')[0]], value]]
                except:
                    prev_output[str(index).split(',')[0]] = value
            else:
                outputs += [value]          
    return inputs, outputs 