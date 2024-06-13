
from omlt.neuralnet.layer import DenseLayer, InputLayer
from omlt.neuralnet.network_definition import NetworkDefinition
import numpy as np
import keras
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
def weights_to_NetDef(NN_name, input_value, model_parameters, input_bounds):
    input_seq_len = np.array(input_value).squeeze(0).shape[0]
    input_dims = np.array(input_value).squeeze(0).shape[1]
    net = []

    for dim in range(input_dims):
        print("DIM ", dim)
        
        # Define the input layer correctly
        input_layer = Input(shape=(1,input_seq_len), name="input_dim_" + str(dim))
        x = input_layer
        
        weights_list = []
        first_layer = True
        count_layers = len(model_parameters[NN_name])

        # Get weights, biases, num inputs, num outputs for each layer of FFN
        for layer_name, val in model_parameters[NN_name].items():
            count_layers -= 1
            print(layer_name, count_layers)
            
            nn_model2 = Sequential(name=NN_name+"_dim_"+str(dim))
            nn_model2.add(keras.Input(np.array([1,input_seq_len])))
            
            if "dense" in layer_name:
                if first_layer:
                    weights = np.repeat(np.expand_dims(np.array(val['W'])[dim, :], axis=0), 10, axis=0)
                    bias = np.array(val['b'])
                    n_layer_inputs, n_layer_nodes = np.shape(weights)
                    first_layer = False
                elif count_layers == 0:
                    weights = np.repeat(np.expand_dims(np.array(val['W'])[:, dim], axis=1), 10, axis=1)
                    bias = np.repeat(np.expand_dims(np.array(val['b'][dim]), axis=0), 10, axis=0)
                    n_layer_inputs, n_layer_nodes = np.shape(weights)
                else:
                    weights = np.array(val['W'])
                    bias = np.array(val['b'])
                    n_layer_inputs, n_layer_nodes = np.shape(weights)
                    
                print(np.array(val['W']).shape, np.shape(weights), np.shape(bias))
                weights_list.append([weights, bias])
                
                dense_layer = Dense(n_layer_nodes, activation='relu', name=layer_name)
                x = dense_layer(x)
                
                #nn_model2.add(Dense(n_layer_nodes, activation='relu',name=layer_name))
    

        nn_model = Model(inputs=input_layer, outputs=x, name=NN_name + "_dim_" + str(dim))
        
        for i, w in enumerate(weights_list):
            nn_model.layers[i + 1].set_weights(w)  # i+1 to skip input layer
            #nn_model2.layers[i].set_weights(w)
            
        print("Model summary", nn_model.summary())
        input_val = np.transpose(input_value, (0, 2, 1))[:, dim, :]

#         #input_value = np.expand_dims(np.array(input_value), axis=0)
        print("input value", np.expand_dims(np.array(input_val), axis=0).shape)
        predictions = nn_model.predict(np.expand_dims(np.array(input_val), axis=0))
        print("end prediction", predictions)

        # Create a new model that outputs every layer's output
        layer_outputs = [layer.output for layer in nn_model.layers]
        print(layer_outputs)
        print("input", nn_model.input)
        model_multi_output = Model(inputs=nn_model.input, outputs=layer_outputs)

        # Make predictions
        outputs = model_multi_output.predict(np.expand_dims(input_val, axis=0))
        print("model 2 prediction",outputs)
        # Format and save
        outputs_list = [output.tolist() for output in outputs]
        layer_outputs_dict = {}
        layer_names = []
        print(nn_model.layers)

        for i,layer in enumerate(nn_model.layers):
            if "dropout" in nn_model.layers[i].name:  # drop out does nothing during inference
                continue

            if nn_model.layers[i].name[-1].isnumeric():
                layer_name = nn_model.layers[i].name.rsplit('_', maxsplit=1)[0]
            else:
                layer_name = nn_model.layers[i].name
            count = layer_names.count(layer_name) + 1
            layer_outputs_dict[layer_name + '_' + str(count)] = outputs_list[i]
            layer_names.append(layer_name)
            print(i, layer_name, layer.get_weights())

        print("layer dict", layer_outputs_dict)
        net.append(keras_reader.load_keras_sequential(nn_model, unscaled_input_bounds=input_bounds))

    return net
# def weights_to_NetDef(NN_name, input_value, model_parameters,input_bounds
# ):
#     input_seq_len = np.array(input_value).squeeze(0).shape[0]
#     input_dims = np.array(input_value).squeeze(0).shape[1]
#     net = []
    
#     for dim in range(input_dims):
#         print("DIM ", dim)
#         nn_model = Sequential(name=NN_name+"_dim_"+str(dim))
#         nn_model.add(keras.Input(np.array([1,input_seq_len])))
    
#         weights_list = [ ]
#         first_layer = True
#         count_layers = len(model_parameters[NN_name])
        
#         # get weights, biases, num inputs, num outputs for each layer of FFN
#         for layer_name, val in model_parameters[NN_name].items():
#             count_layers -= 1
#             print(layer_name, count_layers)
            
#             if "dense" in layer_name:
#                 if first_layer:
#                     #print(np.array(val['W'])[dim,:])
#                     weights = np.repeat(np.expand_dims(np.array(val['W'])[dim,:],axis=0),10, axis=0)
#                     bias = np.array(val['b'])
#                     n_layer_inputs, n_layer_nodes = np.shape(weights)
#                     first_layer = False
                    
#                 elif count_layers == 0:
#                     #print(np.array(val['W'])[:,dim])
#                     weights = np.repeat(np.expand_dims(np.array(val['W'])[:,dim],axis=1),10,axis=1)
#                     bias = np.repeat(np.expand_dims(np.array(val['b'][dim]), axis=0),10, axis=0)
#                     n_layer_inputs, n_layer_nodes = np.shape(weights)
                    
#                 else:
#                     weights = np.array(val['W'])
#                     bias = np.array(val['b'])
#                     n_layer_inputs, n_layer_nodes = np.shape(weights)
                   
#                 print(np.array(val['W']).shape, np.shape(weights), np.shape(bias)) 
#                 #print(weights)
#                 #print(bias)
#                 weights_list += [[weights, bias]]
                
#                 print(n_layer_nodes)
#                 nn_model.add(Dense(n_layer_nodes, activation='relu',name=layer_name))
    
#         for i, w in enumerate(weights_list):
#             print(i)
#             nn_model.layers[i].set_weights(w)

#         print("Model summary", nn_model.summary())
#         input_val = np.transpose(input_value,(0,2,1))[:,dim,:]
#         #input_value = np.expand_dims(np.array(input_value), axis=0)#[:,:,dim]
#         #input_value = np.expand_dims(input_value, axis=0)
#         print("input value",np.expand_dims(input_val, axis=0))
#         predictions = nn_model.predict(np.expand_dims(input_val, axis=0))
#         print("end prediction",predictions)
       
#         # Create a new model that outputs every layer's output
#         layer_outputs = [layer.output for layer in nn_model.layers]
#         print(layer_outputs)
#         #print("input",nn_model.input)
#         model_multi_output = keras.models.Model(inputs=keras.Input(np.array([1,input_seq_len])), outputs=layer_outputs)
#         print(model_multi_output)
#         # Make predictions
#         outputs = model_multi_output.predict(np.expand_dims(input_val, axis=0))
#         print(outputs)
#         # format and save
#         outputs_list = [output.tolist() for output in outputs]
#         layer_outputs_dict = {}
#         layer_names = []
#         print(nn_model.layers)
#         for i in range(len(nn_model.layers)):
#             print(i)
#             if "dropout" in nn_model.layers[i].name: # drop out does nothing during inference
#                 continue

#             if nn_model.layers[i].name[-1].isnumeric():
#                 layer_name = nn_model.layers[i].name.rsplit('_', maxsplit=1)[0]
#             else:
#                 layer_name = nn_model.layers[i].name
#             count = layer_names.count(layer_name) + 1
#             layer_outputs_dict[layer_name+'_'+str(count)] = outputs_list[i]
#             layer_names += [layer_name]
        
#         print("layer dict",layer_outputs_dict)
    
#         net += [keras_reader.load_keras_sequential(nn_model, unscaled_input_bounds=input_bounds)]

#     return net
    
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
    #         nn_model.add(Dense(n_layer_nodes, activation='relu',name=layer_name))
  
    # for i, w in enumerate(weights_list):
    #     nn_model.layers[i].set_weights(w)

    # net = keras_reader.load_keras_sequential(nn_model,unscaled_input_bounds=input_bounds)
    # return net