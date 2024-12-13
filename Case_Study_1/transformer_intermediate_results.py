from helpers import extract_from_pretrained
import numpy as np 
import os
#from data_gen import gen_x_u
"""
Get outputs from the layers of a transformer model specified by model_path
"""

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off
model_path = "..\\Transformer_Toy\\transformer_small_relu_2_TOY.keras" 

def generate_layer_outputs(model_path, define=True, x_input=None, u_input=None):

    ## input to transformer
    if define:
        x_input = x_input[0:10]
        u_input = u_input[0:10]
        U_extreme = 5*u_input
        transformer_input = np.array([[ [x,u] for x,u in zip(x_input, u_input)]])
    else:
        x_input_10 = [1.0, 1.10657895, 1.21388889, 1.32205882, 1.43125, 1.54166667, 1.65357143, 1.76730769, 1.88333333, 2.00227273]
        u_input_10 = [0.25, 0.26315789, 0.27777778, 0.29411765, 0.3125, 0.33333333, 0.35714286, 0.38461538, 0.41666667, 0.45454545]
        
        transformer_input = np.array([[ [x,u] for x,u in zip(x_input_10, u_input_10)]])


    print(transformer_input)

    ## read model weights and intermediate layer outputs
    layer_outputs_dict = extract_from_pretrained.get_intermediate_values(model_path, transformer_input) 

    # output= []
    # output += layer_outputs_dict["dense_4"]
    
    # x_input = tps.gen_x[0, -window+1 : -window + seq_len + 1]
    # u_input = tps.gen_u[0, -window+1 : -window + seq_len + 1]
    
    
    
    # transformer_input = np.array([[ [x,u] for x,u in zip(x_input, u_input)]])
    # layer_outputs_dict = extract_from_pretrained.get_intermediate_values(model_path, transformer_input) 

    # output += layer_outputs_dict["dense_4"]
    # print(output)
    # for i, j in layer_outputs_dict.items():
    #     print(f"{i}:", j)

    # layer_outputs_dict['layer_normalization_2']
    #print(layer_outputs_dict)
    
    return layer_outputs_dict



#model_path = ".\\TNN_enc_0002.keras"
# hyper_params = '.\\data\\toy_config_relu_10.json' 
# T = 9000 # time steps
# seq_len = 10
# pred_len = 2
# window = seq_len + pred_len
# model = tps.setup_toy( T, seq_len, pred_len, model_path, hyper_params)

# gen_x, gen_u, _,_ = gen_x_u(9000)
# output = generate_layer_outputs(model_path, True, gen_x[0], gen_u[0])
# print(output["dense_4"])


def generate_TNN_outputs(model_path, input_data):

    transformer_input = np.array([[ [x,u] for x,u in zip(input_data[0], input_data[1])]])


    layer_outputs_dict = extract_from_pretrained.get_intermediate_values(".\\TNN_enc_0002.keras", transformer_input)
    output = layer_outputs_dict["dense_4"]
    return output

