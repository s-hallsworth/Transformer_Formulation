import extract_from_pretrained 
import numpy as np 
import os

"""
Get outputs from the layers of a transformer model specified by model_path
"""

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off
model_path = "..\\Transformer_Toy\\transformer_small_relu_2_TOY.keras" 
#model_path = "..\\Transformer_Toy\\transformer_small_2_TOY.keras" 
#model_path = "..\\Transformer_Toy\\transformer_small_2-no_w_mha.keras" 

# ## input to transformer
x_input_10 = [1.0, 1.10657895, 1.21388889, 1.32205882, 1.43125, 1.54166667, 1.65357143, 1.76730769, 1.88333333, 2.00227273]
u_input_10 = [0.25, 0.26315789, 0.27777778, 0.29411765, 0.3125, 0.33333333, 0.35714286, 0.38461538, 0.41666667, 0.45454545]
transformer_input = np.array([[ [x,u] for x,u in zip(x_input_10, u_input_10)]])

## read model weights and intermediate layer outputs
layer_outputs_dict = extract_from_pretrained.get_intermediate_values(model_path, transformer_input) 

# for i, j in layer_outputs_dict.items():
#     print(f"{i}:", j)

# layer_outputs_dict['layer_normalization_2']
#print(layer_outputs_dict)


