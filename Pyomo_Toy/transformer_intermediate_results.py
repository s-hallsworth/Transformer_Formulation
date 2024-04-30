import extract_from_pretrained as extract_from_pretrained
import numpy as np 

# define model
model_path = "..\\Transformer_Toy\\transformer_TOY.keras"

# define input
x_input_10 = [1.0, 1.10657895, 1.21388889, 1.32205882, 1.43125, 1.54166667, 1.65357143, 1.76730769, 1.88333333, 2.00227273]
u_input_10 = [0.25, 0.26315789, 0.27777778, 0.29411765, 0.3125, 0.33333333, 0.35714286, 0.38461538, 0.41666667, 0.45454545]
input = np.array([[[x_input_10], [u_input_10]]])

# get transformer intermidate outputs
layer_outputs_dict = extract_from_pretrained.get_intermediate_values(model_path, input) #, "intermediary_results_10.json")

# get and reformat layer_normalization_130 output (first layer norm block)
layer_norm = np.array(layer_outputs_dict["layer_normalization_130"])
layer_norm_output = np.array([ [x,u] for x,u in zip(layer_norm[0][0], layer_norm[0][1])])
print(layer_norm_output.shape)