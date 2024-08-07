import torch
from extract_from_pretrained import get_pytorch_learned_parameters, get_pytorch_intermediate_values

# create model
transformer_model = torch.nn.Transformer(d_model= 4, nhead=4, num_encoder_layers=1, num_decoder_layers=1,dim_feedforward=10, batch_first=True)
src = torch.rand((5, 10, 4))
tgt = torch.rand((5, 10, 4))
out = transformer_model(src, tgt)
# print(out)
# #save model
model_path = ".\\pytorch_model.pt"
torch.save(transformer_model.state_dict(), model_path)

# load model
device = torch.device('cpu')
model = torch.nn.Transformer(d_model= 4, nhead=4, num_encoder_layers=1, num_decoder_layers=1,dim_feedforward=10, batch_first=True)
model.load_state_dict(torch.load(model_path, map_location=device))
out_pre_trained = model(src, tgt)

print("---------")
# print(out)


layer_names, dict_transformer_params, model = get_pytorch_learned_parameters(model, input_shape= (5, 10, 4))
#print(dict_transformer_params)

sample_input = torch.rand((5, 10, 4))  # Adjust based on your model's input size
sample_input2 = torch.rand((5, 10, 4))
# Call the function
intermediate_outputs = get_pytorch_intermediate_values(model, sample_input, sample_input2)
print(layer_names)
print(intermediate_outputs)