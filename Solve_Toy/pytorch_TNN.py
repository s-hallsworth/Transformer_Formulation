import torch
from extract_from_pretrained import get_pytorch_learned_parameters, get_pytorch_intermediate_values
import transformer_b as TNN
# create model
transformer_model = torch.nn.Transformer(d_model= 6, nhead=2, num_encoder_layers=1, num_decoder_layers=1,dim_feedforward=10, batch_first=True)
src = torch.rand((5, 10, 6))
tgt = torch.rand((5, 10, 6))
out = transformer_model(src, tgt)
# print(out)
# #save model
model_path = ".\\pytorch_model.pt"
torch.save(transformer_model.state_dict(), model_path)

# load model
device = torch.device('cpu')
model = torch.nn.Transformer(d_model= 6, nhead=2, num_encoder_layers=1, num_decoder_layers=1,dim_feedforward=10, batch_first=True)
model.load_state_dict(torch.load(model_path, map_location=device))
out_pre_trained = model(src, tgt)

# print("---------")
# print(out)

# Get learned parameters
#layer_names, dict_transformer_params, model = get_pytorch_learned_parameters(model, input_shape= (5, 10, 4), head_size=2)
#print(dict_transformer_params)

# Get intermediate outputs of model for testing
sample_input = torch.rand(( 10, 6)) 
sample_input2 = torch.rand(( 10, 6))
intermediate_outputs = get_pytorch_intermediate_values(model, sample_input, sample_input2)

# Create transformer 
transformer = TNN.Transformer(model, ".\\data\\toy_config_pytorch.json") 
result =  transformer.build_from_pytorch( model,  (10, 6), "PYTORCH_TESTER", sample_input, sample_input2)
for item in result:
    print(item)