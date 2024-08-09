import torch
from extract_from_pretrained import get_pytorch_learned_parameters, get_pytorch_intermediate_values
import transformer_b as TNN
import pyomo.environ as pyo
from omlt import OmltBlock
import convert_pyomo
from gurobipy import Model, GRB, GurobiError
from print_stats import solve_gurobipy

# create model
transformer_model = torch.nn.Transformer(d_model= 6, nhead=2, num_encoder_layers=1, num_decoder_layers=1,dim_feedforward=10, batch_first=True)
src = torch.rand((5, 10, 6))
tgt = torch.rand((5, 10, 6))
out = transformer_model(src, tgt) #src: input to encoder, tgt: input to decoder
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
sample_input = src[0]
sample_input2 = tgt[0]
intermediate_outputs = get_pytorch_intermediate_values(model, sample_input, sample_input2)

# Create transformer 
opt_model = pyo.ConcreteModel(name="opt_model_name")
transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", opt_model) 
result =  transformer.build_from_pytorch( model, sample_input, sample_input2)
print(result)

## Set objective
tnn_output = getattr( opt_model, result[-2])
opt_model.obj = pyo.Objective( expr= sum(tnn_output[i] for i in tnn_output.index_set()), sense=1) 

## Convert to gurobipy
gurobi_model, map_var = convert_pyomo.to_gurobi(opt_model)

## Solve
time_limit = 240
solve_gurobipy(gurobi_model, time_limit) ## Solve and print


if gurobi_model.status == GRB.INFEASIBLE:
        gurobi_model.computeIIS()
        gurobi_model.write("pytorch_model.ilp")