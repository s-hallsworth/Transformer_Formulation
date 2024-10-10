import torch
import numpy as np
#from helpers.extract_from_pretrained import get_pytorch_learned_parameters, get_pytorch_intermediate_values
import transformer_b_flag_cuts as TNN
import pyomo.environ as pyo
from omlt import OmltBlock
import helpers.convert_pyomo
from gurobipy import Model, GRB, GurobiError
from helpers.print_stats import solve_gurobipy
from torchinfo import summary
from torch.nn import Transformer
from helpers import extract_from_pretrained

import transformers, sys
sys.modules['transformers.src'] = transformers
sys.modules['transformers.src.transformers'] = transformers
from transformers.models.time_series_transformer.configuration_time_series_transformer import TimeSeriesTransformerConfig
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerForPrediction
# cloned transformers from: https://github.com/s-hallsworth/transformers.git


"""
    Case Study Explanation here
"""

# import trained transformer model

train_tnn_path = ".\\trained_transformer\\case_study\\model_TimeSeriesTransformer_final.pth"
#train_tnn_path = ".\\trained_transformer\\pytorch_model.pt"
#model = torch.nn.Transformer()

# Model Configuration
device = "cpu"
NUMBER_OF_POINTS = 8
CONTEXT_LENGTH = 3
data_files = ["T", "P", "CO", "CO2", "H2", "CH4", "CH3OH", "H2O", "N2"]

config = TimeSeriesTransformerConfig(
    prediction_length=NUMBER_OF_POINTS,
)
tnn_model = TimeSeriesTransformerForPrediction(config).to(device)

# model = TimeSeriesTransformerForPrediction.from_pretrained(train_tnn_path).to(device)
#model.load_state_dict(torch.load(train_tnn_path, weights_only=False, map_location=torch.device('cpu')))

tnn_model = torch.load(train_tnn_path, weights_only=False, map_location=torch.device('cpu'))
tnn_model.config.prediction_length = NUMBER_OF_POINTS
tnn_model.config.context_length=3
tnn_model.config.embedding_dimension=60
tnn_model.config.scaling=False
tnn_model.config.lags_sequence=[0]
tnn_model.config.num_time_features=1
tnn_model.config.input_size=len(data_files)
tnn_model.config.num_parallel_samples=1
#loss="mse"
# print(type(tnn_model), tnn_model)
#print(summary(tnn_model))
#torch_model = get_torch_model(train_tnn_path)

src = torch.ones(1, 1, len(data_files)) #input 1 point
tgt = torch.ones(1,  NUMBER_OF_POINTS, len(data_files)) # predict 
bounds_target = (-10,10)
L_t = 8.0               # [m] length of the reactor

z = np.linspace(0, L_t, NUMBER_OF_POINTS).reshape(-1,1,1) / L_t
z = torch.from_numpy(z).to(device).permute(1, 0, 2)

past_time_features =  z[:, 0:1].repeat(src.size(0), CONTEXT_LENGTH, 1).to(device).float()#torch.zeros_like(torch.linspace(-1, 0, CONTEXT_LENGTH).reshape(1, -1, 1).repeat(x_batch.size(0), 1, 1)).to(device)
future_time_features = z.repeat(src.size(0), 1, 1).to(device).float() #torch.zeros_like(y_batch[..., 0]).unsqueeze(-1).to(device)
past_values = src.repeat(1, CONTEXT_LENGTH, 1).to(device)
past_observed_mask = torch.zeros_like(past_values).to(device)
past_observed_mask[:, -1:, :] = 1

print(past_values.shape)
print(past_time_features.shape)
print(past_observed_mask.shape)
print(future_time_features.shape)
print(tgt.shape)

hugging_face_dict = {}
hugging_face_dict["past_values"] =  past_values
hugging_face_dict["past_time_features"] = past_time_features
hugging_face_dict["past_observed_mask"] = past_observed_mask
hugging_face_dict["future_time_features"] = future_time_features



# create optimization transformer
opt_model = pyo.ConcreteModel(name="(Reactor_TNN)")

# define case study problem
src = src.repeat(CONTEXT_LENGTH,1)
src = torch.nn.functional.pad( src, (0,28 - len(data_files)), "constant", 0)
tgt = torch.nn.functional.pad( tgt, (0,28 - len(data_files)), "constant", 0)
padding = list(range(28 - len(data_files)))
dims = data_files + padding

space =  np.linspace(0, L_t, NUMBER_OF_POINTS)/ L_t
start_time = space[0] - (CONTEXT_LENGTH - 1) * (space[1]-space[0])
enc_space = np.linspace(start_time, space[0], CONTEXT_LENGTH)
opt_model.enc_space = pyo.Set(initialize=enc_space)
opt_model.dec_space = pyo.Set(initialize=space)
opt_model.dims = pyo.Set(initialize=dims) # states: ["T", "P", "CO", "CO2", "H2", "CH4", "CH3OH", "H2O", "N2"]

states_max = [569.952065200784, 71.49265445971363, 0.0534738227626869, 0.0839279358015094, 0.4739118921128102, 0.1961240582176027, 0.043617617295987, 0.0166983631358979, 0.0286116689671041] + [0] * (28 - len(data_files))# from training data
states_max_dict = {}
for d , val in zip(opt_model.dims, states_max):
    states_max_dict[d] = val
opt_model.states_max = pyo.Param(opt_model.dims, initialize = states_max_dict)

states_min  = [466.35539818346194, 57.31174829828023, 0.0172916368293674, 0.0552752589680291, 0.3095623691919211, 0.1604881777757451, 0.0028584153155807, 0.0006125105511711, 0.0234112567627298] + [0] * (28 - len(data_files))
states_min_dict = {}
for d , val in zip(opt_model.dims, states_min):
    states_min_dict[d] = val
opt_model.states_min = pyo.Param(opt_model.dims, initialize = states_min_dict)

# state var
opt_model.x = pyo.Var(opt_model.dec_space, opt_model.dims) # state vars
opt_model.x_enc = pyo.Var(opt_model.enc_space, opt_model.dims)

# CO outlet  constraint
opt_model.x[opt_model.dec_space.last(), "CO"].ub = 0.02

# Temperature inlet constraints
opt_model.x[opt_model.dec_space.first(), "T"].ub = 550
opt_model.x[opt_model.dec_space.first(), "T"].lb = 450

# Pressure inlet constraints
opt_model.x[opt_model.dec_space.first(), "P"].ub = 68
opt_model.x[opt_model.dec_space.first(), "P"].lb = 62

# x bounds
for s in opt_model.dec_space:
    for d, dim in enumerate(opt_model.dims):
        opt_model.x[s,dim].ub = 1.5 * opt_model.states_max[dim]
        opt_model.x[s,dim].lb = 0.5 * opt_model.states_min[dim] 

# x encoder constraints
opt_model.x_enc_constraints = pyo.ConstraintList()
for s in opt_model.enc_space:
    for dim in opt_model.dims:
        opt_model.x_enc_constraints.add(expr= opt_model.x_enc[s,dim] == opt_model.x[opt_model.dec_space.first(), dim])

layer_names, parameters, _, enc_dec_count, layer_outputs_dict = extract_from_pretrained.get_hugging_learned_parameters(tnn_model, src , tgt, 2, hugging_face_dict)



# define transformer to model reactor
transformer = TNN.Transformer( ".\\data\\reactor_config_huggingface.json", opt_model) 
result =  transformer.build_from_pytorch( tnn_model,sample_enc_input=src, sample_dec_input=src,enc_bounds = bounds_target , dec_bounds=bounds_target, Transformer='huggingface', default=False, hugging_face_dict=hugging_face_dict)
print("transformer built: ",result)
tnn_input_enc = getattr( opt_model, result[0][0])
tnn_input_dec = getattr( opt_model, result[0][1])
tnn_output = getattr( opt_model, result[-2])

print(tnn_input_enc)
print(tnn_input_dec)
print(tnn_output)

# get outputs from trained transformer
# _, _, _, _, layer_outputs_dict = extract_from_pretrained.get_pytorch_learned_parameters(tnn_model, src , tgt, transformer.d_H, transformer.N, 'huggingface', hugging_face_dict)
    