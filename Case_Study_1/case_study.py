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
from transformers.src.transformers.models.time_series_transformer.configuration_time_series_transformer import TimeSeriesTransformerConfig
from transformers.src.transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerForPrediction
# cloned transformers from: https://github.com/s-hallsworth/transformers.git
# pip install -e .

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
# print(summary(tnn_model))
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
    