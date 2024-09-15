import torch
#from helpers.extract_from_pretrained import get_pytorch_learned_parameters, get_pytorch_intermediate_values
import transformer_b as TNN
import pyomo.environ as pyo
from omlt import OmltBlock
import helpers.convert_pyomo
from gurobipy import Model, GRB, GurobiError
from helpers.print_stats import solve_gurobipy
from torchinfo import summary
from torch.nn import Transformer
from transformers.src.transformers.models.time_series_transformer.configuration_time_series_transformer import TimeSeriesTransformerConfig
from transformers.src.transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerForPrediction
# cloned transformers from: https://github.com/s-hallsworth/transformers.git


"""
    Case Study Explanation here
"""

# import trained transformer model

train_tnn_path = ".\\trained_transformer\\case_study\\model_TimeSeriesTransformer_final.pth"
#train_tnn_path = ".\\trained_transformer\\pytorch_model.pt"
#model = torch.nn.Transformer()

# Model Configuration
DEVICE = "cpu"
NUMBER_OF_POINTS = 8
data_files = ["T", "P", "CO", "CO2", "H2", "CH4", "CH3OH", "H2O", "N2"]

config = TimeSeriesTransformerConfig(
    prediction_length=NUMBER_OF_POINTS,
    context_length=3,
    embedding_dimension=60,
    scaling=False,
    lags_sequence=[0],
    num_time_features=1,
    input_size=len(data_files),
    num_parallel_samples=1,
    #loss="mse"
)
tnn_model = TimeSeriesTransformerForPrediction(config=config).to(DEVICE)




# model = TimeSeriesTransformerForPrediction.from_pretrained(train_tnn_path).to(DEVICE)
#model.load_state_dict(torch.load(train_tnn_path, weights_only=False, map_location=torch.device('cpu')))

tnn_model = torch.load(train_tnn_path, weights_only=False, map_location=torch.device('cpu'))
print(type(tnn_model), tnn_model)
print(summary(tnn_model))
#torch_model = get_torch_model(train_tnn_path)

src = torch.ones(NUMBER_OF_POINTS, len(data_files))
tgt = src
bounds_target = (-1,1)
# create optimization transformer
transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", tnn_model) 
result =  transformer.build_from_pytorch( tnn_model,sample_enc_input=src, sample_dec_input=src,enc_bounds = bounds_target , dec_bounds=bounds_target, Transfromer='huggingface' )
print("transformer built: ",result)
tnn_input_enc = getattr( tnn_model, result[0][0])
tnn_input_dec = getattr( tnn_model, result[0][1])
tnn_output = getattr( tnn_model, result[-2])