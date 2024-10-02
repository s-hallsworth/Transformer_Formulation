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

from functools import partial
from neural_compressor.config import AccuracyCriterion, TuningCriterion, PostTrainingQuantConfig
import evaluate
from datasets import load_dataset
from optimum.intel import INCQuantizer

# import trained transformer model
train_tnn_path = ".\\trained_transformer\\case_study\\model_TimeSeriesTransformer_final.pth"

# Model Configuration
device = "cpu"
NUMBER_OF_POINTS = 8
CONTEXT_LENGTH = 3
data_files = ["T", "P", "CO", "CO2", "H2", "CH4", "CH3OH", "H2O", "N2"]

config = TimeSeriesTransformerConfig(
    prediction_length=NUMBER_OF_POINTS,
)
tnn_model = TimeSeriesTransformerForPrediction(config).to(device)

tnn_model = torch.load(train_tnn_path, weights_only=False, map_location=torch.device('cpu'))
tnn_model.config.prediction_length = NUMBER_OF_POINTS
tnn_model.config.context_length=3
tnn_model.config.embedding_dimension=60
tnn_model.config.scaling=False
tnn_model.config.lags_sequence=[0]
tnn_model.config.num_time_features=1
tnn_model.config.input_size=len(data_files)
tnn_model.config.num_parallel_samples=1

# Prune config
compression_config = [{
        "compression":
        {
        "algorithm":  "movement_sparsity",
        "params": {
            "warmup_start_epoch":  1,
            "warmup_end_epoch":    4,
            "importance_regularization_factor":  0.01,
            "enable_structured_masking":  True
        },
        "sparse_structure_by_scopes": [
            {"mode":  "block",   "sparse_factors": [32, 32], "target_scopes": "{re}.*BertAttention.*"},
            {"mode":  "per_dim", "axis":  0,                 "target_scopes": "{re}.*BertIntermediate.*"},
            {"mode":  "per_dim", "axis":  1,                 "target_scopes": "{re}.*BertOutput.*"},
        ],
        "ignored_scopes": ["{re}.*NNCFEmbedding", "{re}.*pooler.*", "{re}.*LayerNorm.*"]
        }
    },{
        "algorithm": "quantization",
        "weights": {"mode": "symmetric"},
        "activations": { "mode": "symmetric"},
    }]

from optimum.intel import OVQuantizer, OVModelForSequenceClassification, OVConfig, OVQuantizationConfig
import pandas as pd
quantizer = OVQuantizer.from_pretrained(tnn_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = r"..\data\Training_Pdrop"
torch.manual_seed(42)

# Load Train Data
print("Load data")
data_files = ["P", "P", "CO", "CO2", "H2", "CH4", "CH3OH", "H2O", "N2"]
States = [pd.read_csv(f"{PATH}\{file}.csv", header=None) for file in data_files]
NUMBER_OF_POINTS = 8
states = [state.to_numpy() for state in States]
df_raw = np.stack(states)
mean = df_raw.mean(axis=(1,2), keepdims=True)
std = df_raw.std(axis=(1,2), keepdims=True)

#df = ((df_raw - mean) / std).astype(np.float32)
df = df_raw
min = df.min(axis=(1,2), keepdims=True)
max = df.max(axis=(1,2), keepdims=True)
print(df.shape, max.shape)
df = ((df - min) / (max - min)).astype(np.float32) #range 0 to 1
#df = df_raw.astype(np.float32)
df = torch.from_numpy(df).permute(1, 2, 0)[:, :NUMBER_OF_POINTS, ...]
dataset = torch.TensorDataset(df[:, 0,:].unsqueeze(1), df[:, :,:])

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.random_split(dataset, [train_size, val_size])

batch_size = 24
train_loader = torch.DataLoader(train_dataset, batch_size=batch_size)
val_loader = torch.DataLoader(val_dataset, batch_size=batch_size)

calibration_dataset = train_loader 
ov_config = OVConfig(quantization_config=OVQuantizationConfig())
quantizer.quantize(ov_config=ov_config, calibration_dataset=calibration_dataset, save_directory=save_dir)
