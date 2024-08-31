import torch
from extract_from_pretrained import get_pytorch_learned_parameters, get_pytorch_intermediate_values
import transformer_b as TNN
import pyomo.environ as pyo
from omlt import OmltBlock
import convert_pyomo
from gurobipy import Model, GRB, GurobiError
from print_stats import solve_gurobipy

"""
    Case Study Explanation here
"""

# import trained transformer model

train_tnn_path = ".\\trained_transformer\\model_TimeSeriesTransformer_final.pth"
torch_model = get_torch_model(train_tnn_path)