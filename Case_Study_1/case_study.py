import torch
#from helpers.extract_from_pretrained import get_pytorch_learned_parameters, get_pytorch_intermediate_values
import transformer_b as TNN
import pyomo.environ as pyo
from omlt import OmltBlock
#import helpers.convert_pyomo
from gurobipy import Model, GRB, GurobiError
#from helpers.print_stats import solve_gurobipy
from torchinfo import summary
from torch.nn import Transformer



"""
    Case Study Explanation here
"""

# import trained transformer model

train_tnn_path = ".\\trained_transformer\\model_TimeSeriesTransformer_final.pth"
#train_tnn_path = ".\\trained_transformer\\pytorch_model.pt"
model = Transformer()
model.load_state_dict(torch.load(train_tnn_path, weights_only=False, map_location=torch.device('cpu')))
print(model)
print(summary(model))
#torch_model = get_torch_model(train_tnn_path)