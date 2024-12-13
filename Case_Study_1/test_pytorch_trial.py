import torch
from MINLP_tnn.helpers.extract_from_pretrained import get_pytorch_intermediate_values
import transformer_b as TNN
import pyomo.environ as pyo
import MINLP_tnn.helpers.convert_pyomo as convert_pyomo
from gurobipy import GRB
from MINLP_tnn.helpers.print_stats import solve_gurobipy
import numpy as np

data = [[6.0 * np.ones(6)], 
         [7.0 * np.ones(6)], 
         [8.0 * np.ones(6)], 
         [9.0 * np.ones(6)], 
         [10.0 * np.ones(6)], 
         [11.0 * np.ones(6)], 
         [12.0 * np.ones(6)],
         [13.0 * np.ones(6)],
         [14.0 * np.ones(6)],
         [15.0 * np.ones(6)]]
data = torch.tensor(data, dtype=torch.float32) /100
print(data.shape, data)
src = data[ :-1, 0, :]
tgt = data[1:  , 0, :]
print(src.shape, tgt.shape)

# # load model
sequence_size = 9
device = torch.device('cpu')
model_path = ".\\pytorch_model.pt"
model = torch.nn.Transformer(d_model= 6, nhead=2, num_encoder_layers=1, num_decoder_layers=1,dim_feedforward=128, batch_first=True)
model.load_state_dict(torch.load(model_path, map_location=device))
out_pre_trained = model(src, tgt)

print(out_pre_trained )
# Get intermediary results
intermediate_outputs = get_pytorch_intermediate_values(model, src, tgt)
print("intermediary output: ", intermediate_outputs)

# Get learned parameters
opt_model = pyo.ConcreteModel(name="opt_model_name")
transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", opt_model) 
result =  transformer.build_from_hug_torch( model, src, tgt, enc_bounds = (0,1), dec_bounds=(0,1))
print("transformer built: ",result)

## Set objective
tnn_output = getattr( opt_model, result[-2])
opt_model.obj = pyo.Objective( expr= sum(tnn_output[i] for i in tnn_output.index_set()), sense=1) 

opt_model.input_var_constraints = pyo.ConstraintList()

def _tnn_out(M, t, d):
    if t < M.dec_time_dims.last():
        return pyo.Constraint.Skip
    return M.dec_input_param[t,d] == tnn_output[d]


model.tnn_con = pyo.Constraint(opt_model.dec_time_dims, opt_model.input_dims, rule=_tnn_out)


## Convert to gurobipy
gurobi_model, map_var, _ = convert_pyomo.to_gurobi(opt_model)

## Solve
time_limit = 240
solve_gurobipy(gurobi_model, time_limit) ## Solve and print


if gurobi_model.status == GRB.INFEASIBLE:
        gurobi_model.computeIIS()
        gurobi_model.write("pytorch_model.ilp")