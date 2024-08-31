import torch
from helpers.extract_from_pretrained import get_pytorch_learned_parameters, get_pytorch_intermediate_values
# import transformer_b as TNN
import pyomo.environ as pyo
from omlt import OmltBlock
import helpers.convert_pyomo
from gurobipy import Model, GRB, GurobiError
from helpers.print_stats import solve_gurobipy
import numpy as np

# create model
d_mod = 2
transformer_model = torch.nn.Transformer(d_model= d_mod, nhead=2, num_encoder_layers=1, num_decoder_layers=1,dim_feedforward=128, batch_first=True)
source = torch.tensor([[[1.0 * np.ones(d_mod)], 
         [2.0 * np.ones(d_mod)], 
         [3.0 * np.ones(d_mod)], 
         [4.0 * np.ones(d_mod)], 
         [5.0 * np.ones(d_mod)], 
         [6.0 * np.ones(d_mod)], 
         [7.0 * np.ones(d_mod)], 
         [8.0 * np.ones(d_mod)], 
         [9.0 * np.ones(d_mod)], 
         [10.0 * np.ones(d_mod)]],

        [
         [2.0 * np.ones(d_mod)], 
         [3.0 * np.ones(d_mod)], 
         [4.0 * np.ones(d_mod)], 
         [5.0 * np.ones(d_mod)], 
         [6.0 * np.ones(d_mod)], 
         [7.0 * np.ones(d_mod)], 
         [8.0 * np.ones(d_mod)], 
         [9.0 * np.ones(d_mod)], 
         [10.0 * np.ones(d_mod)],
         [11.0 * np.ones(d_mod)] ],

        [ 
         [3.0 * np.ones(d_mod)], 
         [4.0 * np.ones(d_mod)], 
         [5.0 * np.ones(d_mod)], 
         [6.0 * np.ones(d_mod)], 
         [7.0 * np.ones(d_mod)], 
         [8.0 * np.ones(d_mod)], 
         [9.0 * np.ones(d_mod)], 
         [10.0 * np.ones(d_mod)],
         [11.0 * np.ones(d_mod)],
         [12.0 * np.ones(d_mod)]],

        [ 
         [4.0 * np.ones(d_mod)], 
         [5.0 * np.ones(d_mod)], 
         [6.0 * np.ones(d_mod)], 
         [7.0 * np.ones(d_mod)], 
         [8.0 * np.ones(d_mod)], 
         [9.0 * np.ones(d_mod)], 
         [10.0 * np.ones(d_mod)],
         [11.0 * np.ones(d_mod)],
         [12.0 * np.ones(d_mod)],
         [13.0 * np.ones(d_mod)],],

        [[5.0 * np.ones(d_mod)], 
         [6.0 * np.ones(d_mod)], 
         [7.0 * np.ones(d_mod)], 
         [8.0 * np.ones(d_mod)], 
         [9.0 * np.ones(d_mod)], 
         [10.0 * np.ones(d_mod)], 
         [11.0 * np.ones(d_mod)],
         [12.0 * np.ones(d_mod)],
         [13.0 * np.ones(d_mod)],
         [14.0 * np.ones(d_mod)],]])
data = torch.tensor(source, dtype=torch.float32) /100
print(data)
# src = data[:, :-1, 0, :]
# tgt = data[:, 1:, 0, :]
# print(src.shape, tgt.shape)
# out = transformer_model(src, tgt) #src: input to encoder, tgt: input to decoder
# print(out)

torch.manual_seed(0)

from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 

optimizer = torch.optim.SGD(transformer_model.parameters(),lr=0.01)
loss_fn = torch.nn.MSELoss()

def train_epoch(model, optimizer, sequence_size, batch_size=64):
    model.train()
    losses = 0 
    source = data[:, :-1, 0, :]
    target = data[:, 1:, 0, :]
    
    src_mask = transformer_model.generate_square_subsequent_mask(sequence_size)
    dataset = TensorDataset(source, target)
    train_dataloader = DataLoader(dataset,  
                        batch_size=batch_size,  
                        shuffle=True)
    for src, tgt in train_dataloader:
        
        out = transformer_model(src,tgt,src_mask)
        optimizer.zero_grad()
        
        loss = loss_fn(out,tgt)
        loss.backward()
        
        optimizer.step()
        losses += loss.item()
        # print(loss.item())
    return losses / len(list(train_dataloader))

from timeit import default_timer as timer
NUM_EPOCHS = 400
sequence_size = 9
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer_model, optimizer, sequence_size, 1)
    end_time = timer()
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))


#save model
model_path = ".\\pytorch_model.pt"
torch.save(transformer_model.state_dict(), model_path)

# # load model
# device = torch.device('cpu')
# model = torch.nn.Transformer(d_model= 6, nhead=2, num_encoder_layers=1, num_decoder_layers=1,dim_feedforward=10, batch_first=True)
# model.load_state_dict(torch.load(model_path, map_location=device))
# out_pre_trained = model(src, tgt)

# # print("---------")
# # print(out)

# # Get learned parameters
# #layer_names, dict_transformer_params, model = get_pytorch_learned_parameters(model, input_shape= (5, 10, 4), head_size=2)
# #print(dict_transformer_params)

# # Get intermediate outputs of model for testing
# sample_input = src[0]
# sample_input2 = tgt[0]
# sample_input2[-1:] = torch.tensor([0, 0, 0, 0, 0, 0])

# intermediate_outputs = get_pytorch_intermediate_values(model, sample_input, sample_input2)

# # Create transformer 
# opt_model = pyo.ConcreteModel(name="opt_model_name")
# transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", opt_model) 
# result =  transformer.build_from_pytorch( model, sample_input, sample_input2, enc_bounds = (0,10), dec_bounds=(0,10))
# print(result)

# ## Set objective
# tnn_output = getattr( opt_model, result[-2])
# opt_model.obj = pyo.Objective( expr= sum(tnn_output[i] for i in tnn_output.index_set()), sense=1) 

# opt_model.input_var_constraints = pyo.ConstraintList()
# opt_model.dec_time_dims.last()
# for d in opt_model.input_dims:
#     opt_model.input_var_constraints.add(expr=opt_model.dec_input_param[d, opt_model.dec_time_dims.last()] == tnn_output[d])
    
# ## Convert to gurobipy
# gurobi_model, map_var = convert_pyomo.to_gurobi(opt_model)

# ## Solve
# time_limit = 240
# solve_gurobipy(gurobi_model, time_limit) ## Solve and print


# if gurobi_model.status == GRB.INFEASIBLE:
#         gurobi_model.computeIIS()
#         gurobi_model.write("pytorch_model.ilp")