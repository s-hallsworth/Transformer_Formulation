import pyomo.environ as pyo
import numpy as np
import math
from helpers.print_stats import solve_pyomo, solve_gurobipy
from gurobipy import Model, GRB, GurobiError
import helpers.convert_pyomo as convert_pyomo
import matplotlib.pyplot as plt
import torch
import transformer_b as TNN
from helpers.extract_from_pretrained import get_pytorch_learned_parameters, get_pytorch_intermediate_values

from trained_transformer.Tmodel import TransformerModel
# from amplpy import AMPL

"""
"""

# instantiate pyomo model component
model = pyo.ConcreteModel(name="(TOY_TRANFORMER)")

# define constants
T_end = 0.0105
steps = 11 ##CHANGE THIS ##
time = np.linspace(0, T_end, num=steps)
dt = time[1] - time[0]
tt = 10 # sequence size
time_history = time[0:tt]
pred_len = 1

g = 9.81
v_l1 = 0.2
v_l2 = 1.5

src = np.array([np.random.rand(1)*time[0:-1] , (2*np.random.rand(1) * time[0:-1]) - (0.5 * 9.81* time[0:-1] * time[0:-1])])# random sample input [x1_targte, x2_target]
src = src.transpose(1,0)

# define sets
model.time = pyo.Set(initialize=time)
model.time_history = pyo.Set(initialize=time[0:-1])
# print(len(time), len(time[0:-1]), time)
 
# define parameters
def target_location_rule(M, t):
    return v_l1 * t
model.history_loc1 = pyo.Param(model.time_history, rule=target_location_rule) 

def target_location2_rule(M, t):
    return (v_l2*t) - (0.5 * g * (t**2))
model.history_loc2 = pyo.Param(model.time_history, rule=target_location2_rule) 
 
history_loc1 = np.array([v for k,v in model.history_loc1.items()])
history_loc2 = np.array([v for k,v in model.history_loc2.items()])
print(history_loc1, history_loc2)

bounds_target = (-3,3)
# define variables
model.loc1 = pyo.Var(model.time, bounds = bounds_target )
model.loc2 = pyo.Var(model.time, bounds = bounds_target )

model.x1 = pyo.Var(model.time) # distance path
model.v1 = pyo.Var() # initial velocity of cannon ball

model.x2 = pyo.Var(model.time) # height path
model.v2 = pyo.Var() # initial velocity of cannon ball

#model.T = pyo.Var(within=model.time)# time when cannon ball hits target

# define initial conditions
model.x1_constr = pyo.Constraint(expr= model.x1[0] == 0) 
model.x2_constr = pyo.Constraint(expr= model.x2[0] == 0) 

# define constraints
def v1_rule(M, t):
    return M.x1[t] == M.v1 * t
model.v1_constr = pyo.Constraint(model.time, rule=v1_rule) 

def v2_rule(M, t):
    return M.x2[t] == (M.v2 * t) - (0.5*g * (t**2))
model.v2_constr = pyo.Constraint(model.time, rule=v2_rule)

model.v1_pos_constr = pyo.Constraint(expr = model.v1 >= 0)
model.v2_pos_constr = pyo.Constraint(expr = model.v2 >= 0)

def loc1_rule(M, t):
    return M.loc1[t] == model.history_loc1[t]
model.loc1_constr = pyo.Constraint(model.time_history, rule=loc1_rule)

def loc2_rule(M, t):
    return M.loc2[t] == model.history_loc2[t]
model.loc2_constr = pyo.Constraint(model.time_history, rule=loc2_rule)

# load trained transformer
sequence_size = tt
device = torch.device('cpu')
tnn_path = ".\\trained_transformer\\toy_pytorch_model_1.pt"
tnn_model = TransformerModel(input_dim=2, output_dim =2, d_model=12, nhead=4, num_encoder_layers=1, num_decoder_layers=1)
tnn_model.load_state_dict(torch.load(tnn_path, map_location=device))

# create optimization transformer
transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", model) 
result =  transformer.build_from_pytorch( tnn_model,sample_enc_input=src, sample_dec_input=src,enc_bounds = bounds_target , dec_bounds=bounds_target )
print("transformer built: ",result)
tnn_input_enc = getattr( model, result[0][0])
tnn_input_dec = getattr( model, result[0][1])
tnn_output = getattr( model, result[-2])

# add constraints to trained TNN input
model.tnn_input_constraints = pyo.ConstraintList()
indices = []
for set in str(tnn_input_enc.index_set()).split("*"):
    indices.append( getattr(model, set) )
for tnn_index, index in zip(indices[0], model.time_history):
    print(tnn_index, index)
    model.tnn_input_constraints.add(expr= tnn_input_enc[tnn_index, indices[1].first()]== model.history_loc1[index])
    model.tnn_input_constraints.add(expr= tnn_input_enc[tnn_index, indices[1].last()] == model.history_loc2[index]) 
    
indices = []
for set in str(tnn_input_dec.index_set()).split("*"):
    indices.append( getattr(model, set) )
for t_index, t in enumerate(model.time):
    index = t_index + 1 # 1 indexing
    
    if index > pred_len and index < tt + pred_len + 1:
        model.tnn_input_constraints.add(expr= tnn_input_dec[indices[0].at(index - pred_len), indices[1].first()] == model.loc1[t])
        model.tnn_input_constraints.add(expr= tnn_input_dec[indices[0].at(index - pred_len), indices[1].last()]  == model.loc2[t])
        
# add constraints to trained TNN output
model.tnn_output_constraints = pyo.ConstraintList()
indices = []
for set in str(tnn_output.index_set()).split("*"):
    indices.append( getattr(model, set) )
print("-------------")
print(np.array([v for k,v in indices[0].items()]))
print(np.array([v for k,v in indices[1].items()]))
model.tnn_output_constraints.add(expr= tnn_output[indices[0].last(), indices[1].first()] == model.loc1[model.time.last()])
model.tnn_output_constraints.add(expr= tnn_output[indices[0].last(), indices[1].last()] == model.loc2[model.time.last()])
    
    
# Set objective
model.obj = pyo.Objective(
    expr= sum((model.x1[t] - model.loc1[t])**2 + (model.x2[t] - model.loc2[t])**2 for t in model.time), sense=1
)  # -1: maximize, +1: minimize (default)

## Convert to gurobipy
gurobi_model, map_var, _ = convert_pyomo.to_gurobi(model)
# gurobi_model.setParam('DualReductions',0)

## Solve
time_limit = None
solve_gurobipy(gurobi_model, time_limit) ## Solve and print

if gurobi_model.status == GRB.INFEASIBLE:
        gurobi_model.computeIIS()
        gurobi_model.write("pytorch_model.ilp")
else:
    ## Get optimal parameters
    if gurobi_model.status == GRB.OPTIMAL:
        optimal_parameters = {}
        
        
        for v in gurobi_model.getVars():
            
            if "[" in v.varName:
                name = v.varname.split("[")[0]
                if name in optimal_parameters.keys():
                    optimal_parameters[name] += [v.x]
                else:
                    optimal_parameters[name] = [v.x]
            else:    
                optimal_parameters[v.varName] = v.x
                
            
        ##
        print(
        "objective value:", gurobi_model.getObjective().getValue()
        )  


## view model
#model.pprint()  # pyomo solve test.py --solver=gurobi --stream-solver --summary

# solver = pyo.SolverFactory("gurobi")
# time_limit = None
# result = solve_pyomo(model, solver, time_limit)

# def get_optimal_dict(result, model):
#     optimal_parameters = {}
#     if result.solver.status == 'ok' and result.solver.termination_condition == 'optimal':
#         for varname, var in model.component_map(pyo.Var).items():
#             # Check if the variable is indexed
#             if var.is_indexed():
#                 optimal_parameters[varname] = {index: pyo.value(var[index]) for index in var.index_set()}
#             else:
#                 optimal_parameters[varname] = pyo.value(var)
#         #print("Optimal Parameters:", optimal_parameters)
#     else:
#         print("No optimal solution obtained.")
    
#     return optimal_parameters
# #----------------------------
# optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat  --> (1, input_feature, sequence_element)

x1 = np.array(list(optimal_parameters['x1'].items()))[:,1]
x2 = np.array(list(optimal_parameters['x2'].items()))[:,1]
loc1 = np.array([v for k,v in model.loc1.items()])
loc2 = np.array([v for k,v in model.loc2.items()])

v1= np.array(optimal_parameters['v1'])
v2= np.array(optimal_parameters['v2'])
# T= np.array(optimal_parameters['T'])


plt.figure(1, figsize=(6, 4))
plt.plot(time, loc2, '-o', label = f'target x2')
plt.plot(time, x2, '--x', label = f'x2')
plt.plot(time, loc1, '-o', label = f'target x1')
plt.plot(time, x1, '--x', label = f'x1')
plt.title(f'Trajectory of cannon ball')
plt.legend()
plt.show()

plt.figure(2, figsize=(6, 4))
plt.plot(loc1, loc2, '-o', label = f'target trajectory')
plt.plot(x1, x2, '--x', label = f'cannon ball trajectory')
plt.title(f'Trajectory of cannon ball')
plt.legend()
plt.show()

