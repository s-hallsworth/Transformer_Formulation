import pyomo.environ as pyo
import numpy as np
import math
from helpers.print_stats import solve_pyomo, solve_gurobipy
import helpers.convert_pyomo as convert_pyomo
import matplotlib.pyplot as plt
import torch
import transformer_b as TNN
from helpers.extract_from_pretrained import get_pytorch_learned_parameters, get_pytorch_intermediate_values
from sklearn.preprocessing import MinMaxScaler

from trained_transformer.Tmodel import TransformerModel
# from amplpy import AMPL

"""
"""

# instantiate pyomo model component
model = pyo.ConcreteModel(name="(TOY_TRANFORMER)")

# define constants
T_end = 30
steps = 50
time = np.linspace(0, T_end, num=steps)
dt = time[1] - time[0]
tt = 10
time_history = time[0:tt]

g = 9.81
v_l1 = 2
v_l2 = 3
loc1_h = v_l1 * time_history
loc2_h = (v_l2 * time_history) - (0.5 * g * (time_history**2)) 


scaler = MinMaxScaler(feature_range=(0, 1))
loc1_h = scaler.fit_transform(loc1_h.reshape(-1,1))
loc2_h = scaler.fit_transform(loc2_h.reshape(-1,1))
src = np.array([loc1_h, loc2_h])[:,:,0].transpose(1,0)
print(src.shape)

loc1 = {}
loc2 = {}
for l, v in zip(time_history, loc1_h):
    loc1[l] = v[0]
    
for l, v in zip(time_history, loc2_h):
    loc2[l] = v[0]
print(loc1, loc2)

# define sets
model.time = pyo.Set(initialize=time)
model.time_history = pyo.Set(initialize=time[0:tt])
 
# define parameters
model.history_loc1 =  pyo.Param(model.time_history, initialize=loc1) 
model.history_loc2 =  pyo.Param(model.time_history, initialize=loc2) 
history_loc1 = np.array([v for k,v in model.history_loc1.items()])
history_loc2 = np.array([v for k,v in model.history_loc2.items()])
print(history_loc1, history_loc2)


# define variables
model.loc1 = pyo.Var(model.time)
model.loc2 = pyo.Var(model.time)

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

# define transformer
sequence_size = tt
device = torch.device('cpu')
tnn_path = ".\\trained_transformer\\toy_pytorch_model_n.pt"
tnn_model = TransformerModel(input_dim=2, output_dim =2)
tnn_model.load_state_dict(torch.load(tnn_path, map_location=device))

transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", model) 
result =  transformer.build_from_pytorch( tnn_model,enc_input=src, dec_input=src,enc_bounds = (0,1), dec_bounds=(0,1))
print("transformer built: ",result)

# Set objective
model.obj = pyo.Objective(
    expr= sum((model.x1[t] - model.loc1[t])**2 + (model.x2[t] - model.loc2[t])**2 for t in model.time), sense=1
)  # -1: maximize, +1: minimize (default)


# view model
model.pprint()  # pyomo solve test.py --solver=gurobi --stream-solver --summary

solver = pyo.SolverFactory("gurobi")
time_limit = None
result = solve_pyomo(model, solver, time_limit)

def get_optimal_dict(result, model):
    optimal_parameters = {}
    if result.solver.status == 'ok' and result.solver.termination_condition == 'optimal':
        for varname, var in model.component_map(pyo.Var).items():
            # Check if the variable is indexed
            if var.is_indexed():
                optimal_parameters[varname] = {index: pyo.value(var[index]) for index in var.index_set()}
            else:
                optimal_parameters[varname] = pyo.value(var)
        #print("Optimal Parameters:", optimal_parameters)
    else:
        print("No optimal solution obtained.")
    
    return optimal_parameters
#----------------------------
optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat  --> (1, input_feature, sequence_element)

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

