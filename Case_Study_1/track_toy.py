import pyomo.environ as pyo
import numpy as np
import math
from helpers.print_stats import solve_pyomo
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from amplpy import AMPL

"""
"""

# instantiate pyomo model component
model = pyo.ConcreteModel(name="(TOY_TRANFORMER)")

# define constants
T_end = 0.0105#0.5
steps = 19 #100
time = np.linspace(0, T_end, num=steps)
dt = time[1] - time[0]
print(time)

g = 9.81
v_l1 = 0.2
v_l2 = 1.5


# define sets
model.time = pyo.Set(initialize=time)

# define parameters
# def target_location_rule(M, t):
#     return v_l1 * t
# model.loc1 = pyo.Param(model.time, rule=target_location_rule) 

# def target_location2_rule(M, t):
#     return (v_l2*t) - (0.5 * g * (t**2)) #+ (np.random.rand(1)/30)
# model.loc2 = pyo.Param(model.time, rule=target_location2_rule) 
bounds_target = (-3,3)
# define variables
model.loc1 = pyo.Var(model.time, bounds = bounds_target )
model.loc2 = pyo.Var(model.time, bounds = bounds_target )

# define variables
model.x1 = pyo.Var(model.time) # distance path
model.v1 = pyo.Var(bounds=(0,None)) # initial velocity of cannon ball

model.x2 = pyo.Var(model.time) # height path
model.v2 = pyo.Var(bounds=(0,None)) # initial velocity of cannon ball

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

# def x1_target_rule(M, t):
#     return M.v1 * t == model.loc1[t]
# model.x1_target_constr = pyo.Constraint(model.time, rule=x1_target_rule)

# def x2_target_rule(M, t):
#     return (M.v2 * t) - (0.5*g * (t**2)) == model.loc2[t]
# model.x2_target_constr = pyo.Constraint(model.time, rule=x2_target_rule)

# Fix model solution
input_x1 =   v_l1 * time  
input_x2 =  (v_l2*time) - (0.5 * g * (time*time))

model.fixed_loc_constraints = pyo.ConstraintList()
for i,t in enumerate(model.time):
    model.fixed_loc_constraints.add(expr= input_x1[i] == model.loc1[t])
    model.fixed_loc_constraints.add(expr= input_x2[i]  == model.loc2[t])




# Set objective
model.obj = pyo.Objective(
    expr= sum((model.x1[t] - model.loc1[t])**2 + (model.x2[t] - model.loc2[t])**2 for t in model.time), sense=1
)  # -1: maximize, +1: minimize (default)


# view model
#model.pprint()  # pyomo solve test.py --solver=gurobi --stream-solver --summary

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
plt.plot(time, loc2, 'o', label = f'x2 data')
plt.plot(time, x2, '--x', label = f'x2 predicted')
plt.plot(time, loc1, 'o', label = f'x1 data')
plt.plot(time, x1, '--x', label = f'x1 predicted')
plt.title(f'Example')
plt.legend()
plt.show()

plt.figure(2, figsize=(6, 4))
plt.plot(loc1, loc2, 'o', label = f'target trajectory')
plt.plot(x1, x2, '--x', label = f'cannon ball trajectory')
plt.title(f'Trajectory of cannon ball')
plt.legend()
plt.show()

