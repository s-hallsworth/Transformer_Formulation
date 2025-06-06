import pyomo.environ as pyo
import numpy as np
from MINLP_tnn.helpers.print_stats import solve_pyomo
import matplotlib.pyplot as plt
# from amplpy import AMPL

"""
A version of the optimal trajectory problem without the TNN.


The projectile motion is modelled with equations of motion but the 
set up of this problem remains the same. The results of this optimisation 
problem are used to test the problem with optimisation-based TNN to model
th eprojectile motion. These results give the expected optimal trajectory.
"""

# instantiate pyomo model component
model = pyo.ConcreteModel(name="(TOY_TRANFORMER)")


# define constants
T_end = 0.5
steps = 19 
time = np.linspace(0, T_end, num=steps)

tt = 2 # sequence size
time_history = time[0:tt]
pred_len = 1

time = time[:tt+pred_len]
steps = len(time)
print(steps)

g = 9.81
v_l1 = 0.2
v_l2 = 1.5
dt = time[-1] - time[0]
    



# define sets
model.time = pyo.Set(initialize=time)

# define parameters
def target_location_rule(M, t):
    return v_l1 * t
model.loc1 = pyo.Param(model.time, rule=target_location_rule) 

def target_location2_rule(M, t):
    np.random.seed(int(v_l2*t*100))
    print(np.random.uniform(-1,1)/30)
    return (v_l2*t) - (0.5 * g * (t**2)) + ( np.random.uniform(-1,1)/30 )
model.loc2 = pyo.Param(model.time, rule=target_location2_rule) 

# define variables
model.x1 = pyo.Var(model.time) # distance path
model.v1 = pyo.Var(bounds=(0,None)) # initial velocity of cannon ball

model.x2 = pyo.Var(model.time) # height path
model.v2 = pyo.Var(bounds=(0,None)) # initial velocity of cannon ball


# define initial conditions
model.x1_constr = pyo.Constraint(expr= model.x1[0] == 0) 
model.x2_constr = pyo.Constraint(expr= model.x2[0] == 0) 

# # define constraints

model.v1_constr = pyo.Constraint(expr= model.v1 >= 0) 
model.v2_constr = pyo.Constraint(expr= model.v2 >= 0) 
def v1_rule(M, t):
    return M.x1[t] == M.v1 * t
model.v1_constr = pyo.Constraint(model.time, rule=v1_rule) 

def v2_rule(M, t):
    return M.x2[t] == (M.v2*t) - (0.5 * g * (t**2))
model.v2_constr = pyo.Constraint(model.time, rule=v2_rule)

# Set objective
model.obj = pyo.Objective(
    expr= sum((model.x1[t] - model.loc1[t])**2 + (model.x2[t] - model.loc2[t])**2 for t in model.time), sense=1
)  # -1: maximize, +1: minimize (default)


# ##------ Fix model solution ------##
input_x1 =   v_l1 * time  
input_x2 =  (v_l2*time) - (0.5 * g * (time*time))


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

print("opt x1: ", x1)
print("opt x2: ", x2)
plt.figure(1, figsize=(6, 4))
plt.plot(time, loc2, 'o', label = 'x2 data')
plt.plot(time, x2, '--x', label = 'x2 predicted')
plt.plot(time, loc1, 'o', label = 'x1 data')
plt.plot(time, x1, '--x', label = 'x1 predicted')
plt.title('Example')
plt.legend()
plt.show()

plt.figure(2, figsize=(6, 4))
plt.plot(loc1, loc2, 'o', label = 'target trajectory')
plt.plot(x1, x2, '--x', label = 'cannon ball trajectory')
plt.title('Trajectory of cannon ball')
plt.legend()
plt.show()

