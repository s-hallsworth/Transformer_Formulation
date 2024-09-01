import pyomo.environ as pyo
import numpy as np
import math
from helpers.print_stats import solve_pyomo
import matplotlib.pyplot as plt
# from amplpy import AMPL

"""
Control of mass-spring damper system with actuator to control the base motion (u) 
and a disturbance acting on the mass (d)
"""

# instantiate pyomo model component
model = pyo.ConcreteModel(name="(TOY_TRANFORMER)")

# define constants
T = 10
steps = 10
time = np.linspace(0, T, num=steps)
dt = time[1] - time[0]
w  = 1
D  = 8 #max amplitude of disturbance
c  = 4
m  = 1
k  = 4

# define sets
model.time = pyo.Set(initialize=time)

# define parameters
def init_disturbance(model, t):
    return  D * pyo.cos(w*t) 
model.d = pyo.Param(model.time, initialize=init_disturbance )

# define variables
model.x = pyo.Var(model.time, bounds=(-D, D)) # state vars
model.dx_dt = pyo.Var(model.time)
model.dx2_dt2 = pyo.Var(model.time)
model.F_x = pyo.Var(model.time)

model.u = pyo.Var(model.time, bounds=(-D, D)) # control vars
model.du_dt = pyo.Var(model.time)

# define constraints

#initially no control input
# model.x_init = pyo.Constraint(expr=model.x[0] == 0) 
model.u_init = pyo.Constraint(expr=model.u[0] == -1) 
model.dudt_init = pyo.Constraint(expr=model.du_dt[0] == -1) 

# dx dt = delta x / delta t
model.x_diff_con = pyo.ConstraintList() 
for t_index, t in enumerate(model.time):
    index = t_index + 1 # vars index starts at 1
    if t_index > 1:
        model.x_diff_con.add(expr= model.dx_dt[t] * dt == model.x[t] - model.x[model.time.at(t_index - 1)])
        model.x_diff_con.add(expr= model.dx2_dt2[t] * dt == model.dx_dt[t] - model.dx_dt[model.time.at(t_index - 1)])

# du dt = delta u / delta t
model.u_diff_con = pyo.ConstraintList() 
for t_index, t in enumerate(model.time):
    index = t_index + 1 # vars index starts at 1
    if t_index > 1:
        model.u_diff_con.add(expr= model.du_dt[t] * dt == model.u[t] - model.u[model.time.at(t_index - 1)])

# net force exerted on m
def F_x_constr_rule(M, t):
    return (m * M.dx2_dt2[t]) + (c * M.dx_dt[t]) + (k*M.x[t]) == M.F_x[t]
model.F_x_constr = pyo.Constraint(model.time, rule=F_x_constr_rule) 

# equation of motion
def EOM_constr_rule(M, t):
    return M.F_x[t] == (c * M.du_dt[t]) + (k*M.u[t]) + M.d[t]
model.EOM_constr = pyo.Constraint(model.time, rule=EOM_constr_rule) 


# Set objective
model.obj = pyo.Objective(
    expr= sum( model.x[t]**2 for t in model.time), sense=1
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

Fx = np.array(list(optimal_parameters['F_x'].items()))[:,1]
x = np.array(list(optimal_parameters['x'].items()))[:,1]
u = np.array(list(optimal_parameters['u'].items()))[:,1]
dudt = np.array(list(optimal_parameters['du_dt'].items()))[:,1]
d = np.array([v for k,v in model.d.items()])
print("d: ", d.shape, d)
print("u: ", u.shape, u)
print("x: ", x.shape, x)
print("dudt: ", dudt.shape, dudt)

u_expected = []
Force_u = (c*dudt) + (k*u)
Force_u_expected = []
for t in time:
    u_expected.append( - math.cos(t) - math.sin(t))
    Force_u_expected.append( (c * (math.sin(t) - math.cos(t))) + (k* (- math.cos(t) - math.sin(t))))
    
plt.figure(1, figsize=(6, 4))
plt.plot(time, x, '--o', label = f'state (mass displacement)')
plt.plot(time, u, '--o', label = f'control input (base displacement)')
plt.plot(time, u_expected, '--x', label = f'control input expected')
plt.legend()
plt.show()


plt.figure(2, figsize=(6, 4))
plt.plot(time, d, '--o', label = f'disturbance force')
plt.plot(time, Fx, '--o', label = f'net force acting on mass')
plt.plot(time, Force_u, '--o', label = f'control force')
plt.plot(time, Force_u_expected, '--x', label = f'control force expected')

plt.legend()
plt.show()

