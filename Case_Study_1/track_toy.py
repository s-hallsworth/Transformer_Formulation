import pyomo.environ as pyo
import numpy as np
from helpers.print_stats import solve_pyomo
import matplotlib.pyplot as plt
# from amplpy import AMPL
"""
close but not exactly the same as analytical solution
"""

# instantiate pyomo model component
model = pyo.ConcreteModel(name="(TOY_TRANFORMER)")

# define sets
T = 10
steps = 20
time = np.linspace(0, T, num=steps)
model.time = pyo.Set(initialize=time)

# define variables
model.e = pyo.Var()
model.g = pyo.Var(model.time, bounds=(0, T))
model.x = pyo.Var(model.time, bounds=(0, T))
model.u = pyo.Var(model.time, initialize=1)


# define constraints
model.x_init = pyo.Constraint(expr=model.x[0] == 0)



def _g_rule(M,t):
    return M.g[t] == (0.25*t) + pyo.sin(t)
model.g_con = pyo.Constraint(model.time, rule=_g_rule)

model.x_diff_con = pyo.ConstraintList() 
for t_index, t in enumerate(model.time):
    index = t_index + 1
    if t_index < steps-1:
        model.x_diff_con.add(expr= model.x[t] + (  (2 * (model.u[t]))) == model.x[model.time.at(index + 1)])


model.e_con = pyo.Constraint(expr= model.e == sum((model.g[t] - model.x[t])**2 for t in model.time) )


model.obj = pyo.Objective(
    expr=model.e, sense=1
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


print(optimal_parameters['e'])

x = np.array(list(optimal_parameters['x'].items()))
u = np.array(list(optimal_parameters['u'].items()))
g = np.array(list(optimal_parameters['g'].items()))

plt.figure(figsize=(6, 4))
plt.plot(time, g[:,1], 's-', label = f'Expected Value')
plt.plot(time, x[:,1], '--x', label = f'input (x)')
plt.plot(time, u[:,1], '--x', label = f'control (u)')

plt.legend()
plt.show()