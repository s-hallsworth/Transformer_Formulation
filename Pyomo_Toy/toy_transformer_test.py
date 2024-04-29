import pyomo.environ as pyo
from pyomo import dae
import numpy as np
import toy_transformer
import Pyomo_Toy.extract_from_pretrained as extract_from_pretrained
import unittest

## create model
model = pyo.ConcreteModel(name="(TOY_TEST)")

## define problem sets, vars, params
T = 11
time = np.linspace(0, 1, num=T)
model.time = dae.ContinuousSet(initialize=time)

x_input = [1.0, 1.10657895, 1.21388889, 1.32205882, 1.43125, 1.54166667, 1.65357143, 1.76730769, 1.88333333, 2.00227273, 2.125]
u_input = [0.25, 0.26315789, 0.27777778, 0.29411765, 0.3125, 0.33333333, 0.35714286, 0.38461538, 0.41666667, 0.45454545, 0.5]

set_variables = ['1', '0']
model.variables = pyo.Set(initialize=set_variables)
dict_inputs = {}
for t, (u_val, x_val) in zip(model.time, zip(u_input, x_input)):
    dict_inputs[(t, '1')] = u_val
    dict_inputs[(t, '0')] = x_val
model.input_param = pyo.Param(model.time, model.variables, initialize=dict_inputs)
model.input_var = pyo.Var(model.time, model.variables, bounds=(0, 10))


## define transformer sets, vars, params
model.time_input = dae.ContinuousSet(initialize=time[:-1])
layer_names, parameters = extract_from_pretrained.get_learned_parameters("model_weights.json")

dict_gamma1 = {(t): val for t,val in zip(model.time_input, parameters['layer_normalization_130','gamma'])}
dict_beta1 = {(t): val for t,val in zip(model.time_input,  parameters['layer_normalization_130','beta'])}
model.gamma1 = pyo.Param(model.time_input, initialize = dict_gamma1)
model.beta1 = pyo.Param(model.time_input, initialize = dict_beta1)

W_q = parameters['multi_head_attention_65','W_q']
W_k = parameters['multi_head_attention_65','W_k']
W_v = parameters['multi_head_attention_65','W_v']
W_o = parameters['multi_head_attention_65','W_o']

b_q = parameters['multi_head_attention_65','b_q']
b_k = parameters['multi_head_attention_65','b_k']
b_v = parameters['multi_head_attention_65','b_v']
b_o = parameters['multi_head_attention_65','b_o']

        
## define constraints
model.x_init_constr = pyo.Constraint(expr=model.input_var[min(model.time),'0'] == 1)

input_array = []
model.input_constraints = pyo.ConstraintList()
for t in model.time:
    if t == model.time.first():
        model.input_constraints.add(expr=model.input_var[t,'1'] == model.input_param[t,'1'])
        continue
    # create arrays for input x and u values
    if t <= 0.9:
        # add constraints that x,u = x,u input values
        model.input_constraints.add(expr=model.input_var[t,'0'] == model.input_param[t,'0'])
        model.input_constraints.add(expr=model.input_var[t,'1'] == model.input_param[t,'1'])

        # create arrays with input values
        input_array.append(
            [model.input_param[t,'1'], model.input_param[t,'0']]
        )  
        
# transformer = toy_transformer.Transformer(model, "toy_config.json")         
# transformer.embed_input(model, "input_var","input_embed", "variables")
# transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")

# transformer.add_attention(model, "layer_norm", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
# transformer.add_residual_connection(model, model.input_embed, model.layer_norm, "mha_residual")
#transformer.add_output_constraints(model, model.mha_residual)


# define objective
def _intX(m, t):
    return m.input_var[t,'0']
def _intU(m, t):
    return m.input_var[t,'1']
model.intX = dae.Integral(model.time, wrt=model.time, rule=_intX)
model.intU = dae.Integral(model.time, wrt=model.time, rule=_intU)

model.obj = pyo.Objective(
    expr=model.intX - model.intU + model.input_var[model.time.last(),'0'], sense=1
)  # -1: maximize, +1: minimize (default)


# Discretize model using Backward Difference method
discretizer = pyo.TransformationFactory("dae.finite_difference")
discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")


# view model
# model.pprint() #pyomo solve test.py --solver=gurobi --stream-solver --summary
# 'ipopt' for non-linear

# ------------------------------------------------------- #
# from pyomo.environ import SolverFactory
# solver = SolverFactory('scip')
# result = solver.solve(model)

# x = []
# t = []
# u = []

# for i in sorted(model.time):
#     x.append(model.x[i])
#     t.append(i)
#     u.append(model.u[i])

# import numpy
# import matplotlib.pyplot as plt

# fig = plt.figure()
# plt.ylabel('X')
# plt.xlabel('time')
# plt.plot( t,x, label='x')
# plt.plot( t,u, label='u')
# fig.show()
