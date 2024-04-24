import pyomo.environ as pyo
from pyomo import dae
import numpy as np
import toy_formulation 

# instantiate pyomo model component
model = pyo.ConcreteModel(name="(TOY)")

# define sets
T = 4
time = np.linspace(0,1,num=T)
model.time = dae.ContinuousSet(initialize=time)

# T= 11    
# x_input = [1.0, 1.10657895, 1.21388889], 1.32205882, 1.43125, 1.54166667, 1.65357143, 1.76730769, 1.88333333, 2.00227273, 2.125]
# u_input = [0.25, 0.26315789, 0.27777778, 0.29411765, 0.3125, 0.33333333, 0.35714286, 0.38461538, 0.41666667, 0.45454545, 0.5]

# T= 4
x_input = [1.0, 1.35833333, 1.72916667, 2.125]
u_input = [0.25,  0.3,   0.375, 0.5 ]

dict_x = {(t): val for t, val in zip(model.time, x_input)}
dict_u = {(t): val for t, val in zip(model.time, u_input)}
model.x_in = pyo.Param(model.time, initialize = dict_x)
model.u_in = pyo.Param(model.time, initialize = dict_u)

# define variables
model.x = pyo.Var(model.time, bounds=(0,10))
model.u = pyo.Var(model.time, bounds=(0,10))

# define constraints
model.x_init_constr = pyo.Constraint(expr = model.x[0] == 1)

transformer = toy_formulation.transformer(model)
transformer.add_input_constraints(model)
transformer.embed_input(model)
transformer.add_layer_norm(model)
transformer.add_attention(model)
transformer.add_output_constraints(model)
# TT.GET_INPUT_SEQ()
# model.x_pred_constr = pyo.Constraint(model.time, rule=toy_formulation._x_transformer)
# model.u_pred_constr = pyo.Constraint(model.time, rule=toy_formulation._u_transformer)

# define objective
def _intX(m,t):
   return m.x[t]
def _intU(m,t):
   return m.u[t]
model.intX = dae.Integral(model.time,wrt=model.time,rule=_intX)
model.intU = dae.Integral(model.time,wrt=model.time,rule=_intU)

model.obj = pyo.Objective(expr = model.intX - model.intU + model.x[model.time.last()], sense=1) # -1: maximize, +1: minimize (default)


# Discretize model using Backward Difference method
discretizer = pyo.TransformationFactory( 'dae.finite_difference')
discretizer.apply_to(model , nfe=T-1, wrt=model.time , scheme='BACKWARD' )

# view model
#model.pprint() #pyomo solve test.py --solver=gurobi --stream-solver --summary
# 'ipopt' for non-linear

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