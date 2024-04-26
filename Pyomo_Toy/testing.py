import pyomo.environ as pyo
from pyomo import dae
import numpy as np

# from amplpy import AMPL
"""
close but not exactly the same as analytical solution
"""

# instantiate pyomo model component
model = pyo.ConcreteModel(name="(TOY_TRANFORMER)")

# define sets
T = 11
time = np.linspace(0, 1, num=T)
model.time = dae.ContinuousSet(initialize=time)

# define variables
model.x = pyo.Var(model.time, bounds=(0, 10))
model.u = pyo.Var(model.time, bounds=(0, 10))
model.dxdt = dae.DerivativeVar(model.x, wrt=(model.time), initialize=0)

# define constraints
model.x_init = pyo.Constraint(expr=model.x[0] == 1)


def _dxdt(M, t):
    if t == M.time.first():
        return pyo.Constraint.Skip
    return M.dxdt[t] == 1 + (M.u[t] ** 2)


model.dxdt_con = pyo.Constraint(model.time, rule=_dxdt)


# define objective
def _intX(m, t):
    return m.x[t]


def _intU(m, t):
    return m.u[t]


model.intX = dae.Integral(model.time, wrt=model.time, rule=_intX)
model.intU = dae.Integral(model.time, wrt=model.time, rule=_intU)

model.obj = pyo.Objective(
    expr=model.intX - model.intU + model.x[model.time.last()], sense=1
)  # -1: maximize, +1: minimize (default)


# Discretize model using Backward Difference method
discretizer = pyo.TransformationFactory("dae.finite_difference")
discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")

# view model
model.pprint()  # pyomo solve test.py --solver=gurobi --stream-solver --summary

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
