"""
A simple Pyomo model to demonstrate different modelling components.

Example: A fixed volume CSTR is modelled to masimise the desired product's concentration.
The CSTR is undergoinga series reation at fixed tempertature.
The CSTR is at steady state

A --> B --> C
Desired product: B
(A --> B) r_1 = k_1 C_A
(B --> C) r_2 = k_2 C_B
where r_1, r_2 are reactions
"""

import pyomo.environ as pyo

# instantiating a pyomo model component
model = pyo.ConcreteModel(name="(CSTR)")

# define set elements using lists
set_components = ["Ain", "A", "B"]
set_reactions = ["1", "2"]

model.components = pyo.Set(initialize=set_components)
model.reactions = pyo.Set(initialize=set_reactions)

# initialising parameter values usig dicts
dict_k = {"1": 5, "2": 2}  # units: s^-1
scalar_vol = 1  # units m^3

model.k = pyo.Param(model.reactions, initialize=dict_k)
model.vol = pyo.Param(initialize=scalar_vol)

# declaring all variables appearing in optimization model
model.conc = pyo.Var(
    model.components, within=pyo.NonNegativeReals
)  # create indexed variable with concentration of every component
model.F = pyo.Var(within=pyo.NonNegativeReals)

# specifying constraints
model.mass_bal_A = pyo.Constraint(
    expr=model.F * (model.conc["Ain"] - model.conc["A"])
    - (model.k["1"] * model.conc["A"] * model.vol)
    == 0
)


def mass_bal_B_rule(m):
    return (-m.F * m.conc["B"]) + (
        m.k["1"] * m.conc["A"] - m.k["2"] * m.conc["B"]
    ) * model.vol == 0  # equations can be indexed too


model.mass_bal_B = pyo.Constraint(rule=mass_bal_B_rule)

# define objective function
model.obj = pyo.Objective(
    expr=model.conc["B"], sense=-1
)  # -1: maximize, +1: minimize (default)

# specify bounds
model.conc["Ain"].setub(1)  # lower bound set by nonnegreal
model.conc["A"].setub(
    0.1
)  # on indexed set can set all of the bounds when declaering if bounds are the same
model.F.setub(20)

# view model
model.pprint()

# run from terminal:
# pyomo solve pyomo_example.py --solver=gurobi --stream-solver --summary

# Or:
# from pyomo.environ import SolverFactory
# solver = SolverFactory('scip')
# result = solver.solve(model)
# print(result)
