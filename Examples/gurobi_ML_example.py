"""
An example of using the Gurobi ML package to build an optimization model with a Neural Network (NN) embedded.
This exmaple is taken from: https://gurobi-machinelearning.readthedocs.io/en/stable/auto_userguide/example_simple.html
"""

import gurobipy as gp
import numpy as np
from sklearn.datasets import (
    make_regression,
)  # to generate random data for the regression
from sklearn.neural_network import MLPRegressor  # import NN
from gurobi_ml import add_predictor_constr

# Build artificial data
X, y = make_regression(n_features=10, noise=1.0)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Create MLPRegressor object and fit it to the data
nn = MLPRegressor(
    hidden_layer_sizes=[20] * 2,
    activation="relu",
    solver="adam",
    max_iter=10000,
    random_state=1,
)  # max_iter: # of epochs
nn.fit(X, y)

# Optimization Model
"""
Adverserial example:
We pick n training examples randomly. For each of the examples, we want to find an input 
that is in a small neighborhood of it that leads to the output that is closer to 0
with the regression. 
"""
## Vars for adverserial example
n = 2  # number of randomly picked training examples
delta = 0.2  # size of small deviation from X_example
index = np.random.choice(X.shape[0], n, replace=False)  # select n random indices
X_examples = X[index, :]
y_examples = y[index]

## Create gurobi model
m = gp.Model()

## Create decision vars
input_vars = m.addMVar(
    X_examples.shape, lb=X_examples - delta, ub=X_examples + 0.2
)  # add matrix variable with same shape as X_examples
output_vars = m.addMVar(
    y_examples.shape, lb=-gp.GRB.INFINITY
)  # set -inf lb since Gurobi vars are non-negative

## Add constraint to link NN input_vars and output_vars
pred_constr = add_predictor_constr(
    m, nn, input_vars, output_vars
)  # modeling object returned. *NB* due to shape of vars, n different constraints will be added

## Print modelling object outputs
pred_constr.print_stats()

## Set objective & optimize
m.setObjective(output_vars @ output_vars, gp.GRB.MINIMIZE)  # minimize output^2
m.optimize()

## Check solution from Gurobi is correct wrt regression model used
print()
print(
    "Error:", pred_constr.get_error()
)  # computes error g(X_example)-y using original regression nn.

## View computed output:
print("Computed values")
print(pred_constr.output_values.flatten())
print("Original values")
print(y_examples)

##Remove pred_constr
# pred_constr.remove()

print()
print(
    "objective value:", m.getObjective().getValue()
)  # output var^2. why only 1 value?

# Print Optimization problem vars:
# for v in m.getVars():
#     print(f"{v.VarName} = {v.X}")
# OR:
# all_vars = mo.getVars()
# values = m.getAttr("X", all_vars)
# names = m.getAttr("VarName", all_vars)

# for name, val in zip(names, values):
#     print(f"{name} = {val}")
