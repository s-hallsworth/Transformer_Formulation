import pyomo.environ as pyo
from pyomo import dae
import numpy as np
import transformer as transformer
import extract_from_pretrained as extract_from_pretrained
from toy_problem_setup import *

"""
Add transformer instance/constraints to toy problem setup and solve

# Commands:
model.pprint() # view model
pyomo solve test.py --solver=gurobi --stream-solver --summary # run model in terminal ('ipopt' for NLP)

"""

# create transformer instance     
transformer = transformer.Transformer(model, ".\data\toy_config.json")

# add trnasformer layers and constraints
transformer.embed_input(model, "input_param","input_embed", "variables")
transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
transformer.add_attention(model, "layer_norm", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
transformer.add_residual_connection(model, model.input_embed, model.layer_norm, "mha_residual")
#transformer.add_output_constraints(model, model.mha_residual)


# Discretize model using Backward Difference method
discretizer = pyo.TransformationFactory("dae.finite_difference")
discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")



# # -------------------- SOLVE MODEL ----------------------------------- #
# from pyomo.opt import SolverFactory # run with python3.10
# solver = SolverFactory('ipopt')
# opts = {'halt_on_ampl_error': 'yes',
#            'tol': 1e-7, 'bound_relax_factor': 0.0}
# result = solver.solve(model, logfile='.\logs\solver_result.log',
#                     symbolic_solver_labels=True, tee=True, load_solutions=True, options=opts)

# print(result)


