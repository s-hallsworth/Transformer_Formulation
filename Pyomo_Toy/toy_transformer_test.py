import pyomo.environ as pyo
import numpy as np
from pyomo import dae
from pyomo.opt import SolverFactory
import json
import unittest

import toy_transformer
import toy_problem_test

class TestTransformer(unittest.TestCase):
    def test_layer_norm(self, model, config_file, T):
        
        transformer = toy_transformer.Transformer(model, config_file)
        transformer.embed_input(model, "input_var","input_embed", "variables")
        transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        
        # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # solve model
        solver = SolverFactory('ipopt')
        opts = {'halt_on_ampl_error': 'yes',
           'tol': 1e-7, 'bound_relax_factor': 0.0}
        result = solver.solve(model, options=opts)
        
        if result.solver.status == 'ok' and result.solver.termination_condition == 'optimal':
            optimal_parameters = {}
            for varname, var in model.component_map(pyo.Var).items():
                # Check if the variable is indexed
                if var.is_indexed():
                    optimal_parameters[varname] = {index: pyo.value(var[index]) for index in var.index_set()}
                else:
                    optimal_parameters[varname] = pyo.value(var)
            print("Optimal Parameters:", optimal_parameters)
        else:
            print("No optimal solution obtained.")
                
        
test_transformer = TestTransformer()
test_transformer.test_layer_norm(toy_problem_test.model, "toy_config.json", T=11)