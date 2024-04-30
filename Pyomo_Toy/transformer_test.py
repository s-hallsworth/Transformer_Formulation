# External imports
import pyomo.environ as pyo
import numpy as np
from pyomo import dae
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import unittest

# Import from repo file
import transformer

# ------- Transformer Test Class ------------------------------------------------------------------------------------
class TestTransformer(unittest.TestCase):
    def test_pyomo_input(self, model, pyomo_input_name ,transformer_input):
        
        input_var = getattr(model, pyomo_input_name)
        pyomo_input_dict = {}

        # Check if the variable is indexed
        if input_var.is_indexed():
            pyomo_input_dict[pyomo_input_name] = {index: pyo.value(input_var[index]) for index in input_var.index_set()}
        else:
            pyomo_input_dict[pyomo_input_name] = pyo.value(input_var)
        print(pyomo_input_dict)
        
        pyomo_input, elements = reformat(dict=pyomo_input_dict, layer_name=pyomo_input_name) 
        print(pyomo_input.shape)
        
        
        # plt.figure(1, figsize=(12, 8))
        # markers = ["o-", "s-"]  # Different markers for each function
        # var = [layer_norm_output, transformer_output]
        # for i in range(len(var)):
        #     plt.plot(elements, var[i][0, 0 , :], markers[i], label=f"x values from array {i}")
        #     plt.plot(elements, var[i][0, 1 , :], markers[i], label=f"u values from array {i}")
        # plt.title("Pyomo and Tranformer results ")
        # plt.xlabel("Sequence")
        # plt.ylabel("Magnitude")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        
        # self.assertIsNone(np.testing.assert_array_equal(layer_norm_output, transformer_input))
    
    def test_layer_norm(self, model, config_file, T, transformer_output):
        
        # Define tranformer and execute up to layer norm
        transformer = transformer.Transformer(model, config_file)
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
        
        # get optimal parameters & reformat first layer norm block --> (1, input_feature, sequence_element)
        optimal_parameters = get_optimal_dict(result, model)
        layer_norm_output, elements = reformat(optimal_parameters,"layer_norm") 
        print(layer_norm_output.shape)
        
        
        plt.figure(1, figsize=(12, 8))
        markers = ["o-", "s-"]  # Different markers for each function
        var = [layer_norm_output, transformer_output]
        for i in range(len(var)):
            plt.plot(elements, var[i][0, 0 , :], markers[i], label=f"x values from array {i}")
            plt.plot(elements, var[i][0, 1 , :], markers[i], label=f"u values from array {i}")
        plt.title("Pyomo and Tranformer results ")
        plt.xlabel("Sequence")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        self.assertIsNone(np.testing.assert_array_equal(layer_norm_output, transformer_output))

# -------- Helper functions ----------------------------------------------------------------------------------       
def get_optimal_dict(self, result, model):
    if result.solver.status == 'ok' and result.solver.termination_condition == 'optimal':
        optimal_parameters = {}
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
    
def reformat(dict, layer_name):
    """
    Reformat pyomo var to match transformer var shape: (1, input_feature, sequence_element)
    """
    print('Reformatting ',layer_name, ' values')

    elements = sorted(set(elem for elem, _ in dict[layer_name].keys()))
    features = sorted(set(feat for _, feat in dict[layer_name].keys())) #x : '0', u: '1' which matches transformer array

    layer_norm_output = np.zeros((1, len(features),len(elements)))
    for (elem, feat), value in dict[layer_name].items():
        elem_index = elements.index(elem)
        feat_index = features.index(feat)
        
        layer_norm_output[0, feat_index, elem_index] = value
        
    
    return layer_norm_output, elements

