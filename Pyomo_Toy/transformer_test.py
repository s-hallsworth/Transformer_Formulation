# External imports
import pyomo.environ as pyo
import numpy as np
from pyomo import dae
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import unittest
import os

# Import from repo file
import transformer as TNN
import toy_problem_setup as tps
import transformer_intermediate_results as tir

"""
Test each module of transformer
"""
# ------- Transformer Test Class ------------------------------------------------------------------------------------
class TestTransformer(unittest.TestCase):    

    # def test_pyomo_input(self): #, model, pyomo_input_name ,transformer_input):
    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     pyomo_input_name = "input_param"
    #     self.solver = 'scip'
        
    #     # Get input var
    #     input_var = getattr(model, pyomo_input_name)
    #     pyomo_input_dict = {}

    #     # Store input pyomo var as dict
    #     if input_var.is_indexed():
    #         pyomo_input_dict[pyomo_input_name] = {index: pyo.value(input_var[index]) for index in input_var.index_set()}
    #     else:
    #         pyomo_input_dict[pyomo_input_name] = pyo.value(input_var)

    #     # Reformat and convert dict to np array
    #     pyomo_input, _ = reformat(dict=pyomo_input_dict, layer_name=pyomo_input_name) 
        
    #     ## layer outputs  
    #     transformer_input = np.array(tir.layer_outputs_dict['input_layer_1'])
        
    #     # Assertions
    #     self.assertIsNone(np.testing.assert_array_equal(pyomo_input.shape, transformer_input.shape)) # pyomo input data and transformer input data must be the same shape
    #     self.assertIsNone(np.testing.assert_array_almost_equal(pyomo_input, transformer_input, decimal = 7))             # both inputs must be equal
        
    # def test_no_embed_input(self):
    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     config_file = '.\\data\\toy_config.json' 
    #     T = 11 
    #     self.solver = 'scip'
        
    #     # Define tranformer and execute up to embed
    #     transformer = TNN.Transformer(model, config_file)
    #     transformer.embed_input(model, "input_param","input_embed", "variables")

    #     # Discretize model using Backward Difference method
    #     discretizer = pyo.TransformationFactory("dae.finite_difference")
    #     discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
    #     # Solve model
    #     solver = SolverFactory('ipopt')
    #     opts = {'halt_on_ampl_error': 'yes',
    #        'tol': 1e-7, 'bound_relax_factor': 0.0}
    #     result = solver.solve(model, options=opts)
        
    #     # Get optimal parameters & reformat
    #     optimal_parameters = get_optimal_dict(result, model)
    #     embed_output, _ = reformat(optimal_parameters,"input_embed") 
        
    #     ## layer outputs  
    #     transformer_input = np.array(tir.layer_outputs_dict['input_layer_1'])
        
    #     # Assertions
    #     self.assertIsNone(np.testing.assert_array_equal(embed_output.shape, transformer_input.shape)) # same shape
    #     self.assertIsNone(np.testing.assert_array_almost_equal(embed_output, transformer_input, decimal = 7))             # equal vlaues
    #     with self.assertRaises(ValueError):  # attempt to overwrite layer_norm var
    #         transformer.embed_input(model, "input_param","input_embed", "variables")
    
    # def test_embed_input(self):
    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     config_file = '.\\data\\toy_config_embed_3.json' 
    #     T = 11
    #     self.solver = 'scip'
        
    #     # Define tranformer and execute up to embed
    #     transformer = TNN.Transformer(model, config_file)
    #     W_emb = np.random.rand(transformer.input_dim, transformer.d_model) # define rand embedding matrix
    #     transformer.embed_input(model, "input_param","input_embed", "variables",W_emb)
        
    #     self.assertIn("input_embed", dir(model))                       # check var created
    #     self.assertIsInstance(model.input_embed, pyo.Var)               # check data type
    #     self.assertTrue(hasattr(model, 'embed_constraints'))      # check constraints created
        
    #     # Discretize model using Backward Difference method
    #     discretizer = pyo.TransformationFactory("dae.finite_difference")
    #     discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
    #     # Solve model
    #     solver = SolverFactory('ipopt')
    #     opts = {'halt_on_ampl_error': 'yes',
    #        'tol': 1e-7, 'bound_relax_factor': 0.0}
    #     result = solver.solve(model, options=opts)
        
    #     # Get optimal parameters & reformat  --> (1, input_feature, sequence_element)
    #     optimal_parameters = get_optimal_dict(result, model)
    #     embed_output, _ = reformat(optimal_parameters,"input_embed") 
        
    #     # Calculate embedded value
    #     transformer_input = np.array(tir.layer_outputs_dict['input_layer_1'])
    #     transformer_embed = np.dot(transformer_input, W_emb) # W_emb dim: (2, 3), transformer_input dim: (1,10,2)

    #     # Assertions
    #     self.assertIsNone(np.testing.assert_array_equal(embed_output.shape, transformer_embed.shape)) # same shape
    #     self.assertIsNone(np.testing.assert_array_almost_equal(embed_output, transformer_embed, decimal=2))  # almost same values
    
    # def test_layer_norm(self):
    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     config_file = '.\\data\\toy_config.json' 
    #     T = 11
    #     self.solver = 'scip'
        
    #     # Define tranformer and execute up to layer norm
    #     transformer = TNN.Transformer(model, config_file)
    #     transformer.embed_input(model, "input_param","input_embed", "variables")
    #     transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        
    #     # Check layer norm var and constraints created
    #     self.assertIn("layer_norm", dir(model))                        # check layer_norm created
    #     self.assertIsInstance(model.layer_norm, pyo.Var)               # check data type
    #     self.assertTrue(hasattr(model, 'layer_norm_constraints'))      # check constraints created
        
    #     # Discretize model using Backward Difference method
    #     discretizer = pyo.TransformationFactory("dae.finite_difference")
    #     discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
    #     # Solve model
    #     solver = SolverFactory(self.solver)
    #     opts = {'halt_on_ampl_error': 'yes',
    #        'tol': 1e-7, 'bound_relax_factor': 0.0}
    #     result = solver.solve(model, options=opts)
        
    #     # get optimal parameters & reformat first layer norm block --> (1, input_feature, sequence_element)
    #     optimal_parameters = get_optimal_dict(result, model)
    #     layer_norm_output, elements = reformat(optimal_parameters,"layer_norm") 
    #     LN_1_output= np.array(tir.layer_outputs_dict["layer_normalization_1"])
        
    #     plt.figure(1, figsize=(12, 8))
    #     markers = ["o-", "x--"]  # Different markers for each function
    #     var = [layer_norm_output, LN_1_output]
    #     labels = ['- Pyomo', '- Transformer']
    #     for i in range(len(var)):
    #         plt.plot(elements, var[i][0, :, 0 ], markers[i], label=f"x values {labels[i]}")
    #         plt.plot(elements, var[i][0, :, 1 ], markers[i], label=f"u values {labels[i]}")
    #     plt.title("Pyomo and Tranformer results ")
    #     plt.xlabel("Sequence")
    #     plt.ylabel("Magnitude")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
        

    #     # print("layer norm Pyomo (as list):", [model.layer_norm[t, d].value for t in model.time_input for d in model.model_dims])
    #     # print("layer norm from NumPy:", transformer_output)
        
        
    #     # Assertions
    #     self.assertIsNone(np.testing.assert_array_equal(layer_norm_output.shape, LN_1_output.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(layer_norm_output,LN_1_output, decimal=5)) # decimal=1 # compare value with transformer output
    #     with self.assertRaises(ValueError):  # attempt to overwrite layer_norm var
    #         transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")

    def test_multi_head_attention(self):
        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config.json' 
        T = 11
        self.solver = 'gurobi'
        
        # Define tranformer and execute 
        transformer = TNN.Transformer(model, config_file)
        transformer.embed_input(model, "input_param","input_embed", "variables")
        transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        transformer.add_attention(model, "layer_norm", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
        
        # Check  var and constraints created
        self.assertIn("attention_output", dir(model))                 # check layer_norm created
        self.assertIsInstance(model.attention_output, pyo.Var)        # check data type
        self.assertTrue(hasattr(model, 'attention_constraints'))      # check constraints created
        
        # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # Solve model
        solver = SolverFactory(self.solver, solver_io='python')
        # #opts = {'halt_on_ampl_error': 'yes',
        #    'tol': 1e-7, 'bound_relax_factor': 0.0}
        
        result = solver.solve(model)
        
        # get optimal parameters & reformat  --> (1, input_feature, sequence_element)
        optimal_parameters = get_optimal_dict(result, model)
        attention_output, elements = reformat(optimal_parameters,"attention_output") 
        print(attention_output)
        attention_output = np.expand_dims(attention_output, axis=2)
        print(attention_output.shape)
        

        # # print(" Pyomo (as list):", [model.layer_norm[t, d].value for t in model.time_input for d in model.model_dims])
        # # print(" from NumPy:", transformer_output)
        
        MHA_output = np.array(tir.layer_outputs_dict["multi_head_attention_1"])
        print(MHA_output.shape)
        # # Assertions
        self.assertIsNone(np.testing.assert_array_equal(attention_output.shape, MHA_output .shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(attention_output, MHA_output , decimal=1)) # compare value with transformer output
        #with self.assertRaises(ValueError):  # attempt to overwrite layer_norm var
        #     transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        
# -------- Helper functions ----------------------------------------------------------------------------------       
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
    
def reformat(dict, layer_name):
    """
    Reformat pyomo var to match transformer var shape: (1, input_feature, sequence_element)
    """
    elements = sorted(set(elem for elem, _ in dict[layer_name].keys()))
    features = sorted(set(feat for _, feat in dict[layer_name].keys())) #x : '0', u: '1' which matches transformer array

    output = np.zeros((1,len(elements), len(features)))
    for (elem, feat), value in dict[layer_name].items():
        elem_index = elements.index(elem)
        feat_index = features.index(feat)
        
        output[0, elem_index, feat_index] = value
        
    return output, elements

# ------- MAIN -----------------------------------------------------------------------------------
if __name__ == '__main__': 
    unittest.main() 

