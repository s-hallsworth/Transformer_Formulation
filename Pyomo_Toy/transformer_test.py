# External imports
import pyomo.environ as pyo
import numpy as np
from pyomo import dae
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import unittest
import os
from omlt import OmltBlock
import convert_pyomo
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

# Import from repo file
import transformer as TNN
import toy_problem_setup_test as tps
import transformer_intermediate_results as tir

"""
Test each module of transformer
"""
# ------- Transformer Test Class ------------------------------------------------------------------------------------
class TestTransformer(unittest.TestCase):    

    def test_pyomo_input(self): #, model, pyomo_input_name ,transformer_input):
        # Define Test Case Params
        model = tps.model.clone()
        pyomo_input_name = "input_param"
        self.solver = 'scip'
        
        # Get input var
        input_var = getattr(model, pyomo_input_name)
        pyomo_input_dict = {}

        # Store input pyomo var as dict
        if input_var.is_indexed():
            pyomo_input_dict[pyomo_input_name] = {index: pyo.value(input_var[index]) for index in input_var.index_set()}
        else:
            pyomo_input_dict[pyomo_input_name] = pyo.value(input_var)

        # Reformat and convert dict to np array
        pyomo_input, _ = reformat(dict=pyomo_input_dict, layer_name=pyomo_input_name) 
        
        ## layer outputs  
        transformer_input = np.array(tir.layer_outputs_dict['input_layer_1'])
        
        # Assertions
        self.assertIsNone(np.testing.assert_array_equal(pyomo_input.shape, transformer_input.shape)) # pyomo input data and transformer input data must be the same shape
        self.assertIsNone(np.testing.assert_array_almost_equal(pyomo_input, transformer_input, decimal = 7))             # both inputs must be equal
        
    def test_no_embed_input(self):
        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config_relu_2.json' 
        T = 11 
        self.solver = 'scip'
        
        # Define tranformer and execute up to embed
        transformer = TNN.Transformer(model, config_file)
        transformer.embed_input(model, "input_param","input_embed", "variables")

        # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # Solve model
        solver = SolverFactory('ipopt')
        opts = {'halt_on_ampl_error': 'yes',
           'tol': 1e-7, 'bound_relax_factor': 0.0}
        result = solver.solve(model, options=opts)
        
        # Get optimal parameters & reformat
        optimal_parameters = get_optimal_dict(result, model)
        embed_output, _ = reformat(optimal_parameters,"input_embed") 
        
        ## layer outputs  
        transformer_input = np.array(tir.layer_outputs_dict['input_layer_1'])
        
        # Assertions
        self.assertIsNone(np.testing.assert_array_equal(embed_output.shape, transformer_input.shape)) # same shape
        self.assertIsNone(np.testing.assert_array_almost_equal(embed_output, transformer_input, decimal = 7))             # equal vlaues
        with self.assertRaises(ValueError):  # attempt to overwrite layer_norm var
            transformer.embed_input(model, "input_param","input_embed", "variables")
    
    def test_embed_input(self):
        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config_embed_3.json' 
        T = 11
        self.solver = 'scip'
        
        # Define tranformer and execute up to embed
        transformer = TNN.Transformer(model, config_file)
        W_emb = np.random.rand(transformer.input_dim, transformer.d_model) # define rand embedding matrix
        transformer.embed_input(model, "input_param","input_embed", "variables",W_emb)
        
        self.assertIn("input_embed", dir(model))                       # check var created
        self.assertIsInstance(model.input_embed, pyo.Var)               # check data type
        self.assertTrue(hasattr(model, 'embed_constraints'))      # check constraints created
        
        # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # Solve model
        solver = SolverFactory('ipopt')
        opts = {'halt_on_ampl_error': 'yes',
           'tol': 1e-7, 'bound_relax_factor': 0.0}
        result = solver.solve(model, options=opts)
        
        # Get optimal parameters & reformat  --> (1, input_feature, sequence_element)
        optimal_parameters = get_optimal_dict(result, model)
        embed_output, _ = reformat(optimal_parameters,"input_embed") 
        
        # Calculate embedded value
        transformer_input = np.array(tir.layer_outputs_dict['input_layer_1'])
        transformer_embed = np.dot(transformer_input, W_emb) # W_emb dim: (2, 3), transformer_input dim: (1,10,2)

        # Assertions
        self.assertIsNone(np.testing.assert_array_equal(embed_output.shape, transformer_embed.shape)) # same shape
        self.assertIsNone(np.testing.assert_array_almost_equal(embed_output, transformer_embed, decimal=2))  # almost same values
    
    def test_layer_norm(self):
        
        print("======= LAYER NORM =======")

        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config_relu_2.json' 
        T = 11
        
        # Define tranformer and execute up to layer norm
        transformer = TNN.Transformer(model, config_file)
        transformer.embed_input(model, "input_param","input_embed", "variables")
        transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        
        
        # Check layer norm var and constraints created
        self.assertIn("layer_norm", dir(model))                        # check layer_norm created
        self.assertIsInstance(model.layer_norm, pyo.Var)               # check data type
        self.assertTrue(hasattr(model, 'layer_norm_constraints'))      # check constraints created
        
        # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # Solve model
        self.solver = 'scip'
        solver = SolverFactory(self.solver)
        result = solver.solve(model, tee=False) 
        optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat first layer norm block --> (1, input_feature, sequence_element)

        # opts = {'halt_on_ampl_error': 'yes',
        #    'tol': 1e-7, 'bound_relax_factor': 0.0}#scip
        # result = solver.solve(model, options=opts) #scip

        # Check layer norm output
        layer_norm_output, elements = reformat(optimal_parameters,"layer_norm") 
        LN_1_output= np.array(tir.layer_outputs_dict["layer_normalization_1"])

        # Plot expected and actual results
        # plt.figure(1, figsize=(12, 8))
        # markers = ["o-", "x--"]  # Different markers for each function
        # var = [layer_norm_output, LN_1_output]
        # labels = ['- Pyomo', '- Transformer']
        # for i in range(len(var)):
        #     plt.plot(elements, var[i][0, :, 0 ], markers[i], label=f"x values {labels[i]}")
        #     plt.plot(elements, var[i][0, :, 1 ], markers[i], label=f"u values {labels[i]}")
        # plt.title("Pyomo and Tranformer results ")
        # plt.xlabel("Sequence")
        # plt.ylabel("Magnitude")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        # print("layer norm Pyomo (as list):", [model.layer_norm[t, d].value for t in model.time_input for d in model.model_dims])
        # print("layer norm from NumPy:", transformer_output)
        
        # Assertions
        self.assertIsNone(np.testing.assert_array_equal(layer_norm_output.shape, LN_1_output.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(layer_norm_output,LN_1_output, decimal=5)) # decimal=1 # compare value with transformer output
        with self.assertRaises(ValueError):  # attempt to overwrite layer_norm var
            transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        print("- LN output formulation == LN output model")

    def test_multi_head_attention_approx(self):
        ### Right now can only support one MHA block in transformer (--> fix add uniquely named parameters for the MHA block)
        
        print("======= MULTIHEAD ATTENTION =======")
        
        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config_relu_2.json' 
        T = 11
        
        # Define tranformer and execute 
        transformer = TNN.Transformer(model, config_file)
        transformer.embed_input(model, "input_param","input_embed", "variables")
        transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        transformer.add_attention_approx(model, "layer_norm", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
        
        #Check  var and constraints created
        self.assertIn("attention_output", dir(model))                 # check layer_norm created
        self.assertIsInstance(model.attention_output, pyo.Var)        # check data type
        self.assertTrue(hasattr(model, 'attention_constraints'))      # check constraints created
        
        # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # Solve model
        self.solver = 'gurobi'
        solver = SolverFactory(self.solver, solver_io='python')#'gurobi_persistent')#self.solver, solver_io='python')
        result = solver.solve(model,tee=False)
        optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat  --> (1, input_feature, sequence_element)

        
        # Check Solve calculations
        input = np.array(tir.layer_outputs_dict['input_layer_1']).squeeze(0)
        transformer_input = np.array(tir.layer_outputs_dict["layer_normalization_1"]).squeeze(0)#np.array(tir.layer_outputs_dict['input_layer_1']).squeeze(0)
        Q = np.dot( transformer_input, np.transpose(np.array(tps.W_q),(1,0,2))) 
        K = np.dot( transformer_input, np.transpose(np.array(tps.W_k),(1,0,2))) 
        V = np.dot( transformer_input, np.transpose(np.array(tps.W_v),(1,0,2))) 

        Q = np.transpose(Q,(1,0,2)) + np.repeat(np.expand_dims(np.array(tps.b_q),axis=1),10 ,axis=1)
        K = np.transpose(K,(1,0,2)) + np.repeat(np.expand_dims(np.array(tps.b_k),axis=1),10 ,axis=1)
        V = np.transpose(V,(1,0,2)) + np.repeat(np.expand_dims(np.array(tps.b_v),axis=1),10 ,axis=1)
        
        LN_output, _ = reformat(optimal_parameters,"layer_norm")
        self.assertIsNone(np.testing.assert_array_almost_equal(np.array(tir.layer_outputs_dict["layer_normalization_1"]),LN_output, decimal =5))
        print("- MHA input formulation == MHA input model")
        
        Q_form, _ = reformat(optimal_parameters,"Q") 
        self.assertIsNone(np.testing.assert_array_equal(Q.shape, Q_form.shape))
        self.assertIsNone(np.testing.assert_array_almost_equal( Q_form,Q, decimal =5))
        print("- Query formulation == Query model")
        print("Q: ", Q_form)
        
        K_form, _ = reformat(optimal_parameters,"K") 
        self.assertIsNone(np.testing.assert_array_equal(K.shape, K_form.shape))
        self.assertIsNone(np.testing.assert_array_almost_equal( K_form,K, decimal =5))
        print("- Key formulation == Key model")
        
        V_form, _ = reformat(optimal_parameters,"V") 
        self.assertIsNone(np.testing.assert_array_equal(V.shape, V_form.shape))
        self.assertIsNone(np.testing.assert_array_almost_equal( V_form,V, decimal =5))
        print("- Value formulation == Value model")
        
        ## Check MHA output
        attention_output, elements = reformat(optimal_parameters,"attention_output") 
        MHA_output = np.array(tir.layer_outputs_dict["multi_head_attention_1"])
        self.assertIsNone(np.testing.assert_array_equal(attention_output.shape, MHA_output.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(attention_output, MHA_output , decimal=5)) # compare value with transformer output
        print("- MHA approx formulation ~= MHA output model")
        
    def test_add_residual(self):
        print("======= RESIDUAL LAYER =======")
        
        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config_relu_2.json' 
        T = 11
        self.solver = 'gurobi'
        
        # Define tranformer and execute 
        transformer = TNN.Transformer(model, config_file)
        transformer.embed_input(model, "input_param","input_embed", "variables")
        transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        transformer.add_attention_approx(model, "layer_norm", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
        transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
            
        #Check  var and constraints created
        self.assertIn("residual_1", dir(model))                 # check layer_norm created
        self.assertIsInstance(model.residual_1, pyo.Var)        # check data type
        self.assertTrue(hasattr(model, 'residual_constraints'))      # check constraints created
        
        # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # Solve model
        solver = SolverFactory(self.solver, solver_io='python')
        result = solver.solve(model, tee=False)
        optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat  --> (1, input_feature, sequence_element)
        
        ## Check Inputs
        input_embed, elements = reformat(optimal_parameters,"input_embed") 
        input = np.array(tir.layer_outputs_dict["input_layer_1"])
        self.assertIsNone(np.testing.assert_array_equal(input_embed.shape, input.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(input_embed, input, decimal=5)) # compare value with transformer output
       
        attention_output, elements = reformat(optimal_parameters,"attention_output") 
        MHA_output = np.array(tir.layer_outputs_dict["multi_head_attention_1"])
        self.assertIsNone(np.testing.assert_array_equal(attention_output.shape, MHA_output.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(attention_output, MHA_output , decimal=5)) # compare value with transformer output
        
        ## Check Output
        residual_output, elements = reformat(optimal_parameters,"residual_1") 
        residual_calc = input + MHA_output
        self.assertIsNone(np.testing.assert_array_equal(residual_output.shape, residual_calc.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(residual_output, residual_calc, decimal=5)) # compare value with transformer output
        print("- Residual output formulation == Residual output model")
    
    def test_layer_norm_2(self):
        print("======= LAYER NORM 2 =======")
        
        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config_relu_2.json' 
        T = 11
        self.solver = 'gurobi'
        
        # Define tranformer and execute 
        transformer = TNN.Transformer(model, config_file)
        transformer.embed_input(model, "input_param","input_embed", "variables")
        transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        transformer.add_attention_approx(model, "layer_norm", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
        transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
        transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
          
        #Check  var and constraints created
        self.assertIn("layer_norm_2", dir(model))                 # check layer_norm created
        self.assertIsInstance(model.layer_norm_2, pyo.Var)        # check data type
        self.assertTrue(hasattr(model, 'layer_norm_constraints'))      # check constraints created
        
        # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # Solve model
        solver = SolverFactory(self.solver, solver_io='python')
        result = solver.solve(model, tee=False)
        optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat  --> (1, input_feature, sequence_element)
        
        # ## Check Inputs
        layer_norm_2_output, _ = reformat(optimal_parameters,"layer_norm_2") 
        LN_2_output= np.array(tir.layer_outputs_dict["layer_normalization_2"])
        self.assertIsNone(np.testing.assert_array_equal(layer_norm_2_output.shape, LN_2_output.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(layer_norm_2_output, LN_2_output, decimal=5)) # compare value with transformer output
        print("- LN2 output formulation == LN2 output model")
        
    def test_FFN1(self):
        print("======= FFN1 =======")
        
        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config_relu_2.json' 
        T = 11
        self.solver = 'gurobi'
        
        # Define tranformer and execute 
        transformer = TNN.Transformer(model, config_file)
        transformer.embed_input(model, "input_param","input_embed", "variables")
        transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        transformer.add_attention_approx(model, "layer_norm", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
        transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
        transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
        transformer.add_FFN_2D(model, "layer_norm_2", "ffn_1", (10,2), tps.parameters)

        #Check  var and constraints created
        self.assertIn("ffn_1", dir(model))                 # check layer_norm created
        self.assertIsInstance(model.ffn_1, pyo.Var)        # check data type
        self.assertIsInstance(model.ffn_1_NN_Block, OmltBlock)
        self.assertTrue(hasattr(model, 'ffn_1_constraints'))      # check constraints created
        
        # # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # # Solve model
        solver = SolverFactory(self.solver, solver_io='python')
        result = solver.solve(model, tee=False)
        optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat  --> (1, input_feature, sequence_element)
        
        ffn_1_output, _ = reformat(optimal_parameters,"ffn_1") 
        FFN_out= np.array(tir.layer_outputs_dict["dense_2"])

        self.assertIsNone(np.testing.assert_array_equal(ffn_1_output.shape,  FFN_out.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(ffn_1_output,  FFN_out, decimal=5)) # compare value with transformer output
        print("- FFN1 output formulation == FFN1 output model")    
        
        
    def test_residual_2(self):
        print("======= RESIDUAL 2 =======")
        
        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config_relu_2.json' 
        T = 11
        self.solver = 'gurobi'
        
        # Define tranformer and execute 
        transformer = TNN.Transformer(model, config_file)
        transformer.embed_input(model, "input_param","input_embed", "variables")
        transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        transformer.add_attention_approx(model, "layer_norm", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
        transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
        transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
        transformer.add_FFN_2D(model, "layer_norm_2", "ffn_1", (10,2), tps.parameters) 
        transformer.add_residual_connection(model,"residual_1", "ffn_1", "residual_2")  
            
        #Check  var and constraints created
        self.assertIn("residual_2", dir(model))                 # check layer_norm created
        self.assertIsInstance(model.residual_2, pyo.Var)        # check data type
        self.assertTrue(hasattr(model, 'residual_constraints'))      # check constraints created
        
        # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # Solve model
        solver = SolverFactory(self.solver, solver_io='python')
        result = solver.solve(model, tee=False)
        optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat  --> (1, input_feature, sequence_element)
        
        residual_1, _ = reformat(optimal_parameters,"residual_1") 
        ffn_1_output, _ = reformat(optimal_parameters,"ffn_1") 
        residual_2_output, _ = reformat(optimal_parameters,"residual_2") 
        residual_out = residual_1 + ffn_1_output

        self.assertIsNone(np.testing.assert_array_equal(residual_2_output.shape,  residual_out.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(residual_2_output,  residual_out, decimal=5)) # compare value with transformer output
        print("- Residual 2 output formulation == Residual 2 output model") 
        
    def test_avg_pool(self):
        print("======= AVG POOL =======")
        
        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config_relu_2.json' 
        T = 11
        self.solver = 'gurobi'
        
        # Define tranformer layers 
        transformer = TNN.Transformer(model, config_file)
        transformer.embed_input(model, "input_param","input_embed", "variables")
        transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        transformer.add_attention_approx(model, "layer_norm", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
        transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
        transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
        transformer.add_FFN_2D(model, "layer_norm_2", "ffn_1", (10,2), tps.parameters) 
        transformer.add_residual_connection(model,"residual_1", "ffn_1", "residual_2")  
        transformer.add_avg_pool(model, "residual_2", "avg_pool")
        
        #Check  var and constraints created
        self.assertIn("avg_pool", dir(model))                 # check layer_norm created
        self.assertIsInstance(model.avg_pool, pyo.Var)        # check data type
        self.assertTrue(hasattr(model, 'avg_pool_constraints'))      # check constraints created
        
        # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # Solve model
        solver = SolverFactory(self.solver, solver_io='python')
        result = solver.solve(model, tee=False)
        optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat  --> (1, input_feature, sequence_element)
        

        avg_pool_output, _ = reformat(optimal_parameters,"avg_pool") 
        avg_pool_out = np.array(tir.layer_outputs_dict["global_average_pooling1d_1"])

        self.assertIsNone(np.testing.assert_array_equal(avg_pool_output.shape,  avg_pool_out.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(avg_pool_output,  avg_pool_out, decimal=5)) # compare value with transformer output
        print("- Avg Pool output formulation == Avg Pool output model")     
        
        
    def test_FFN2(self):
        print("======= FFN2 =======")
        
        # Define Test Case Params
        model = tps.model.clone()
        config_file = '.\\data\\toy_config_relu_2.json' 
        T = 11
        self.solver = 'gurobi'
        
        # Define tranformer and execute 
        transformer = TNN.Transformer(model, config_file)
        transformer.embed_input(model, "input_param","input_embed", "variables")
        transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        transformer.add_attention_approx(model, "layer_norm", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
        transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
        transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
        transformer.add_FFN_2D(model, "layer_norm_2", "ffn_1", (10,2), tps.parameters) 
        transformer.add_residual_connection(model,"residual_1", "ffn_1", "residual_2")  
        transformer.add_avg_pool(model, "residual_2", "avg_pool")
        transformer.add_FFN_2D(model, "avg_pool", "ffn_2", (1,2), tps.parameters)
        
        #Check  var and constraints created
        self.assertIn("ffn_2", dir(model))                 # check layer_norm created
        self.assertIsInstance(model.ffn_2, pyo.Var)        # check data type
        self.assertIsInstance(model.ffn_2_NN_Block, OmltBlock)
        self.assertTrue(hasattr(model, 'ffn_2_constraints'))      # check constraints created
        
        # # Discretize model using Backward Difference method
        discretizer = pyo.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")
        
        # # Solve model
        solver = SolverFactory(self.solver, solver_io='python')
        result = solver.solve(model, tee=False)
        optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat  --> (1, input_feature, sequence_element)
        
        ffn_2_output, _ = reformat(optimal_parameters,"ffn_2") 
        FFN_out= np.array(tir.layer_outputs_dict["dense_4"])

        self.assertIsNone(np.testing.assert_array_equal(ffn_2_output.shape,  FFN_out.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(ffn_2_output,  FFN_out, decimal=5)) # compare value with transformer output
        print("- FFN2 output formulation == FFN2 output model")    
        
        
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
    else:
        print("No optimal solution obtained.")
    
    return optimal_parameters
    
def reformat(dict, layer_name):
    """
    Reformat pyomo var to match transformer var shape: (1, input_feature, sequence_element)
    """
    key_indices = len(list(dict[layer_name].keys())[0])
    
    if key_indices == 1:
        elements = sorted(set(elem for elem in dict[layer_name].keys()))

        output = np.zeros((1,len(elements)))
        for elem, value in dict[layer_name].items():
            elem_index = elements.index(elem)
            output[0, elem_index] = value
        
        return output, elements
    if key_indices == 2:
        elements = sorted(set(elem for elem,_ in dict[layer_name].keys()))
        features = sorted(set(feat for _, feat in dict[layer_name].keys())) #x : '0', u: '1' which matches transformer array

        output = np.zeros((1,len(elements), len(features)))
        for (elem, feat), value in dict[layer_name].items():
            elem_index = elements.index(elem)
            feat_index = features.index(feat)
        
            output[0, elem_index, feat_index] = value
        
        return output, elements
    if key_indices == 3:
        key_1 = sorted(set(array[0] for array in dict[layer_name].keys()))
        key_2 = sorted(set(array[1] for array in dict[layer_name].keys()))
        key_3 = sorted(set(array[2] for array in dict[layer_name].keys()))

        output = np.zeros((len(key_1),len(key_2), len(key_3)))
        for (k1, k2, k3), value in dict[layer_name].items():
            k1_index = key_1.index(k1)
            k2_index = key_2.index(k2)
            k3_index = key_3.index(k3)
        
            output[k1_index, k2_index, k3_index] = value
        
        return output, key_1
    raise ValueError('Reformat only handles layers with 1, 2 or 3 keys indexing the layer values')

# ------- MAIN -----------------------------------------------------------------------------------
if __name__ == '__main__': 
    unittest.main() 

