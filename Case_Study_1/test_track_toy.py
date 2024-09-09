# External imports
import pyomo.environ as pyo
import numpy as np
from pyomo import dae
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import unittest
import os
from omlt import OmltBlock
import torch
from helpers.print_stats import solve_pyomo, solve_gurobipy
import helpers.convert_pyomo as convert_pyomo
from gurobipy import Model, GRB
from gurobi_ml import add_predictor_constr
from helpers.GUROBI_ML_helper import get_inputs_gurobipy_FNN

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

# Import from repo file
import transformer_b as TNN
from trained_transformer.Tmodel import TransformerModel
import helpers.extract_from_pretrained as extract_from_pretrained

"""
Test each module of transformer for optimal control toy tnn 1
"""
# ------- Transformer Test Class ------------------------------------------------------------------------------------
class TestTransformer(unittest.TestCase):    

    def test_instantiation_input(self): #, model, pyomo_input_name ,transformer_input):
        m = model.clone()
        enc_input_name = "enc_input"
        dec_input_name = "dec_input"
        
        # create optimization transformer
        transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", m) 
        
        # define sets for inputs
        enc_dim_1 = transformer.N 
        dec_dim_1 = transformer.N 
        transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
        transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
        transformer.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) # - 2
        transformer.M.model_dims = pyo.Set(initialize= list(range(transformer.d_model)))
        transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
        enc_flag = False
        dec_flag = False
        
        # Add TNN input vars
        transformer.M.enc_input= pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        transformer.M.dec_input = pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        
        # add constraints to trained TNN input
        m.tnn_input_constraints = pyo.ConstraintList()
        indices = []
        for set in str(transformer.M.enc_input.index_set()).split("*"):
            indices.append( getattr(m, set) )
        for tnn_index, index in zip(indices[0], m.time_history):
            print(tnn_index, index)
            m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].first()]== m.history_loc1[index])
            m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].last()] == m.history_loc2[index]) 
            
        indices = []
        for set in str(transformer.M.dec_input.index_set()).split("*"):
            indices.append( getattr(m, set) )
        for t_index, t in enumerate(m.time):
            index = t_index + 1 # 1 indexing
            
            if index > pred_len and index < tt + pred_len + 1:
                m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(index - pred_len), indices[1].first()] == m.loc1[t])
                m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(index - pred_len), indices[1].last()]  == m.loc2[t])
                
        # Set objective
        m.obj = pyo.Objective(
            expr= sum((m.x1[t] - m.loc1[t])**2 + (m.x2[t] - m.loc2[t])**2 for t in m.time), sense=1
        )  # -1: maximize, +1: minimize (default)
    
    
        # Convert to gurobipy
        gurobi_model, _, _ = convert_pyomo.to_gurobi(m)
        
        
        # Solve
        gurobi_model.optimize()

        if gurobi_model.status == GRB.INFEASIBLE:
                gurobi_model.computeIIS()
                gurobi_model.write("pytorch_model.ilp")
        
        # Get input value (before solve)
        parameters = {}
        for v in gurobi_model.getVars():
            #print(f'var name: {v.varName}, var type {type(v)}')
            # print(v.LB, v.UB)
            if "[" in v.varName:
                name = v.varname.split("[")[0]
                if name in parameters.keys():
                    parameters[name] += [v.x]
                else:
                    parameters[name] = [v.x]
            else:    
                parameters[v.varName] = v.x

        # model input
        model_enc_input = np.array(parameters[enc_input_name])
        model_dec_input = np.array(parameters[dec_input_name])
        
        expected_enc_input = transformer_input[0, 0:tt, :].flatten() 
        expected_dec_input = transformer_input[0, 1:tt+1, :].flatten() 

        # Assertions
        self.assertIsNone(np.testing.assert_array_equal(model_enc_input.shape, expected_enc_input.shape)) # pyomo input data and transformer input data must be the same shape
        self.assertIsNone(np.testing.assert_array_almost_equal(model_enc_input, expected_enc_input, decimal = 7))             # both inputs must be equal
        
        self.assertIsNone(np.testing.assert_array_equal(model_dec_input.shape, expected_dec_input.shape)) # pyomo input data and transformer input data must be the same shape
        self.assertIsNone(np.testing.assert_array_almost_equal(model_dec_input, expected_dec_input, decimal = 7))             # both inputs must be equal
        

    def test_embed_input(self):
        # create optimization transformer
        m = model.clone()
        transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", m) 
        
        # define sets for inputs
        enc_dim_1 = transformer.N 
        dec_dim_1 = transformer.N 
        transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
        transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
        transformer.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) # - 2
        transformer.M.model_dims = pyo.Set(initialize= list(range(transformer.d_model)))
        transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
        enc_flag = False
        dec_flag = False
        
        # Add TNN input vars
        transformer.M.enc_input= pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        transformer.M.dec_input = pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)

        # Add Embedding (linear) layer
        embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
        layer = "linear_1"
        W_linear = parameters[layer,'W']
        try:
            b_linear = parameters[layer,'b']
        except:
            b_linear = None
        transformer.embed_input( "enc_input", "enc_linear_1", embed_dim, W_linear, b_linear)
        transformer.embed_input( "dec_input", "dec_linear_1", embed_dim, W_linear, b_linear)
        
        # add constraints to trained TNN input
        m.tnn_input_constraints = pyo.ConstraintList()
        indices = []
        for set in str(transformer.M.enc_input.index_set()).split("*"):
            indices.append( getattr(m, set) )
        for tnn_index, index in zip(indices[0], m.time_history):
            print(tnn_index, index)
            m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].first()]== m.history_loc1[index])
            m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].last()] == m.history_loc2[index]) 
            
        indices = []
        for set in str(transformer.M.dec_input.index_set()).split("*"):
            indices.append( getattr(m, set) )
        for t_index, t in enumerate(m.time):
            index = t_index + 1 # 1 indexing
            
            if index > pred_len and index < tt + pred_len + 1:
                m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(index - pred_len), indices[1].first()] == m.loc1[t])
                m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(index - pred_len), indices[1].last()]  == m.loc2[t])
                
        # Set objective
        m.obj = pyo.Objective(
            expr= sum((m.x1[t] - m.loc1[t])**2 + (m.x2[t] - m.loc2[t])**2 for t in m.time), sense=1
        )  # -1: maximize, +1: minimize (default)
    
    
        # Convert to gurobipy
        gurobi_model, _, _ = convert_pyomo.to_gurobi(m)

        # Check new model attributes added
        self.assertIn("enc_linear_1", dir(transformer.M))                        # check var created
        self.assertIn("dec_linear_1", dir(transformer.M))                        # check var created
        self.assertIsInstance(transformer.M.enc_linear_1, pyo.Var)               # check data type
        self.assertIsInstance(transformer.M.dec_linear_1, pyo.Var)               # check data type
        self.assertTrue(hasattr(transformer.M, 'embed_constraints'))            # check constraints created
        
        # Convert to gurobipy
        gurobi_model, _ , _ = convert_pyomo.to_gurobi(m)
        gurobi_model.optimize()

        if gurobi_model.status == GRB.OPTIMAL:
            optimal_parameters = {}
            for v in gurobi_model.getVars():
                #print(f'var name: {v.varName}, var type {type(v)}')
                if "[" in v.varName:
                    name = v.varname.split("[")[0]
                    if name in optimal_parameters.keys():
                        optimal_parameters[name] += [v.x]
                    else:
                        optimal_parameters[name] = [v.x]
                else:    
                    optimal_parameters[v.varName] = v.x
            
        # outputs
        
        embed_enc = np.array(optimal_parameters["enc_linear_1"])
        embed_dec = np.array(optimal_parameters["dec_linear_1"])
        
        expected_enc = np.array(list(layer_outputs_dict['linear1'])[0]).flatten()
        expected_dec = np.array(list(layer_outputs_dict['linear1'])[1]).flatten()
        
        # Assertions
        self.assertIsNone(np.testing.assert_array_equal(embed_enc.shape, expected_enc.shape)) # same shape
        self.assertIsNone(np.testing.assert_array_almost_equal(embed_enc, expected_enc, decimal=2))  # almost same values
    
        self.assertIsNone(np.testing.assert_array_equal(embed_dec.shape, expected_dec.shape)) # same shape
        self.assertIsNone(np.testing.assert_array_almost_equal(embed_dec, expected_dec, decimal=2))  # almost same values
    
    
    # def test_layer_norm(self):
        
    #     print("======= LAYER NORM =======")

    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     # config_file = '.\\data\\toy_config_relu_2.json' 
    #     config_file = tps.config_file 
    #     T = 11
        
    #     # Define tranformer and execute up to layer norm
    #     transformer = TNN.Transformer(model, config_file, "time_input")  
    #     transformer.embed_input(model, "input_param","input_embed", "variables")
    #     transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
        
        
    #     # Check layer norm var and constraints created
    #     self.assertIn("layer_norm", dir(model))                        # check layer_norm created
    #     self.assertIsInstance(model.layer_norm, pyo.Var)               # check data type
    #     self.assertTrue(hasattr(model, 'layer_norm_constraints'))      # check constraints created
        
        
    #     # Convert to gurobipy
    #     gurobi_model, _  = convert_pyomo.to_gurobi(model)
    #     gurobi_model.optimize()

    #     if gurobi_model.status == GRB.OPTIMAL:
    #         optimal_parameters = {}
    #         for v in gurobi_model.getVars():
    #             #print(f'var name: {v.varName}, var type {type(v)}')
    #             if "[" in v.varName:
    #                 name = v.varname.split("[")[0]
    #                 if name in optimal_parameters.keys():
    #                     optimal_parameters[name] += [v.x]
    #                 else:
    #                     optimal_parameters[name] = [v.x]
    #             else:    
    #                 optimal_parameters[v.varName] = v.x

    #     # model output
    #     layer_norm_output = np.array(optimal_parameters["layer_norm"])
        
    #     # transformer output
    #     LN_1_output= np.array(layer_outputs_dict["layer_normalization_1"]).flatten()
        
    #     # Assertions
    #     self.assertIsNone(np.testing.assert_array_equal(layer_norm_output.shape, LN_1_output.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(layer_norm_output,LN_1_output, decimal=5)) # decimal=1 # compare value with transformer output
    #     with self.assertRaises(ValueError):  # attempt to overwrite layer_norm var
    #         transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
    #     print("- LN output formulation == LN output model")

    # def test_multi_head_attention(self):
    #     print("======= MULTIHEAD ATTENTION =======")

    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     # config_file = '.\\data\\toy_config_relu_2.json' 
    #     config_file = tps.config_file 
    #     T = 11
        
    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(model, config_file, "time_input")  
    #     transformer.embed_input(model, "input_param","input_embed", "variables")
    #     transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
    #     transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
        
    #     #Check  var and constraints created
    #     self.assertIn("attention_output", dir(model))                 # check layer_norm created
    #     self.assertIsInstance(model.attention_output, pyo.Var)        # check data type
    #     #self.assertTrue(hasattr(model, 'Block_attention_output.constraints'))      # check constraints created
        
        
        
    #     # Convert to gurobipy
    #     gurobi_model, _  = convert_pyomo.to_gurobi(model)
    #     gurobi_model.optimize()

    #     if gurobi_model.status == GRB.OPTIMAL:
    #         optimal_parameters = {}
    #         for v in gurobi_model.getVars():
    #             #print(f'var name: {v.varName}, var type {type(v)}')
    #             if "[" in v.varName:
    #                 name = v.varname.split("[")[0]
    #                 if name in optimal_parameters.keys():
    #                     optimal_parameters[name] += [v.x]
    #                 else:
    #                     optimal_parameters[name] = [v.x]
    #             else:    
    #                 optimal_parameters[v.varName] = v.x

    #     # model output
    #     LN_output = np.array(optimal_parameters["layer_norm"])
    #     Q_form = np.array(optimal_parameters["Block_attention_output.Q"])
    #     K_form = np.array(optimal_parameters["Block_attention_output.K"])
    #     V_form = np.array(optimal_parameters["Block_attention_output.V"])
    #     attention_output= np.array(optimal_parameters["attention_output"]) 
        
        
        
    #     # Check Solve calculations
    #     input = np.array(layer_outputs_dict['input_layer_1']).squeeze(0)
    #     transformer_input = np.array(layer_outputs_dict["layer_normalization_1"]).squeeze(0)#np.array(layer_outputs_dict['input_layer_1']).squeeze(0)
    #     Q = np.dot( transformer_input, np.transpose(np.array(tps.W_q),(1,0,2))) 
    #     K = np.dot( transformer_input, np.transpose(np.array(tps.W_k),(1,0,2))) 
    #     V = np.dot( transformer_input, np.transpose(np.array(tps.W_v),(1,0,2))) 

    #     Q = np.transpose(Q,(1,0,2)) + np.repeat(np.expand_dims(np.array(tps.b_q),axis=1), transformer.N ,axis=1)
    #     K = np.transpose(K,(1,0,2)) + np.repeat(np.expand_dims(np.array(tps.b_k),axis=1), transformer.N ,axis=1)
    #     V = np.transpose(V,(1,0,2)) + np.repeat(np.expand_dims(np.array(tps.b_v),axis=1), transformer.N ,axis=1)
        
    #     Q = Q.flatten()
    #     K = K.flatten()
    #     V = V.flatten()
        
    #     self.assertIsNone(np.testing.assert_array_almost_equal(np.array(layer_outputs_dict["layer_normalization_1"]).flatten(),LN_output, decimal =5))
    #     print("- MHA input formulation == MHA input model")
        
    #     self.assertIsNone(np.testing.assert_array_equal(Q.shape, Q_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( Q_form,Q, decimal =5))
    #     print("- Query formulation == Query model")
        
    #     self.assertIsNone(np.testing.assert_array_equal(K.shape, K_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( K_form,K, decimal =5))
    #     print("- Key formulation == Key model")
        
    #     self.assertIsNone(np.testing.assert_array_equal(V.shape, V_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( V_form,V, decimal =5))
    #     print("- Value formulation == Value model")
        
    #     ## Check MHA output
    #     MHA_output = np.array(layer_outputs_dict["multi_head_attention_1"]).flatten()
    #     self.assertIsNone(np.testing.assert_array_equal(attention_output.shape, MHA_output.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(attention_output, MHA_output , decimal=5)) # compare value with transformer output
    #     print("- MHA output formulation == MHA output model")
        
    # def test_add_residual(self):
    #     print("======= RESIDUAL LAYER =======")
        
    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     # config_file = '.\\data\\toy_config_relu_2.json' 
    #     config_file = tps.config_file 
    #     T = 11

    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(model, config_file, "time_input")  
    #     transformer.embed_input(model, "input_param","input_embed", "variables")
    #     transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
    #     transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
    #     transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
            
    #     #Check  var and constraints created
    #     self.assertIn("residual_1", dir(model))                 # check layer_norm created
    #     self.assertIsInstance(model.residual_1, pyo.Var)        # check data type
    #     self.assertTrue(hasattr(model, 'residual_constraints'))      # check constraints created
        
    #     # Convert to gurobipy
    #     gurobi_model, _  = convert_pyomo.to_gurobi(model)
    #     gurobi_model.optimize()

    #     if gurobi_model.status == GRB.OPTIMAL:
    #         optimal_parameters = {}
    #         for v in gurobi_model.getVars():
    #             #print(f'var name: {v.varName}, var type {type(v)}')
    #             if "[" in v.varName:
    #                 name = v.varname.split("[")[0]
    #                 if name in optimal_parameters.keys():
    #                     optimal_parameters[name] += [v.x]
    #                 else:
    #                     optimal_parameters[name] = [v.x]
    #             else:    
    #                 optimal_parameters[v.varName] = v.x
                    
    #     ## Check Inputs
    #     input_embed = np.array(optimal_parameters["input_embed"]) 
    #     input = np.array(layer_outputs_dict["input_layer_1"]).flatten()
    #     self.assertIsNone(np.testing.assert_array_equal(input_embed.shape, input.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(input_embed, input, decimal=5)) # compare value with transformer output
       
    #     attention_output = np.array(optimal_parameters["attention_output"]) 
    #     MHA_output = np.array(layer_outputs_dict["multi_head_attention_1"]).flatten()
    #     self.assertIsNone(np.testing.assert_array_equal(attention_output.shape, MHA_output.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(attention_output, MHA_output , decimal=5)) # compare value with transformer output
        
    #     ## Check Output
    #     residual_output = np.array(optimal_parameters["residual_1"]) 
    #     residual_calc = input + MHA_output
    
    #     self.assertIsNone(np.testing.assert_array_equal(residual_output.shape, residual_calc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(residual_output, residual_calc, decimal=5)) # compare value with transformer output
    #     print("- Residual output formulation == Residual output model")
    
    # def test_layer_norm_2(self):
    #     print("======= LAYER NORM 2 =======")
        
    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     # config_file = '.\\data\\toy_config_relu_2.json' 
    #     config_file = tps.config_file 
    #     T = 11
        
    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(model, config_file, "time_input")  
    #     transformer.embed_input(model, "input_param","input_embed", "variables")
    #     transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
    #     transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
    #     transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
    #     transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
          
    #     #Check  var and constraints created
    #     self.assertIn("layer_norm_2", dir(model))                 # check layer_norm created
    #     self.assertIsInstance(model.layer_norm_2, pyo.Var)        # check data type
    #     self.assertTrue(hasattr(model, 'layer_norm_constraints'))      # check constraints created
        
    #     # Convert to gurobipy
    #     gurobi_model, _  = convert_pyomo.to_gurobi(model)
    #     gurobi_model.optimize()

    #     if gurobi_model.status == GRB.OPTIMAL:
    #         optimal_parameters = {}
    #         for v in gurobi_model.getVars():
    #             #print(f'var name: {v.varName}, var type {type(v)}')
    #             if "[" in v.varName:
    #                 name = v.varname.split("[")[0]
    #                 if name in optimal_parameters.keys():
    #                     optimal_parameters[name] += [v.x]
    #                 else:
    #                     optimal_parameters[name] = [v.x]
    #             else:    
    #                 optimal_parameters[v.varName] = v.x
        
    #     # ## Check Inputs
    #     layer_norm_2_output = np.array(optimal_parameters["layer_norm_2"]) 
    #     LN_2_output= np.array(layer_outputs_dict["layer_normalization_2"]).flatten()
    #     self.assertIsNone(np.testing.assert_array_equal(layer_norm_2_output.shape, LN_2_output.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(layer_norm_2_output, LN_2_output, decimal=5)) # compare value with transformer output
    #     print("- LN2 output formulation == LN2 output model")
        
    # def test_FFN1(self):
    #     print("======= FFN1 =======")
        
    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     # config_file = '.\\data\\toy_config_relu_2.json' 
    #     config_file = tps.config_file 
    #     T = 11
        
    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(model, config_file, "time_input")  
    #     transformer.embed_input(model, "input_param","input_embed", "variables")
    #     transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
    #     transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
    #     transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
    #     transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
    #     nn, input_nn, output_nn = transformer.get_fnn(model, "layer_norm_2", "ffn_1", "ffn_1", (10,2), tps.parameters)

    #     # # Convert to gurobipy
    #     gurobi_model, map_var = convert_pyomo.to_gurobi(model)

    #     ## Add  NN to gurobi model
    #     inputs_1, outputs_1 = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
    #     pred_constr1 = add_predictor_constr(gurobi_model, nn, inputs_1, outputs_1)
    #     gurobi_model.update()
    #     #pred_constr.print_stats()
        
    #     ## Optimizes
    #     gurobi_model.optimize()

    #     if gurobi_model.status == GRB.OPTIMAL:
    #         optimal_parameters = {}
    #         for v in gurobi_model.getVars():
    #             #print(f'var name: {v.varName}, var type {type(v)}')
    #             if "[" in v.varName:
    #                 name = v.varname.split("[")[0]
    #                 if name in optimal_parameters.keys():
    #                     optimal_parameters[name] += [v.x]
    #                 else:
    #                     optimal_parameters[name] = [v.x]
    #             else:    
    #                 optimal_parameters[v.varName] = v.x
                    
                    
    #     # Check outputs           
    #     ffn_1_output = np.array(optimal_parameters["ffn_1"]) 
    #     FFN_out= np.array(layer_outputs_dict["dense_2"]).flatten()

    #     self.assertIsNone(np.testing.assert_array_equal(ffn_1_output.shape,  FFN_out.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(ffn_1_output,  FFN_out, decimal=5)) # compare value with transformer output
    #     print("- FFN1 output formulation == FFN1 output model")    
        
        
    # def test_residual_2(self):
    #     print("======= RESIDUAL 2 =======")
        
    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     # config_file = '.\\data\\toy_config_relu_2.json' 
    #     config_file = tps.config_file 
    #     T = 11
        
    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(model, config_file, "time_input")  
    #     transformer.embed_input(model, "input_param","input_embed", "variables")
    #     transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
    #     transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
    #     transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
    #     transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
    #     nn, input_nn, output_nn = transformer.get_fnn(model, "layer_norm_2", "ffn_1", "ffn_1", (10,2), tps.parameters)
    #     transformer.add_residual_connection(model,"residual_1", "ffn_1", "residual_2")  
            
    #     #Check  var and constraints created
    #     self.assertIn("residual_2", dir(model))                 # check layer_norm created
    #     self.assertIsInstance(model.residual_2, pyo.Var)        # check data type
    #     self.assertTrue(hasattr(model, 'residual_constraints'))      # check constraints created
        
    #     #Convert to gurobipy
    #     gurobi_model, map_var = convert_pyomo.to_gurobi(model)

    #     ## Add  NN to gurobi model
    #     inputs_1, outputs_1 = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
    #     pred_constr1 = add_predictor_constr(gurobi_model, nn, inputs_1, outputs_1)
    #     gurobi_model.update()
    #     #pred_constr.print_stats()
        
    #     ## Optimizes
    #     gurobi_model.optimize()

    #     if gurobi_model.status == GRB.OPTIMAL:
    #         optimal_parameters = {}
    #         for v in gurobi_model.getVars():
    #             #print(f'var name: {v.varName}, var type {type(v)}')
    #             if "[" in v.varName:
    #                 name = v.varname.split("[")[0]
    #                 if name in optimal_parameters.keys():
    #                     optimal_parameters[name] += [v.x]
    #                 else:
    #                     optimal_parameters[name] = [v.x]
    #             else:    
    #                 optimal_parameters[v.varName] = v.x
                    
                    
    #     # Check outputs
    #     residual_1 = np.array(optimal_parameters["residual_1"]) 
    #     ffn_1_output = np.array(optimal_parameters["ffn_1"]) 
    #     residual_2_output = np.array(optimal_parameters["residual_2"]) 
    #     residual_out = residual_1 + ffn_1_output

    #     self.assertIsNone(np.testing.assert_array_equal(residual_2_output.shape,  residual_out.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(residual_2_output,  residual_out, decimal=5)) # compare value with transformer output
    #     print("- Residual 2 output formulation == Residual 2 output model") 
        
    # def test_avg_pool(self):
    #     print("======= AVG POOL =======")
        
    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     # config_file = '.\\data\\toy_config_relu_2.json' 
    #     config_file = tps.config_file 
    #     T = 11
        
    #     # Define tranformer layers 
    #     transformer = TNN.Transformer(model, config_file, "time_input")  
    #     transformer.embed_input(model, "input_param","input_embed", "variables")
    #     transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
    #     transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
    #     transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
    #     transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
    #     nn, input_nn, output_nn = transformer.get_fnn(model, "layer_norm_2", "ffn_1", "ffn_1", (10,2), tps.parameters)
    #     transformer.add_residual_connection(model,"residual_1", "ffn_1", "residual_2")  
    #     transformer.add_avg_pool(model, "residual_2", "avg_pool")
        
    #     #Check  var and constraints created
    #     self.assertIn("avg_pool", dir(model))                 # check layer_norm created
    #     self.assertIsInstance(model.avg_pool, pyo.Var)        # check data type
    #     self.assertTrue(hasattr(model, 'avg_pool_constr_avg_pool'))      # check constraints created

    #     # # Convert to gurobipy
    #     gurobi_model, map_var = convert_pyomo.to_gurobi(model)

    #     ## Add  NN to gurobi model
    #     inputs_1, outputs_1 = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
    #     pred_constr1 = add_predictor_constr(gurobi_model, nn, inputs_1, outputs_1)
    #     gurobi_model.update()
    #     #pred_constr.print_stats()
        
    #     ## Optimizes
    #     gurobi_model.optimize()

    #     if gurobi_model.status == GRB.OPTIMAL:
    #         optimal_parameters = {}
    #         for v in gurobi_model.getVars():
    #             #print(f'var name: {v.varName}, var type {type(v)}')
    #             if "[" in v.varName:
    #                 name = v.varname.split("[")[0]
    #                 if name in optimal_parameters.keys():
    #                     optimal_parameters[name] += [v.x]
    #                 else:
    #                     optimal_parameters[name] = [v.x]
    #             else:    
    #                 optimal_parameters[v.varName] = v.x
                    
    #     #Check outputs
    #     avg_pool_output = np.array(optimal_parameters["avg_pool"]) 
    #     avg_pool_out = np.array(layer_outputs_dict["global_average_pooling1d_1"]).flatten()

    #     self.assertIsNone(np.testing.assert_array_equal(avg_pool_output.shape,  avg_pool_out.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(avg_pool_output,  avg_pool_out, decimal=5)) # compare value with transformer output
    #     print("- Avg Pool output formulation == Avg Pool output model")     
        
        
    # def test_FFN2(self):
    #     print("======= FFN2 =======")
        
    #     # Define Test Case Params
    #     model = tps.model.clone()
    #     # config_file = '.\\data\\toy_config_relu_2.json' 
    #     config_file = tps.config_file 
    #     T = 11
        
    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(model, config_file, "time_input")  
    #     transformer.embed_input(model, "input_param","input_embed", "variables")
    #     transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")
    #     transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
    #     transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
    #     transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
    #     nn, input_nn, output_nn = transformer.get_fnn(model, "layer_norm_2", "ffn_1", "ffn_1", (10,10), tps.parameters)
    #     transformer.add_residual_connection(model,"residual_1", "ffn_1", "residual_2")  
    #     transformer.add_avg_pool(model, "residual_2", "avg_pool")
    #     nn2, input_nn2, output_nn2 = transformer.get_fnn(model, "avg_pool", "ffn_2", "ffn_2", (1,2), tps.parameters)

    #     # # Convert to gurobipy
    #     gurobi_model, map_var = convert_pyomo.to_gurobi(model)

    #     ## Add FNN1 to gurobi model
    #     inputs_1, outputs_1 = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
    #     pred_constr1 = add_predictor_constr(gurobi_model, nn, inputs_1, outputs_1)
        
    #     inputs_2, outputs_2 = get_inputs_gurobipy_FNN(input_nn2, output_nn2, map_var)
    #     pred_constr2 = add_predictor_constr(gurobi_model, nn2, inputs_2, outputs_2)
    #     gurobi_model.update()
    #     #pred_constr.print_stats()
        
    #     ## Optimizes
    #     gurobi_model.optimize()

    #     if gurobi_model.status == GRB.OPTIMAL:
    #         optimal_parameters = {}
    #         for v in gurobi_model.getVars():
    #             #print(f'var name: {v.varName}, var type {type(v)}')
    #             if "[" in v.varName:
    #                 name = v.varname.split("[")[0]
    #                 if name in optimal_parameters.keys():
    #                     optimal_parameters[name] += [v.x]
    #                 else:
    #                     optimal_parameters[name] = [v.x]
    #             else:    
    #                 optimal_parameters[v.varName] = v.x
        
    #     #Check outputs
    #     ffn_2_output = np.array(optimal_parameters["ffn_2"]) 
    #     FFN_out = np.array(layer_outputs_dict["dense_4"]).flatten()
    #     print(ffn_2_output,  FFN_out)
    #     self.assertIsNone(np.testing.assert_array_equal(ffn_2_output.shape,  FFN_out.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(ffn_2_output,  FFN_out, decimal=5)) # compare value with transformer output
    #     print("- FFN2 output formulation == FFN2 output model")   
        
    #     print("Output: ", ffn_2_output)
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
    key_indices = len(list(dict[layer_name].keys())[0])
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
    raise ValueError('Reformat only handles layers with 2 or 3 keys indexing the layer values')

# ------- MAIN -----------------------------------------------------------------------------------
if __name__ == '__main__': 
    # instantiate pyomo model component
    model = pyo.ConcreteModel(name="(TOY_TRANFORMER)")

    # define constants
    T_end = 0.0105
    steps = 11 ##CHANGE THIS ##
    time = np.linspace(0, T_end, num=steps)
    dt = time[1] - time[0]
    tt = 10 # sequence size
    time_history = time[0:tt]
    pred_len = 1

    g = 9.81
    v_l1 = 0.2
    v_l2 = 1.5

    src = np.array([np.random.rand(1)*time[0:-1] , (2*np.random.rand(1) * time[0:-1]) - (0.5 * 9.81* time[0:-1] * time[0:-1])])# random sample input [x1_targte, x2_target]
    src = src.transpose(1,0)

    # define sets
    model.time = pyo.Set(initialize=time)
    model.time_history = pyo.Set(initialize=time[0:-1])
    # print(time, time[0:-1])
    
    # define parameters
    def target_location_rule(M, t):
        return v_l1 * t
    model.history_loc1 = pyo.Param(model.time_history, rule=target_location_rule) 

    def target_location2_rule(M, t):
        return (v_l2*t) - (0.5 * g * (t**2))
    model.history_loc2 = pyo.Param(model.time_history, rule=target_location2_rule) 
    
    history_loc1 = np.array([v for k,v in model.history_loc1.items()])
    history_loc2 = np.array([v for k,v in model.history_loc2.items()])
    print(history_loc1, history_loc2)

    bounds_target = (-3,3)
    # define variables
    model.loc1 = pyo.Var(model.time, bounds = bounds_target )
    model.loc2 = pyo.Var(model.time, bounds = bounds_target )

    model.x1 = pyo.Var(model.time) # distance path
    model.v1 = pyo.Var() # initial velocity of cannon ball

    model.x2 = pyo.Var(model.time) # height path
    model.v2 = pyo.Var() # initial velocity of cannon ball

    #model.T = pyo.Var(within=model.time)# time when cannon ball hits target

    # define initial conditions
    model.x1_constr = pyo.Constraint(expr= model.x1[0] == 0) 
    model.x2_constr = pyo.Constraint(expr= model.x2[0] == 0) 

    # define constraints
    def v1_rule(M, t):
        return M.x1[t] == M.v1 * t
    model.v1_constr = pyo.Constraint(model.time, rule=v1_rule) 

    def v2_rule(M, t):
        return M.x2[t] == (M.v2 * t) - (0.5*g * (t**2))
    model.v2_constr = pyo.Constraint(model.time, rule=v2_rule)

    model.v1_pos_constr = pyo.Constraint(expr = model.v1 >= 0)
    model.v2_pos_constr = pyo.Constraint(expr = model.v2 >= 0)

    def loc1_rule(M, t):
        return M.loc1[t] == model.history_loc1[t]
    model.loc1_constr = pyo.Constraint(model.time_history, rule=loc1_rule)

    def loc2_rule(M, t):
        return M.loc2[t] == model.history_loc2[t]
    model.loc2_constr = pyo.Constraint(model.time_history, rule=loc2_rule)

    # load trained transformer
    sequence_size = tt
    head_size = 4
    device = torch.device('cpu')
    tnn_path = ".\\trained_transformer\\toy_pytorch_model_1.pt"
    tnn_model = TransformerModel(input_dim=2, output_dim =2, d_model=12, nhead=head_size, num_encoder_layers=1, num_decoder_layers=1)
    tnn_model.load_state_dict(torch.load(tnn_path, map_location=device))
    
    # Fix model solution
    input_x1 =   v_l1 * time  
    input_x2 =  (v_l2*time) - (0.5 * g * (time*time))
    model.x1_last = pyo.Constraint(expr= input_x1[-1] == model.loc1[model.time.last()])
    model.x2_last = pyo.Constraint(expr= input_x2[-1]  == model.loc2[model.time.last()])

    # get intermediate results dictionary for optimal input values
    transformer_input = np.array([[ [x1,x2] for x1,x2 in zip(input_x1, input_x2)]], dtype=np.float32)
    print(transformer_input.shape)
    
    layer_names, parameters, _, enc_dec_count, layer_outputs_dict = extract_from_pretrained.get_pytorch_learned_parameters(tnn_model,transformer_input[0, 0:tt, :] ,transformer_input[0, 1:tt+1, :], head_size, tt)
    
    # print(dict_outputs)
    #output_dense_4 = layer_outputs_dict["dense_4"]
    
    # unit test
    unittest.main() 

