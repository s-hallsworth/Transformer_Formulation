# External imports
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
import unittest
import os
import MINLP_tnn.helpers.convert_pyomo as convert_pyomo
from gurobipy import GRB
from gurobi_ml import add_predictor_constr
from MINLP_tnn.helpers.GUROBI_ML_helper import get_inputs_gurobipy_FFN

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

# Import from repo file
import transformer_b_flag_cuts as TNN
import MINLP_tnn.helpers.extract_from_pretrained as extract_from_pretrained

"""
Test each module of transformer for optimal control toy tnn 1
"""
# ------- Transformer Test Class ------------------------------------------------------------------------------------
class TestTransformer(unittest.TestCase):    
    # def test_input(self):
    #     print("======= INPUT =======")
        
    #     # Define Test Case Params
    #     m = model.clone()
    #     seq_len = tt
        
    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(hyper_params, m)  
    #     transformer.add_input_var("input_embed", dims=(seq_len, transformer.input_dim), bounds=(-3,3))
        
    #     # add constraints to trained TNN input
    #     m.tnn_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.input_embed.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.time_history):
    #         m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].first()]== m.x1[index])
    #         m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].last()] == m.x2[index]) 
        
    #     # # Convert to gurobipy
    #     gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
        
    #     ## Optimizes
    #     # gurobi_model.setParam('DualReductions',0)
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
                    
    #     if gurobi_model.status == GRB.INFEASIBLE:
    #             gurobi_model.computeIIS()
    #             gurobi_model.write("pytorch_model.ilp")
                
        
    #     ## Check output
    #     actual = np.array(list(optimal_parameters['input_embed']))
    #     expected = input[0,0:seq_len,:].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(actual.shape, expected.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(actual, expected, decimal=5)) # compare value with transformer output
    #     print("- input formulation == input model")
      
    # def test_LN1(self):
    #     print("======= LN1 =======")
        
    #     # Define Test Case Params
    #     m = model.clone()
    #     seq_len = tt
    #     layer = 'multi_head_attention_1'

    #     gamma1 = parameters['layer_normalization_1', 'gamma']
    #     beta1  = parameters['layer_normalization_1', 'beta']

        
    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(hyper_params, m)  
    #     transformer.add_input_var("input_embed", dims=(seq_len, transformer.input_dim), bounds=(-3,3))
    #     transformer.add_layer_norm( "input_embed", "layer_norm", gamma1, beta1)
        
        
    #     # add constraints to trained TNN input
    #     m.tnn_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.input_embed.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.time_history):
    #         m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].first()]== m.x1[index])
    #         m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].last()] == m.x2[index]) 
            
        
    #     # # Convert to gurobipy
    #     gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
        
    #     ## Optimizes
    #     # gurobi_model.setParam('DualReductions',0)
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
                    
    #     if gurobi_model.status == GRB.INFEASIBLE:
    #             gurobi_model.computeIIS()
    #             gurobi_model.write("pytorch_model.ilp")
                
    #     ## Check output
    #     actual = np.array(list(optimal_parameters['input_embed']))
    #     expected = input[0,0:seq_len,:].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(actual.shape, expected.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(actual, expected, decimal=7)) # compare value with transformer output
    #     print("- input formulation == input model")            
        
    #     ## Check output
    #     actual = np.array(list(optimal_parameters['layer_norm']))
    #     expected = np.array(layer_outputs_dict['layer_normalization_1'])[0].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(actual.shape, expected.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(actual, expected, decimal=3)) # compare value with transformer output
    #     print("- LN1 formulation == LN1 model")
        
    #     print(actual)
                    
    # def test_MHA(self):
    #     print("======= MHA =======")
        
    #     # Define Test Case Params
    #     m = model.clone()
    #     seq_len = tt
    #     layer = 'multi_head_attention_1'

    #     gamma1 = parameters['layer_normalization_1', 'gamma']
    #     beta1  = parameters['layer_normalization_1', 'beta']

    #     layer = 'multi_head_attention_1'
    #     W_q = parameters[layer,'W_q']
    #     W_k = parameters[layer,'W_k']
    #     W_v = parameters[layer,'W_v']
    #     W_o = parameters[layer,'W_o']

    #     try:
    #         b_q = parameters[layer,'b_q']
    #         b_k = parameters[layer,'b_k']
    #         b_v = parameters[layer,'b_v']
    #         b_o = parameters[layer,'b_o']
    #     except: # no bias values found
    #             b_q = 0
    #             b_k = 0
    #             b_v = 0
    #             b_o = 0
        
    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(hyper_params, m)  
    #     transformer.add_input_var("input_embed", dims=(seq_len, transformer.input_dim), bounds=(-3,3))
    #     transformer.add_layer_norm( "input_embed", "layer_norm", gamma1, beta1)
    #     transformer.add_attention( "layer_norm","attention_output", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
         
        
    #     # add constraints to trained TNN input
    #     m.tnn_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.input_embed.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.time_history):
    #         m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].first()]== m.x1[index])
    #         m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].last()] == m.x2[index]) 
            
        
    #     # # Convert to gurobipy
    #     gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
        
    #     ## Optimizes
    #     # gurobi_model.setParam('DualReductions',0)
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
                    
    #     if gurobi_model.status == GRB.INFEASIBLE:
    #             gurobi_model.computeIIS()
    #             gurobi_model.write("pytorch_model.ilp")
                
        
    #     # model output
    #     LN_output = np.array(optimal_parameters["layer_norm"])
    #     Q_form = np.array(optimal_parameters["Block_attention_output.Q"])
    #     K_form = np.array(optimal_parameters["Block_attention_output.K"])
    #     V_form = np.array(optimal_parameters["Block_attention_output.V"])
    #     attn_score_form = np.array(optimal_parameters["Block_attention_output.compatibility"])
    #     attn_weight_form = np.array(optimal_parameters["Block_attention_output.attention_weight"])
        
    #     # Check Solve calculations
    #     input = np.array(layer_outputs_dict['input_layer_1']).squeeze(0)
    #     transformer_input = np.array(layer_outputs_dict["layer_normalization_1"])[0]
    #     Q = np.dot( transformer_input, np.transpose(np.array(W_q),(1,0,2))) 
    #     K = np.dot( transformer_input, np.transpose(np.array(W_k),(1,0,2))) 
    #     V = np.dot( transformer_input, np.transpose(np.array(W_v),(1,0,2))) 

    #     Q = np.transpose(Q,(1,0,2)) + np.repeat(np.expand_dims(np.array(b_q),axis=1), transformer.N ,axis=1)
    #     K = np.transpose(K,(1,0,2)) + np.repeat(np.expand_dims(np.array(b_k),axis=1), transformer.N ,axis=1)
    #     V = np.transpose(V,(1,0,2)) + np.repeat(np.expand_dims(np.array(b_v),axis=1), transformer.N ,axis=1)
        
    #     #################### Calculate other intermediary vars
    #     print(Q.shape)
    #     q = Q
    #     k = K
    #     v = V
    #     d_k = Q.shape[-1]
    #     q_scaled = q / np.sqrt(d_k)

    #     # Attention scores: (batch_size, num_heads, seq_len_q, seq_len_k)
    #     attn_scores = np.matmul(q_scaled, np.transpose(k, (0, 2, 1)))

    #     # Apply softmax to get attention weights
    #     # Softmax along the last axis (seq_len_k) for each query position
    #     attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
    #     attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)

    #     # Attention output: weighted sum of the values (batch_size, num_heads, seq_len_q, depth_per_head)
    #     attn_output = np.matmul(attn_weights, v)

    #     # Combine heads (projected output): (batch_size, seq_len_q, num_heads, depth_per_head)
    #     # Final output projection (optional squeezing if needed)
    #     computed_attn_output = np.matmul(attn_output, W_o) + b_o
        
    #     ####################
    #     # Compare results:
        
    #     Q = Q.flatten()
    #     K = K.flatten()
    #     V = V.flatten()
        
    #     self.assertIsNone(np.testing.assert_array_almost_equal(np.array(layer_outputs_dict["layer_normalization_1"]).flatten(), LN_output, decimal =3))
    #     print("- MHA input formulation == MHA input model")
        
    #     self.assertIsNone(np.testing.assert_array_equal(V.shape, V_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( V_form,V, decimal =3))
    #     print("- Value formulation == Value model") 
        
    #     self.assertIsNone(np.testing.assert_array_equal(K.shape, K_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( K_form,K, decimal =3))
    #     print("- Key formulation == Key model")
          
    #     self.assertIsNone(np.testing.assert_array_equal(Q.shape, Q_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( Q_form,Q, decimal =3))
    #     print("- Query formulation == Query model")
        
    #     expected = np.array(layer_outputs_dict["multi_head_attention_1"]).flatten()
    #     self.assertIsNone(np.testing.assert_array_almost_equal(computed_attn_output.flatten(), expected, decimal=3))
    #     self.assertIsNone(np.testing.assert_array_equal(attn_scores.flatten().shape, attn_score_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( attn_score_form, attn_scores.flatten(), decimal =3))
    #     print("- Attn Score formulation == Attn Score model")
        
    #     self.assertIsNone(np.testing.assert_array_equal(attn_weights.flatten().shape, attn_weight_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( attn_weight_form, attn_weights.flatten(), decimal =3))
    #     print("- Attn Weights formulation == Attn weights model")  

    #     ## Check output 
    #     actual = np.array(optimal_parameters["attention_output"]) 
    #     self.assertIsNone(np.testing.assert_array_equal(actual.shape, expected.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(actual, expected, decimal=3)) # compare value with transformer output
    #     print("- MHA formulation == MHA model")
        
    # def test_ADDNORM1(self):
    #     print("======= ADD & NORM 1 =======")
        
    #     # Define Test Case Params
    #     m = model.clone()
    #     seq_len = tt
    #     layer = 'multi_head_attention_1'

    #     gamma1 = parameters['layer_normalization_1', 'gamma']
    #     beta1  = parameters['layer_normalization_1', 'beta']
        
    #     gamma2 = parameters['layer_normalization_2', 'gamma']
    #     beta2  = parameters['layer_normalization_2', 'beta']

    #     layer = 'multi_head_attention_1'
    #     W_q = parameters[layer,'W_q']
    #     W_k = parameters[layer,'W_k']
    #     W_v = parameters[layer,'W_v']
    #     W_o = parameters[layer,'W_o']

    #     try:
    #         b_q = parameters[layer,'b_q']
    #         b_k = parameters[layer,'b_k']
    #         b_v = parameters[layer,'b_v']
    #         b_o = parameters[layer,'b_o']
    #     except: # no bias values found
    #             b_q = 0
    #             b_k = 0
    #             b_v = 0
    #             b_o = 0
        
    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(hyper_params, m)  
    #     transformer.add_input_var("input_embed", dims=(seq_len, transformer.input_dim), bounds=(-3,3))
    #     transformer.add_layer_norm( "input_embed", "layer_norm", gamma1, beta1)
    #     transformer.add_attention( "layer_norm","attention_output", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
    #     transformer.add_residual_connection("input_embed", "attention_output", "residual_1")
    #     transformer.add_layer_norm( "residual_1", "layer_norm_2", gamma2, beta2)
         
        
    #     # add constraints to trained TNN input
    #     m.tnn_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.input_embed.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.time_history):
    #         m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].first()]== m.x1[index])
    #         m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].last()] == m.x2[index]) 
            
        
    #     # # Convert to gurobipy
    #     gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
        
    #     ## Optimizes
    #     # gurobi_model.setParam('DualReductions',0)
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
                    
    #     if gurobi_model.status == GRB.INFEASIBLE:
    #             gurobi_model.computeIIS()
    #             gurobi_model.write("pytorch_model.ilp")
                
        
        

    #     ## Check output 
    #     actual = np.array(optimal_parameters["layer_norm_2"]) 
    #     expected = np.array(layer_outputs_dict["layer_normalization_2"]).flatten()
    #     self.assertIsNone(np.testing.assert_array_equal(actual.shape, expected.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(actual, expected, decimal=3)) # compare value with transformer output
    #     print("- Add & Norm 1 formulation == Add & Norm 1 model")
                        
                        
    # def test_FFN1(self):
    #     print("======= FFN 1 =======")
        
    #     # Define Test Case Params
    #     m = model.clone()
    #     seq_len = tt
    #     layer = 'multi_head_attention_1'

    #     gamma1 = parameters['layer_normalization_1', 'gamma']
    #     beta1  = parameters['layer_normalization_1', 'beta']
        
    #     gamma2 = parameters['layer_normalization_2', 'gamma']
    #     beta2  = parameters['layer_normalization_2', 'beta']

    #     layer = 'multi_head_attention_1'
    #     W_q = parameters[layer,'W_q']
    #     W_k = parameters[layer,'W_k']
    #     W_v = parameters[layer,'W_v']
    #     W_o = parameters[layer,'W_o']

    #     try:
    #         b_q = parameters[layer,'b_q']
    #         b_k = parameters[layer,'b_k']
    #         b_v = parameters[layer,'b_v']
    #         b_o = parameters[layer,'b_o']
    #     except: # no bias values found
    #             b_q = 0
    #             b_k = 0
    #             b_v = 0
    #             b_o = 0
        
    #     # for i,v in parameters.items():
    #     #     print(f"{i}: {v}")
        
    #     # Define tranformer and execute 
    #     transformer = TNN.Transformer(hyper_params, m)  
    #     transformer.add_input_var("input_embed", dims=(seq_len, transformer.input_dim), bounds=(-3,3))
    #     transformer.add_layer_norm( "input_embed", "layer_norm", gamma1, beta1)
    #     transformer.add_attention( "layer_norm","attention_output", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
    #     transformer.add_residual_connection("input_embed", "attention_output", "residual_1")
    #     transformer.add_layer_norm( "residual_1", "layer_norm_2", gamma2, beta2)
    #     nn, input_nn, output_nn = transformer.get_ffn("layer_norm_2", "ffn_1", "ffn_1", (seq_len,2), parameters)
         
        
    #     # add constraints to trained TNN input
    #     m.tnn_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.input_embed.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.time_history):
    #         m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].first()]== m.x1[index])
    #         m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].last()] == m.x2[index]) 
            
        
    #     # # Convert to gurobipy
    #     gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
        
    #     ## Add FNN1 to gurobi model
    #     input_1, output_1 = get_inputs_gurobipy_FFN(input_nn, output_nn, map_var)
    #     pred_constr1 = add_predictor_constr(gurobi_model, nn, input_1, output_1)
        
    #     gurobi_model.update()
        
    #     ## Optimizes
    #     # gurobi_model.setParam('DualReductions',0)
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
                    
    #     if gurobi_model.status == GRB.INFEASIBLE:
    #             gurobi_model.computeIIS()
    #             gurobi_model.write("pytorch_model.ilp")
                
        
        

    #     ## Check output 
    #     actual = np.array(optimal_parameters["ffn_1"]) 
    #     expected = np.array(layer_outputs_dict["dense_2"]).flatten()
    #     self.assertIsNone(np.testing.assert_array_equal(actual.shape, expected.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(actual, expected, decimal=3)) # compare value with transformer output
    #     print("- FFN1 formulation == FFN1 model")
        
    def test_layers(self):
        print("======= FFN2 =======")
        
        # Define Test Case Params
        m = model.clone()
        seq_len = tt
        layer = 'multi_head_attention_1'
        W_q = parameters[layer,'W_q']
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']

        try:
            b_q = parameters[layer,'b_q']
            b_k = parameters[layer,'b_k']
            b_v = parameters[layer,'b_v']
            b_o = parameters[layer,'b_o']
        except: # no bias values found
                b_q = 0
                b_k = 0
                b_v = 0
                b_o = 0

        gamma1 = parameters['layer_normalization_1', 'gamma']
        beta1  = parameters['layer_normalization_1', 'beta']
        
        gamma2 = parameters['layer_normalization_2', 'gamma']
        beta2  = parameters['layer_normalization_2', 'beta']
        
        # for i,v in parameters.items():
        #     print(f"{i}: {v}")
            
        # ffn_1_params = parameters['ffn_1']
        # parameters['ffn_2']  = {'input_shape':ffn_1_params['input_shape']}#, "input": ffn_1_params['input']}
        # parameters['ffn_2'] |= {'dense_14':  ffn_1_params['dense_14']}
        # parameters['ffn_2'] |= {'dense_15':ffn_1_params['dense_15']}
        
        # parameters['ffn_1']  = {'input_shape': ffn_1_params['input_shape']}
        # parameters['ffn_1'] |= {'dense_12': ffn_1_params['dense_12']}
        # parameters['ffn_1'] |= {'dense_13': ffn_1_params['dense_13']}
        
        
        # Define tranformer and execute 
        transformer = TNN.Transformer(hyper_params, m, set_bound_cut=bound_cut)  
        transformer.add_input_var("input_embed", dims=(seq_len, transformer.input_dim), bounds=(-3,3))
        transformer.add_layer_norm( "input_embed", "layer_norm", gamma1, beta1)
        transformer.add_attention( "layer_norm","attention_output", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        transformer.add_residual_connection("input_embed", "attention_output", "residual_1")
        transformer.add_layer_norm( "residual_1", "layer_norm_2", gamma2, beta2)
        nn, input_nn, output_nn = transformer.get_ffn("layer_norm_2", "ffn_1", "ffn_1", (seq_len,2), parameters)
        transformer.add_residual_connection("residual_1", "ffn_1", "residual_2")  
        nn2, input_nn2, output_nn2 = transformer.get_ffn( "residual_2", "ffn_2", "ffn_2", (seq_len, 2), parameters)
        
        
        # add constraints to trained TNN input
        m.tnn_constraints = pyo.ConstraintList()
        indices = []
        for set in str(transformer.M.input_embed.index_set()).split("*"):
            indices.append( getattr(m, set) )
        for tnn_index, index in zip(indices[0], m.time_history):
            m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].first()]== m.x1[index])
            m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].last()] == m.x2[index]) 
            
        # add constraints to trained TNN output
        indices = []
        for set in str(output_nn2.index_set()).split("*"): 
            indices.append( getattr(m, set) )
        out_index = 0
        for t_index, t in enumerate(m.time):
            index = t_index + 1 # 1 indexing
            
            if t > m.time_history.last(): # since overlap is 1
                out_index += 2
                print(t, indices[0].at(out_index), indices[1].first(), indices[1].last())
                m.tnn_constraints.add(expr= output_nn2[indices[0].at(out_index), indices[1].first()] == m.x1[t])
                m.tnn_constraints.add(expr= output_nn2[indices[0].at(out_index), indices[1].last()]  == m.x2[t])
        

        # # Convert to gurobipy
        gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)
        
        # for i,v in map_var.items():
        #     print(i, v)

        ## Add FNN1 to gurobi model
        input_1, output_1 = get_inputs_gurobipy_FFN(input_nn, output_nn, map_var)
        pred_constr1 = add_predictor_constr(gurobi_model, nn, input_1, output_1)
        
        inputs_2, outputs_2 = get_inputs_gurobipy_FFN(input_nn2, output_nn2, map_var)
        pred_constr2 = add_predictor_constr(gurobi_model, nn2, inputs_2, outputs_2)
        gurobi_model.update()
        #pred_constr.print_stats()
        
        ## Optimizes
        # gurobi_model.setParam('DualReductions',0)
        #gurobi_model.setParam('MIPFocus',1)
        PATH = r".\Experiments"
        gurobi_model.setParam('LogFile', PATH+'\\toy_no_enhancement.log')
        gurobi_model.setParam('TimeLimit', 21600)
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
                    
        if gurobi_model.status == GRB.INFEASIBLE:
                gurobi_model.computeIIS()
                gurobi_model.write("pytorch_model.ilp")
                
        x1 = np.array(optimal_parameters['x1'])
        x2 = np.array(optimal_parameters['x2'])
        loc1 = np.array([v for k,v in model.loc1.items()])
        loc2 = np.array([v for k,v in model.loc2.items()])
        FFN_out = np.array(layer_outputs_dict["dense_4"])[0].transpose(1,0)
              
        plt.figure(1, figsize=(8, 4))
        plt.plot(time[2], FFN_out[1][1],'s', color='tab:cyan',label= "y TNN pred.")
        plt.plot(time[2], FFN_out[0][1],'s', color='tab:gray',label= "x TNN pred.")
        
        plt.plot(time, loc2, 'o', color='tab:blue', label = 'y targets')
        plt.plot(time, loc1, 'o', color='m', label = 'x targets')
        
        opt_x1 = [0.0, 0.00555569, 0.01111138]
        opt_x2 = [0.0, 0.05100613, 0.09444281]
        plt.plot(time, opt_x1, color='tab:green', label= "y expected opt. trajectory")
        plt.plot(time, opt_x2, color='tab:orange', label= "x expected opt. trajectory")
        
        plt.plot(time, x2, '--x', color='r', label = 'y opt. trajectory')
        plt.plot(time, x1, '--x', color='b', label = 'x opt. trajectory')
        
        plt.title('Optimal Trajectory Toy with fixed inputs')
        plt.legend()
        plt.show()

        plt.figure(2, figsize=(6, 4))
        plt.plot(loc1, loc2, 'o', label = 'target trajectory')
        plt.plot(opt_x1, opt_x2, label= "expected opt. trajectory")
        plt.plot(x1, x2, '--x', label = 'opt. trajectory')
        plt.title('Trajectory of cannon ball')
        plt.legend()
        plt.show()
        
        ## Check output
        actual = np.array(list(optimal_parameters['input_embed']))
        expected = input[0,0:seq_len,:].flatten()
        
        self.assertIsNone(np.testing.assert_array_equal(actual.shape, expected.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(actual, expected, decimal=7)) # compare value with transformer output
        print("- input formulation == input model")   
        
        #Check outputs FFN 2
        ffn_2_output = np.array([optimal_parameters["ffn_2"]]).squeeze(0)
        FFN_out = np.array(layer_outputs_dict["dense_4"])[0].flatten()
        
        print("Output: ", ffn_2_output)
        print("expected:", FFN_out)
        print( np.array([optimal_parameters["ffn_2"]]))
        print( np.array([optimal_parameters["x1"]])[0][-pred_len - 1:])
        print( np.array([optimal_parameters["x2"]])[0][-pred_len - 1:])
        
        self.assertIsNone(np.testing.assert_array_equal(ffn_2_output.shape,  FFN_out.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(ffn_2_output,  FFN_out, decimal=3)) # compare value with transformer output
        print("- FFN2 formulation == FFN2 trained model")   
        
        
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
    hyper_params = '.\\data\\toy_track_k_enc_config_2.json' 
    
    # define constants
    T_end = 0.5#0.0105
    steps = 19 ##CHANGE THIS ##
    time = np.linspace(0, T_end, num=steps)
    
    tt = 2 # sequence size
    time_history = time[0:tt]
    pred_len = 1
    
    time = time[:tt+pred_len]
    steps = len(time)
    print(steps)

    g = 9.81
    v_l1 = 0.2
    v_l2 = 1.5
    dt = time[-1] - time[0]
    
    # define sets
    model.time = pyo.Set(initialize=time)
    model.time_history = pyo.Set(initialize=time_history)
    
    # define parameters
    def target_location_rule(M, t):
        return v_l1 * t
    model.loc1 = pyo.Param(model.time, rule=target_location_rule) 

    def target_location2_rule(M, t):
        np.random.seed(int(v_l2*t*100))
        print(np.random.uniform(-1,1)/30)
        return (v_l2*t) - (0.5 * g * (t**2)) + ( np.random.uniform(-1,1)/30 )
    model.loc2 = pyo.Param(model.time, rule=target_location2_rule) 

    
    # define variables
    bounds_target = (-3,3)
    model.x1 = pyo.Var(model.time, bounds = bounds_target ) # distance path
    model.x2 = pyo.Var(model.time, bounds = bounds_target) # height path

    # define initial conditions
    model.x1_constr = pyo.Constraint(expr= model.x1[0] == 0) 
    model.x2_constr = pyo.Constraint(expr= model.x2[0] == 0) 
    
            
    # load trained transformer
    model_path = ".\\trained_transformer\\TNN_traj_enc_2.keras" # dmodel 4, num heads 1, n ence 1, n dec 1, head dim 4, pred_len 2+1 
    layer_names, parameters ,_ = extract_from_pretrained.get_learned_parameters(model_path)
    
    # get intermediate results dictionary for optimal input values
    # input_x1 =   v_l1 * time  
    # input_x2 =  (v_l2*time) - (0.5 * g * (time*time))
    input_x1 = [0.0, 0.00555569, 0.01111138] # from solution track_toy.py
    input_x2 = [0.0, 0.05100613, 0.09444281]
    input = np.array([[ [x1,x2] for x1,x2 in zip(input_x1, input_x2)]], dtype=np.float32)
    layer_outputs_dict = extract_from_pretrained.get_intermediate_values(model_path, input[:, 0:tt, :])

    # for i,v in parameters.items():
    #     print(i, v)
    # Fix ffn params: add layer not in architecture which is between the two ffns
    ffn_1_params = parameters['ffn_1']
    parameters['ffn_2']  = {'input_shape':ffn_1_params['input_shape']}#, "input": ffn_1_params['input']}
    parameters['ffn_2'] |= {'dense_3':  ffn_1_params['dense_3']}
    parameters['ffn_2'] |= {'dense_4':ffn_1_params['dense_4']}
    
    parameters['ffn_1']  = {'input_shape': ffn_1_params['input_shape']}
    parameters['ffn_1'] |= {'dense_1': ffn_1_params['dense_1']}
    parameters['ffn_1'] |= {'dense_2': ffn_1_params['dense_2']}



    # ##------ Fix model solution ------##
    FFN_out = np.array(layer_outputs_dict["dense_4"])[0].transpose(1,0)
    #print("FFN out",FFN_out.shape, FFN_out, FFN_out[0], FFN_out[1])
    
    model.fixed_loc_constraints = pyo.ConstraintList()
    # for i,t in enumerate(model.time):
    #     if t <= model.time_history.last():
    #         model.fixed_loc_constraints.add(expr= input_x1[i] == model.x1[t])
    #         model.fixed_loc_constraints.add(expr= input_x2[i]  == model.x2[t])
    #     else:
    #         print(i, FFN_out[0][i-1], FFN_out[1][i-1])
    #         model.fixed_loc_constraints.add(expr= FFN_out[0][i-1] == model.x1[t])
    #         model.fixed_loc_constraints.add(expr= FFN_out[1][i-1]  == model.x2[t])

    # Set objective
    model.obj = pyo.Objective(
        expr= sum((model.x1[t] - model.loc1[t])**2 + (model.x2[t] - model.loc2[t])**2 for t in model.time), sense=pyo.minimize
    )  # -1: maximize, +1: minimize (default)


    # select which bounds and cuts to activate:
    ACTI_LIST_FULL = [
            "LN_var", "LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum",
             "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight",
            "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC", "MHA_attn_score", "MHA_output", 
            "RES_var", "MHA_softmax_env", "AVG_POOL_var", "embed_var"]

    bound_cut = {}
    for key in ACTI_LIST_FULL:
        bound_cut[key] = False
        
    # unit test
    unittest.main() 

