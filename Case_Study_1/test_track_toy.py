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
import transformer_b_flag_cuts as TNN
from training_scripts.Tmodel import TransformerModel
import helpers.extract_from_pretrained as extract_from_pretrained

"""
Test each module of transformer for optimal control toy tnn 1
"""
# ------- Transformer Test Class ------------------------------------------------------------------------------------
class TestTransformer(unittest.TestCase):    
    ## commented out to debug last test
    
    
    # def test_instantiation_input(self): #, model, pyomo_input_name ,input):
    #     m = model.clone()
        
    #     # create optimization transformer
    #     transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", m) 
        
    #     # define sets for inputs
    #     enc_dim_1 = transformer.N 
    #     dec_dim_1 = 3
    #     transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
    #     transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
    #     transformer.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) # - 2
    #     transformer.M.model_dims = pyo.Set(initialize= list(range(transformer.d_model)))
    #     transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
    #     enc_flag = False
    #     dec_flag = False
        
    #     # Add TNN input vars
    #     transformer.M.enc_input= pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
    #     transformer.M.dec_input = pyo.Var(transformer.M.dec_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        
    #     # add constraints to trained TNN input
    #     m.tnn_input_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.enc_input.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.time_history):
    #         m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].first()]== m.x1[index])
    #         m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].last()] == m.x2[index]) 
            
    #     indices = []
    #     for set in str(transformer.M.dec_input.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
            
    #     dec_index = 0
    #     for t_index, t in enumerate(m.time):
    #         index = t_index + 1 # 1 indexing
            
    #         if t >= m.time_history.last():
    #             dec_index += 1
    #             print(dec_index, t )
    #             m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(dec_index), indices[1].first()] == m.x1[t])
    #             m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(dec_index), indices[1].last()]  == m.x2[t])
                
             
    #     # Set objective
    #     m.obj = pyo.Objective(
    #         expr= sum((m.x1[t] - m.loc1[t])**2 + (m.x2[t] - m.loc2[t])**2 for t in m.time), sense=1
    #     )  # -1: maximize, +1: minimize (default)
    
    
    #     # Convert to gurobipy
    #     gurobi_model, _, _ = convert_pyomo.to_gurobi(m)
        
        
    #     # Solve
    #     gurobi_model.optimize()

    #     if gurobi_model.status == GRB.INFEASIBLE:
    #             gurobi_model.computeIIS()
    #             gurobi_model.write("pytorch_model.ilp")
        
    #     # Get input value (before solve)
    #     parameters = {}
    #     for v in gurobi_model.getVars():
    #         #print(f'var name: {v.varName}, var type {type(v)}')
    #         # print(v.LB, v.UB)
    #         if "[" in v.varName:
    #             name = v.varname.split("[")[0]
    #             if name in parameters.keys():
    #                 parameters[name] += [v.x]
    #             else:
    #                 parameters[name] = [v.x]
    #         else:    
    #             parameters[v.varName] = v.x

    #     # model input
    #     model_enc_input = np.array(parameters[enc_input_name])
    #     model_dec_input = np.array(parameters[dec_input_name])
        
    #     expected_enc_input = input[0, 0:tt, :].flatten() 
    #     expected_dec_input = input[0, -3:, :].flatten() 

    #     # Assertions
    #     self.assertIsNone(np.testing.assert_array_equal(model_enc_input.shape, expected_enc_input.shape)) # pyomo input data and transformer input data must be the same shape
    #     self.assertIsNone(np.testing.assert_array_almost_equal(model_enc_input, expected_enc_input, decimal = 7))             # both inputs must be equal
        
    #     self.assertIsNone(np.testing.assert_array_equal(model_dec_input.shape, expected_dec_input.shape)) # pyomo input data and transformer input data must be the same shape
    #     self.assertIsNone(np.testing.assert_array_almost_equal(model_dec_input, expected_dec_input, decimal = 7))             # both inputs must be equal
        

    # def test_embed_input(self):
    #     m = model.clone()
        
    #     # create optimization transformer
    #     transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", m) 
        
    #     # define sets for inputs
    #     enc_dim_1 = transformer.N 
    #     dec_dim_1 = 3
    #     transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
    #     transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
    #     transformer.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) # - 2
    #     transformer.M.model_dims = pyo.Set(initialize= list(range(transformer.d_model)))
    #     transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
    #     enc_flag = False
    #     dec_flag = False
        
    #     # Add TNN input vars
    #     transformer.M.enc_input= pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
    #     transformer.M.dec_input = pyo.Var(transformer.M.dec_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        
    #     # add constraints to trained TNN input
    #     m.tnn_input_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.enc_input.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.time_history):
    #         m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].first()]== m.x1[index])
    #         m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].last()] == m.x2[index]) 
            
    #     indices = []
    #     for set in str(transformer.M.dec_input.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
            
    #     dec_index = 0
    #     for t_index, t in enumerate(m.time):
    #         index = t_index + 1 # 1 indexing
            
    #         if t >= m.time_history.last():
    #             dec_index += 1
    #             print(dec_index, t )
    #             m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(dec_index), indices[1].first()] == m.x1[t])
    #             m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(dec_index), indices[1].last()]  == m.x2[t])
                
    #     # Add Embedding (linear) layer
    #     embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
    #     layer = "linear_1"
    #     W_linear = parameters[layer,'W']
    #     try:
    #         b_linear = parameters[layer,'b']
    #     except:
    #         b_linear = None
    #     transformer.embed_input( enc_input_name, "enc_linear_1", embed_dim, W_linear, b_linear)
    #     transformer.embed_input( dec_input_name, "dec_linear_1", embed_dim, W_linear, b_linear)
        
    #     # Set objective
    #     m.obj = pyo.Objective(
    #         expr= sum((m.x1[t] - m.loc1[t])**2 + (m.x2[t] - m.loc2[t])**2 for t in m.time), sense=1
    #     )  # -1: maximize, +1: minimize (default)
    
    
    #     # Convert to gurobipy
    #     gurobi_model, _, _ = convert_pyomo.to_gurobi(m)

    #     # Check new model attributes added
    #     self.assertIn("enc_linear_1", dir(transformer.M))                        # check var created
    #     self.assertIn("dec_linear_1", dir(transformer.M))                        # check var created
    #     self.assertIsInstance(transformer.M.enc_linear_1, pyo.Var)               # check data type
    #     self.assertIsInstance(transformer.M.dec_linear_1, pyo.Var)               # check data type
    #     self.assertTrue(hasattr(transformer.M, 'embed_constraints'))            # check constraints created
        
    #     # Convert to gurobipy
    #     gurobi_model, _ , _ = convert_pyomo.to_gurobi(m)
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
            
    #     # outputs
        
    #     embed_enc = np.array(optimal_parameters["enc_linear_1"])
    #     embed_dec = np.array(optimal_parameters["dec_linear_1"])
        
    #     expected_enc = np.array(list(layer_outputs_dict['linear1'])[0]).flatten()
    #     expected_dec = np.array(list(layer_outputs_dict['linear1'])[1]).flatten()
        
    #     # Assertions
    #     self.assertIsNone(np.testing.assert_array_equal(embed_enc.shape, expected_enc.shape)) # same shape
    #     self.assertIsNone(np.testing.assert_array_almost_equal(embed_enc, expected_enc, decimal=2))  # almost same values
    
    #     self.assertIsNone(np.testing.assert_array_equal(embed_dec.shape, expected_dec.shape)) # same shape
    #     self.assertIsNone(np.testing.assert_array_almost_equal(embed_dec, expected_dec, decimal=2))  # almost same values
    
    # def test_encoder_self_attention(self):
    #     print("======= MULTIHEAD ATTENTION =======")
    #     m = model.clone()
        
    #     # create optimization transformer
    #     transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", m) 
        
    #     # define sets for inputs
    #     enc_dim_1 = transformer.N 
    #     dec_dim_1 = 3
    #     transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
    #     transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
    #     transformer.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) # - 2
    #     transformer.M.model_dims = pyo.Set(initialize= list(range(transformer.d_model)))
    #     transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
    #     enc_flag = False
    #     dec_flag = False
        
    #     # Add TNN input vars
    #     transformer.M.enc_input= pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
    #     transformer.M.dec_input = pyo.Var(transformer.M.dec_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        
    #     # add constraints to trained TNN input
    #     m.tnn_input_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.enc_input.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.time_history):
    #         m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].first()]== m.x1[index])
    #         m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].last()] == m.x2[index]) 
            
    #     indices = []
    #     for set in str(transformer.M.dec_input.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
            
    #     dec_index = 0
    #     for t_index, t in enumerate(m.time):
    #         index = t_index + 1 # 1 indexing
            
    #         if t >= m.time_history.last():
    #             dec_index += 1
    #             print(dec_index, t )
    #             m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(dec_index), indices[1].first()] == m.x1[t])
    #             m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(dec_index), indices[1].last()]  == m.x2[t])
                
    #     # Add Embedding (linear) layer
    #     embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
    #     layer = "linear_1"
    #     W_linear = parameters[layer,'W']
    #     try:
    #         b_linear = parameters[layer,'b']
    #     except:
    #         b_linear = None
    #     transformer.embed_input( enc_input_name, "enc_linear_1", embed_dim, W_linear, b_linear)
    #     transformer.embed_input( dec_input_name, "dec_linear_1", embed_dim, W_linear, b_linear)
        
        
                
    #     # Add encoder self attention layer
    #     input_name = "enc_linear_1"
    #     layer = "enc__self_attention_1"
        
    #     W_q = parameters[layer,'W_q']
    #     W_k = parameters[layer,'W_k']
    #     W_v = parameters[layer,'W_v']
    #     W_o = parameters[layer,'W_o']

        
    #     b_q = parameters[layer,'b_q']
    #     b_k = parameters[layer,'b_k']
    #     b_v = parameters[layer,'b_v']
    #     b_o = parameters[layer,'b_o']
        
    #     if not b_q is None:  
              
    #         transformer.add_attention( input_name, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
    #     else:
    #         print('no bias', b_q, b_k, b_v, b_o) 
    #         transformer.add_attention( input_name, layer, W_q, W_k, W_k, W_o)
            
    #     # Set objective
    #     m.obj = pyo.Objective(
    #         expr= sum((m.x1[t] - m.loc1[t])**2 + (m.x2[t] - m.loc2[t])**2 for t in m.time), sense=1
    #     )  # -1: maximize, +1: minimize (default)
    
    
    #     # Check new model attributes added
    #     self.assertIn(layer, dir(transformer.M))                        # check var created
    #     self.assertIsInstance(transformer.M.enc__self_attention_1, pyo.Var)               # check data type
    #     self.assertTrue(hasattr(transformer.M, 'Block_enc__self_attention_1'))            # check constraints created
        
     
    #     # Convert to gurobipy
    #     gurobi_model, _ , _ = convert_pyomo.to_gurobi(m)
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
            
    #     # outputs
    #     output_name = layer
    #     self_attn_enc = torch.tensor(optimal_parameters[output_name])
    #     embed_enc = torch.tensor(optimal_parameters[input_name])
    #     Q_form = torch.tensor(optimal_parameters[f"Block_{output_name}.Q"])
    #     K_form = torch.tensor(optimal_parameters[f"Block_{output_name}.K"])
    #     V_form = torch.tensor(optimal_parameters[f"Block_{output_name}.V"])
    #     attn_score_form = torch.tensor(optimal_parameters[f"Block_{output_name}.compatibility"])
    #     attn_weight_form = torch.tensor(optimal_parameters[f"Block_{output_name}.attention_weight"])
        
    #     O_form = torch.tensor(optimal_parameters[f"Block_{output_name}.W_o"])

    #     expected_out = torch.tensor(list(layer_outputs_dict['transformer.encoder.layers.0.self_attn'])[0][0]).detach()
    #     # print(torch.tensor(list(layer_outputs_dict['transformer.encoder.layers.0.self_attn'])[0][0]))
    #     # print(self_attn_enc)
        
        
    #     # Check Solve calculations
    #     expected_enc_input = torch.tensor(list(layer_outputs_dict["linear1"])[0]).unsqueeze(0) #[b,n,d]
    #     W_q = torch.tensor(W_q).unsqueeze(0) #[b, d, h, k]
    #     W_k = torch.tensor(W_k).unsqueeze(0) 
    #     W_v = torch.tensor(W_v).unsqueeze(0) 
        
    #     W_q = W_q.permute(0,2,1,3) #[b, h, d, k]
    #     W_k = W_k.permute(0,2,1,3)
    #     W_v = W_v.permute(0,2,1,3) 
        
        
    #     Q = torch.matmul( expected_enc_input, W_q).squeeze(-2) #[b,n,d] *[b,h,d,k]--> [b,h,n,k]
    #     K = torch.matmul( expected_enc_input, W_k).squeeze(-2) 
    #     V = torch.matmul( expected_enc_input, W_v).squeeze(-2)
        
    #     print("Q shape: [1,1,10,4]", Q.shape)
    
    #     Q += torch.tensor(b_q) # [b,h,n,k]
    #     K += torch.tensor(b_k)#.unsqueeze(1).repeat_interleave(transformer.N , dim=1)
    #     V += torch.tensor(b_v)#.unsqueeze(1).repeat_interleave(transformer.N , dim=1)
        
    #     ##########################
    #     # Calculate other intermediary vars
    #     q = Q #[h,n,k]
    #     k = K
    #     v = V
     
    #     d_k = Q.shape[-1]
    #     q_scaled = q / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
    #     attn_scores = torch.matmul(q_scaled, k.permute(0,1,3,2))
    #     print("attn score", attn_scores.shape)

    #     attn_weights = torch.exp(attn_scores) 
    #     attn_weights /= torch.sum(attn_weights, dim=-1, keepdims=True)
    #     print("attn w", attn_weights.shape)


    #     attn_output = torch.matmul(attn_weights, v)
    #     print("attn out", attn_output.shape)
    #     print("v       ", v.shape)

    #     W_o = torch.tensor(W_o).unsqueeze(0) #[b,h,k,d]
    #     print("W_o ", W_o.shape)
    #     computed_attn_output = torch.sum(torch.matmul(attn_output, W_o), dim=1) 
    #     print(computed_attn_output.shape)
    #     computed_attn_output += torch.tensor(b_o)
        
    #     Q = Q.flatten()
    #     K = K.flatten()
    #     V = V.flatten()
    #     expected_out = expected_out.flatten()
        
    #     self.assertIsNone(np.testing.assert_array_almost_equal(expected_enc_input.flatten(), embed_enc, decimal =5))
    #     print("- MHA input formulation == MHA input model")
        
    #     self.assertIsNone(np.testing.assert_array_equal(Q.shape, Q_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( Q, Q_form, decimal =5))
    #     print("- Query formulation == Query model")
        
    #     self.assertIsNone(np.testing.assert_array_equal(K.shape, K_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( K, K_form, decimal =5))
    #     print("- Key formulation == Key model")
        
    #     self.assertIsNone(np.testing.assert_array_equal(V.shape, V_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( V, V_form, decimal =5))
    #     print("- Value formulation == Value model")
        
    #     self.assertIsNone(np.testing.assert_array_equal(attn_scores.flatten().shape, attn_score_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( attn_score_form, attn_scores.flatten(), decimal =4))
    #     print("- Attn Score formulation == Attn Score model")
        
    #     self.assertIsNone(np.testing.assert_array_equal(attn_weights.flatten().shape, attn_weight_form.shape))
    #     self.assertIsNone(np.testing.assert_array_almost_equal( attn_weight_form, attn_weights.flatten(), decimal =4))
    #     print("- Attn Weights formulation == Attn weights model")  

    #     self.assertIsNone(np.testing.assert_array_almost_equal(computed_attn_output.flatten(), expected_out, decimal=4))
    #     print("- Calculated Attn == Attn Trained TNN")
        
    #     ## Check MHA output
    #     self.assertIsNone(np.testing.assert_array_equal(expected_out.shape, self_attn_enc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(expected_out, self_attn_enc , decimal=5)) # compare value with transformer output
    #     print("- MHA output formulation == MHA Trained TNN")
      
    # def test_encoder(self):
    #     print("======= MULTIHEAD ATTENTION =======")
    #     m = model.clone()
        
    #     # create optimization transformer
    #     transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", m) 
        
    #     # define sets for inputs
    #     enc_dim_1 = transformer.N 
    #     dec_dim_1 = 3
    #     transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
    #     transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
    #     transformer.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) # - 2
    #     transformer.M.model_dims = pyo.Set(initialize= list(range(transformer.d_model)))
    #     transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
    #     enc_flag = False
    #     dec_flag = False
        
    #     # Add TNN input vars
    #     transformer.M.enc_input= pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
    #     transformer.M.dec_input = pyo.Var(transformer.M.dec_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        
    #     # add constraints to trained TNN input
    #     m.tnn_input_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.enc_input.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.time_history):
    #         m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].first()]== m.x1[index])
    #         m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].last()] == m.x2[index]) 
            
    #     indices = []
    #     for set in str(transformer.M.dec_input.index_set()).split("*"):
    #         indices.append( getattr(m, set) )
            
    #     dec_index = 0
    #     for t_index, t in enumerate(m.time):
    #         index = t_index + 1 # 1 indexing
            
    #         if t >= m.time_history.last():
    #             dec_index += 1
    #             print(dec_index, t )
    #             m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(dec_index), indices[1].first()] == m.x1[t])
    #             m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(dec_index), indices[1].last()]  == m.x2[t])
                
    #     # Add Embedding (linear) layer
    #     embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
    #     layer = "linear_1"
    #     W_linear = parameters[layer,'W']
    #     try:
    #         b_linear = parameters[layer,'b']
    #     except:
    #         b_linear = None
    #     transformer.embed_input( enc_input_name, "enc_linear_1", embed_dim, W_linear, b_linear)
    #     transformer.embed_input( dec_input_name, "dec_linear_1", embed_dim, W_linear, b_linear)
        
        
                
    #     # Add encoder self attention layer
    #     input_name = "enc_linear_1"
    #     layer = "enc__self_attention_1"
        
    #     W_q = parameters[layer,'W_q']
    #     W_k = parameters[layer,'W_k']
    #     W_v = parameters[layer,'W_v']
    #     W_o = parameters[layer,'W_o']

        
    #     b_q = parameters[layer,'b_q']
    #     b_k = parameters[layer,'b_k']
    #     b_v = parameters[layer,'b_v']
    #     b_o = parameters[layer,'b_o']
        
    #     if not b_q is None:  
    #         transformer.add_attention( input_name, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
    #     else:
    #         print('no bias', b_q, b_k, b_v, b_o) 
    #         transformer.add_attention( input_name, layer, W_q, W_k, W_k, W_o)
        
    #     # add res+norm1
    #     enc_layer = 0
    #     gamma1 = parameters["enc__layer_normalization_1", 'gamma']
    #     beta1 = parameters["enc__layer_normalization_1", 'beta']
        
    #     transformer.add_residual_connection(input_name, "enc__self_attention_1", f"{layer}__{enc_layer}_residual_1")
    #     transformer.add_layer_norm(f"{layer}__{enc_layer}_residual_1", "enc_norm_1", gamma1, beta1)
        
    #     # add ffn1
    #     ffn_parameter_dict = {}
    #     input_shape = np.array(parameters["enc__ffn_1"]['input_shape'])
    #     print(input_shape)
    #     ffn_params = transformer.get_fnn( "enc_norm_1", "enc__ffn_1", "enc__ffn_1", (10,4), parameters)
    #     ffn_parameter_dict["enc__ffn_1"] = ffn_params # ffn_params: nn, input_nn, output_nn
        

    #     # add res+norm2
    #     gamma2 = parameters["enc__layer_normalization_2", 'gamma']
    #     beta2 = parameters["enc__layer_normalization_2", 'beta']
        
    #     transformer.add_residual_connection("enc_norm_1", "enc__ffn_1", f"{layer}__{enc_layer}_residual_2")
    #     transformer.add_layer_norm(f"{layer}__{enc_layer}_residual_2", "enc_norm_2", gamma2, beta2)
        
        
    #     #add enc norm (norm over various encoder layers)
    #     gamma3 = parameters["enc_layer_normalization_1", 'gamma']
    #     beta3 = parameters["enc_layer_normalization_1", 'beta']
    #     transformer.add_layer_norm("enc_norm_2", "enc_norm_3", gamma3, beta3)
        
        
    #     # Set objective
    #     m.obj = pyo.Objective(
    #         expr= sum((m.x1[t] - m.loc1[t])**2 + (m.x2[t] - m.loc2[t])**2 for t in m.time), sense=1
    #     )  # -1: maximize, +1: minimize (default)
    
    
    #     # Check new model attributes added
    #     self.assertIn(layer, dir(transformer.M))                        # check var created
    #     self.assertIsInstance(transformer.M.enc__self_attention_1, pyo.Var)               # check data type
    #     self.assertTrue(hasattr(transformer.M, 'Block_enc__self_attention_1'))            # check constraints created
        
     
    #     # Convert to gurobipy
    #     gurobi_model, map_var , _ = convert_pyomo.to_gurobi(m)
        
        
    #     # Add FNN1 to gurobi model
    #     for key, value in ffn_parameter_dict.items():
    #         nn, input_nn, output_nn = value
    #         input, output = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
    #         pred_constr = add_predictor_constr(gurobi_model, nn, input, output)
        
    #     gurobi_model.update() # update gurobi model with FFN constraints
        
    #     # Optimize
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
            
    #     # outputs
    #     output_name = layer
    #     self_attn_enc = np.array(optimal_parameters[output_name])
    #     self_attn_expected_out = np.array(list(layer_outputs_dict['transformer.encoder.layers.0.self_attn'])[0][0]).flatten()

    #     norm1_enc = np.array(optimal_parameters["enc_norm_1"])
    #     norm1_expected = np.array(list(layer_outputs_dict['transformer.encoder.layers.0.norm1']))[0].flatten()

    #     ffn1_enc = np.array(optimal_parameters["enc__ffn_1"])
    #     ffn1_expected = np.array(list(layer_outputs_dict['transformer.encoder.layers.0.linear2']))[0].flatten()
        
    #     norm2_enc = np.array(optimal_parameters["enc_norm_2"])
    #     norm2_expected = np.array(list(layer_outputs_dict['transformer.encoder.layers.0.norm2']))[0].flatten()

    #     norm3_enc = np.array(optimal_parameters["enc_norm_3"])
    #     norm3_expected = np.array(list(layer_outputs_dict['transformer.encoder.norm']))[0].flatten()

    #     ## Check MHA output
    #     self.assertIsNone(np.testing.assert_array_equal(self_attn_expected_out.shape, self_attn_enc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(self_attn_expected_out, self_attn_enc , decimal=5)) # compare value with transformer output
    #     print("- Enc MHA output formulation == Enc MHA Trained TNN")  
        
    #     self.assertIsNone(np.testing.assert_array_equal(norm1_expected.shape, norm1_enc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(norm1_expected, norm1_enc , decimal=4)) # compare value with transformer output
    #     print("- Enc Norm1 formulation == Enc Norm1 Trained TNN")  
        
    #     self.assertIsNone(np.testing.assert_array_equal(ffn1_expected.shape, ffn1_enc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(ffn1_expected, ffn1_enc , decimal=4)) # compare value with transformer output
    #     print("- Enc FFN1 formulation == Enc FNN1 Trained TNN") 
        
    #     self.assertIsNone(np.testing.assert_array_equal(norm2_expected.shape, norm2_enc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(norm2_expected, norm2_enc , decimal=4)) # compare value with transformer output
    #     print("- Enc Norm2 formulation == Enc Norm2 Trained TNN")  
        
    #     self.assertIsNone(np.testing.assert_array_equal(norm3_expected.shape, norm3_enc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(norm3_expected, norm3_enc , decimal=4)) # compare value with transformer output
    #     print("- Enc Output formulation == Enc Output Trained TNN")  
    
    def test_decoder(self):
        print("======= MULTIHEAD ATTENTION =======")
        m = model.clone()
        
        # create optimization transformer
        transformer = TNN.Transformer( ".\\data\\toy_config_pytorch.json", m) 
        
        # define sets for inputs
        enc_dim_1 = transformer.N 
        dec_dim_1 = 3
        transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
        transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
        transformer.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) # - 2
        transformer.M.model_dims = pyo.Set(initialize= list(range(transformer.d_model)))
        transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
        enc_flag = False
        dec_flag = False
        
        # Add TNN input vars
        transformer.M.enc_input= pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        transformer.M.dec_input = pyo.Var(transformer.M.dec_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        
        # add constraints to trained TNN input
        m.tnn_input_constraints = pyo.ConstraintList()
        indices = []
        for set in str(transformer.M.enc_input.index_set()).split("*"):
            indices.append( getattr(m, set) )
        for tnn_index, index in zip(indices[0], m.time_history):
            m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].first()]== m.x1[index])
            m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, indices[1].last()] == m.x2[index]) 
            
        indices = []
        for set in str(transformer.M.dec_input.index_set()).split("*"):
            indices.append( getattr(m, set) )
            
        dec_index = 0
        for t_index, t in enumerate(m.time):
            index = t_index + 1 # 1 indexing
            
            if t >= m.time_history.last():
                dec_index += 1
                m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(dec_index), indices[1].first()] == m.x1[t])
                m.tnn_input_constraints.add(expr= transformer.M.dec_input[indices[0].at(dec_index), indices[1].last()]  == m.x2[t])
                
        # Add Embedding (linear) layer
        embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
        layer = "linear_1"
        W_linear = parameters[layer,'W']
        try:
            b_linear = parameters[layer,'b']
        except:
            b_linear = None
        transformer.embed_input( enc_input_name, "enc_linear_1", embed_dim, W_linear, b_linear)
        transformer.embed_input( dec_input_name, "dec_linear_1", embed_dim, W_linear, b_linear)
        
        
                
        # Add encoder self attention layer
        input_name = "enc_linear_1"
        layer = "enc__self_attention_1"
        
        W_q = parameters[layer,'W_q']
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']

        
        b_q = parameters[layer,'b_q']
        b_k = parameters[layer,'b_k']
        b_v = parameters[layer,'b_v']
        b_o = parameters[layer,'b_o']
        
        if not b_q is None:  
            transformer.add_attention( input_name, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        else:
            print('no bias', b_q, b_k, b_v, b_o) 
            transformer.add_attention( input_name, layer, W_q, W_k, W_k, W_o)
        
        # add res+norm1
        enc_layer = 0
        gamma1 = parameters["enc__layer_normalization_1", 'gamma']
        beta1 = parameters["enc__layer_normalization_1", 'beta']
        
        transformer.add_residual_connection(input_name, "enc__self_attention_1", f"{layer}__{enc_layer}_residual_1")
        transformer.add_layer_norm(f"{layer}__{enc_layer}_residual_1", "enc_norm_1", gamma1, beta1)
        
        # add ffn1
        ffn_parameter_dict = {}
        input_shape = np.array(parameters["enc__ffn_1"]['input_shape'])
        ffn_params = transformer.get_fnn( "enc_norm_1", "enc__ffn_1", "enc__ffn_1", input_shape, parameters)
        ffn_parameter_dict["enc__ffn_1"] = ffn_params # ffn_params: nn, input_nn, output_nn
        

        # add res+norm2
        gamma2 = parameters["enc__layer_normalization_2", 'gamma']
        beta2 = parameters["enc__layer_normalization_2", 'beta']
        
        transformer.add_residual_connection("enc_norm_1", "enc__ffn_1", f"{layer}__{enc_layer}_residual_2")
        transformer.add_layer_norm(f"{layer}__{enc_layer}_residual_2", "enc_norm_2", gamma2, beta2)
        
        
        #add enc norm (norm over various encoder layers)
        gamma3 = parameters["enc_layer_normalization_1", 'gamma']
        beta3 = parameters["enc_layer_normalization_1", 'beta']
        transformer.add_layer_norm("enc_norm_2", "enc_norm_3", gamma3, beta3)
        
        # Add decoder
        # Add decoder self attention layer
        W_q = parameters["dec__self_attention_1",'W_q']
        W_k = parameters["dec__self_attention_1",'W_k']
        W_v = parameters["dec__self_attention_1",'W_v']
        W_o = parameters["dec__self_attention_1",'W_o']

        
        b_q = parameters["dec__self_attention_1",'b_q']
        b_k = parameters["dec__self_attention_1",'b_k']
        b_v = parameters["dec__self_attention_1",'b_v']
        b_o = parameters["dec__self_attention_1",'b_o']
        
        if not b_q is None:  
            transformer.add_attention( "dec_linear_1", "dec__self_attention_1", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        else:
            print('no bias', b_q, b_k, b_v, b_o) 
            transformer.add_attention( "dec_linear_1", "dec__self_attention_1", W_q, W_k, W_k, W_o)
        
        # decoder add res+norm1
        dec_layer = 0
        gamma1 = parameters["dec__layer_normalization_1", 'gamma']
        beta1 = parameters["dec__layer_normalization_1", 'beta']
        
        transformer.add_residual_connection("dec_linear_1", "dec__self_attention_1", f"dec__{dec_layer}_residual_1")
        transformer.add_layer_norm(f"dec__{dec_layer}_residual_1", "dec_norm_1", gamma1, beta1)
        
        # Add decoder cross attention
        W_q = parameters["dec__mutli_head_attention_1",'W_q'] # query from encoder
        W_k = parameters["dec__mutli_head_attention_1",'W_k']
        W_v = parameters["dec__mutli_head_attention_1",'W_v']
        W_o = parameters["dec__mutli_head_attention_1",'W_o']
        
        b_q = parameters["dec__mutli_head_attention_1",'b_q'] # query from encoder
        b_k = parameters["dec__mutli_head_attention_1",'b_k']
        b_v = parameters["dec__mutli_head_attention_1",'b_v']
        b_o = parameters["dec__mutli_head_attention_1",'b_o']
            
        transformer.add_attention( "dec_norm_1", "dec__mutli_head_attention_1", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, cross_attn=True, encoder_output="enc_norm_3")

        # decoder add res+norm2
        dec_layer = 0
        gamma2 = parameters["dec__layer_normalization_2", 'gamma']
        beta2 = parameters["dec__layer_normalization_2", 'beta']
        
        transformer.add_residual_connection("dec_norm_1", "dec__mutli_head_attention_1", f"dec__{dec_layer}_residual_2")
        transformer.add_layer_norm(f"dec__{dec_layer}_residual_2", "dec_norm_2", gamma2, beta2)
        
        # Add decoder FFN
        input_shape = np.array(parameters["dec__ffn_1"]['input_shape'])
        ffn_params = transformer.get_fnn( "dec_norm_2", "dec__ffn_1", "dec__ffn_1", input_shape, parameters)
        ffn_parameter_dict["dec__ffn_1"] = ffn_params # ffn_params: nn, input_nn, output_nn
        
        
        # decoder add res+norm3
        dec_layer = 0
        gamma3 = parameters["dec__layer_normalization_3", 'gamma']
        beta3 = parameters["dec__layer_normalization_3", 'beta']
        
        transformer.add_residual_connection("dec_norm_2", "dec__ffn_1", f"dec__{dec_layer}_residual_3")
        transformer.add_layer_norm(f"dec__{dec_layer}_residual_3", "dec_norm_3", gamma3, beta3)
        
        #add dec norm (norm over various decoder layers)
        gamma4 = parameters["dec_layer_normalization_1", 'gamma']
        beta4 = parameters["dec_layer_normalization_1", 'beta']
        transformer.add_layer_norm("dec_norm_3", "dec_norm_4", gamma4, beta4)
        
        # Linear transform
        embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
        W_linear = parameters["linear_2",'W']
        try:
            b_linear = parameters["linear_2",'b']
        except:
            b_linear = None
            
        embed_dim = transformer.M.input_dims
        transformer.embed_input( "dec_norm_4", "dec_linear_2", embed_dim, W_linear, b_linear)
        

        # Set objective
        m.obj = pyo.Objective(
            expr= sum((m.x1[t] - m.loc1[t])**2 + (m.x2[t] - m.loc2[t])**2 for t in m.time), sense=1
        )  # -1: maximize, +1: minimize (default)
    
    
        
    
        # Check new model attributes added
        self.assertIn(layer, dir(transformer.M))                        # check var created
        self.assertIsInstance(transformer.M.enc__self_attention_1, pyo.Var)               # check data type
        self.assertTrue(hasattr(transformer.M, 'Block_enc__self_attention_1'))            # check constraints created
        
     
        # Convert to gurobipy
        gurobi_model, map_var , _ = convert_pyomo.to_gurobi(m)
        
        
        # Add FNN1 to gurobi model
        for key, value in ffn_parameter_dict.items():
            nn, input_nn, output_nn = value
            input, output = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
            pred_constr = add_predictor_constr(gurobi_model, nn, input, output)
        
        gurobi_model.update() # update gurobi model with FFN constraints
        
        # Optimize
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
            
            
        # Check encoder outputs
        output_name = layer
        self_attn_enc = np.array(optimal_parameters[output_name])
        self_attn_expected_out = np.array(list(layer_outputs_dict['transformer.encoder.layers.0.self_attn'])[0][0]).flatten()

        norm1_enc = np.array(optimal_parameters["enc_norm_1"])
        norm1_expected = np.array(list(layer_outputs_dict['transformer.encoder.layers.0.norm1']))[0].flatten()

        ffn1_enc = np.array(optimal_parameters["enc__ffn_1"])
        ffn1_expected = np.array(list(layer_outputs_dict['transformer.encoder.layers.0.linear2']))[0].flatten()
        
        norm2_enc = np.array(optimal_parameters["enc_norm_2"])
        norm2_expected = np.array(list(layer_outputs_dict['transformer.encoder.layers.0.norm2']))[0].flatten()

        norm3_enc = np.array(optimal_parameters["enc_norm_3"])
        norm3_expected = np.array(list(layer_outputs_dict['transformer.encoder.norm']))[0].flatten()

        self.assertIsNone(np.testing.assert_array_equal(self_attn_expected_out.shape, self_attn_enc.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(self_attn_expected_out, self_attn_enc , decimal=5)) # compare value with transformer output
        print("- Enc MHA output formulation == Enc MHA Trained TNN")  
        
        self.assertIsNone(np.testing.assert_array_equal(norm1_expected.shape, norm1_enc.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(norm1_expected, norm1_enc , decimal=4)) # compare value with transformer output
        print("- Enc Norm1 formulation == Enc Norm1 Trained TNN")  
        
        self.assertIsNone(np.testing.assert_array_equal(ffn1_expected.shape, ffn1_enc.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(ffn1_expected, ffn1_enc , decimal=4)) # compare value with transformer output
        print("- Enc FFN1 formulation == Enc FNN1 Trained TNN") 
        
        self.assertIsNone(np.testing.assert_array_equal(norm2_expected.shape, norm2_enc.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(norm2_expected, norm2_enc , decimal=4)) # compare value with transformer output
        print("- Enc Norm2 formulation == Enc Norm2 Trained TNN")  
        
        self.assertIsNone(np.testing.assert_array_equal(norm3_expected.shape, norm3_enc.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(norm3_expected, norm3_enc , decimal=4)) # compare value with transformer output
        print("- Enc Output formulation == Enc Output Trained TNN")  
    
    
        # Check decoder outputs
        self_attn_dec = np.array(optimal_parameters["dec__self_attention_1"])
        self_attn_expected_out = np.array(list(layer_outputs_dict['transformer.decoder.layers.0.self_attn'])[0][0]).flatten()

        norm1_dec = np.array(optimal_parameters["dec_norm_1"])
        norm1_expected = np.array(list(layer_outputs_dict['transformer.decoder.layers.0.norm1']))[0].flatten()

        cross_dec = np.array(optimal_parameters["dec__mutli_head_attention_1"])
        cross_dec_expected = np.array(list(layer_outputs_dict['transformer.decoder.layers.0.multihead_attn'])[0][0]).flatten()

        norm2_dec = np.array(optimal_parameters["dec_norm_2"])
        norm2_dec_expected = np.array(list(layer_outputs_dict['transformer.decoder.layers.0.norm2']))[0].flatten()
        
        ffn1_dec = np.array(optimal_parameters["dec__ffn_1"])
        ffn1_dec_expected = np.array(list(layer_outputs_dict['transformer.decoder.layers.0.linear2']))[0].flatten()
        
        norm3_dec = np.array(optimal_parameters["dec_norm_3"])
        norm3_dec_expected = np.array(list(layer_outputs_dict['transformer.decoder.layers.0.norm3']))[0].flatten()
        
        norm4_dec = np.array(optimal_parameters["dec_norm_4"])
        norm4_dec_expected = np.array(list(layer_outputs_dict['transformer.decoder.norm']))[0].flatten()
        
    
        self.assertIsNone(np.testing.assert_array_equal(self_attn_expected_out.shape, self_attn_dec.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(self_attn_expected_out, self_attn_dec , decimal=4)) # compare value with transformer output
        print("- Dec MHA output formulation == Dec MHA Trained TNN")  
        
        self.assertIsNone(np.testing.assert_array_equal(norm1_expected.shape, norm1_dec.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(norm1_expected, norm1_dec , decimal=3)) # compare value with transformer output
        print("- Dec Norm1 formulation == Dec Norm1 Trained TNN") 
        
        self.assertIsNone(np.testing.assert_array_equal(cross_dec_expected.shape, cross_dec.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(cross_dec_expected, cross_dec , decimal=3)) # compare value with transformer output
        print("- Dec Cross Attn formulation == Dec Cross Attn Trained TNN") 
        
        self.assertIsNone(np.testing.assert_array_equal(norm2_dec_expected.shape, norm2_dec.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(norm2_dec_expected, norm2_dec , decimal=3)) # compare value with transformer output
        print("- Dec Norm2 formulation == Dec Norm2 Trained TNN") 
        
        self.assertIsNone(np.testing.assert_array_equal(ffn1_dec_expected.shape, ffn1_dec.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(ffn1_dec_expected, ffn1_dec , decimal=4)) # compare value with transformer output
        print("- Dec FFN1 formulation == Dec FFN1 Trained TNN") 
        
        self.assertIsNone(np.testing.assert_array_equal(norm3_dec_expected.shape, norm3_dec.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(norm3_dec_expected, norm3_dec , decimal=3)) # compare value with transformer output
        print("- Dec Norm3 formulation == Dec Norm3 Trained TNN") 
        
        self.assertIsNone(np.testing.assert_array_equal(norm4_dec_expected.shape, norm4_dec.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(norm4_dec_expected, norm4_dec , decimal=3)) # compare value with transformer output
        print("- Dec Output formulation == Dec Output Trained TNN") 
        
    
        # Check Transformer output
        out_dec = np.array(optimal_parameters["dec_linear_2"])
        out_expected_dec = np.array(list(layer_outputs_dict['linear2'])).flatten()
        
        # Assertions
        self.assertIsNone(np.testing.assert_array_equal(out_dec.shape, out_dec.shape)) # same shape
        self.assertIsNone(np.testing.assert_array_almost_equal(out_dec, out_dec, decimal=2))  # almost same values
     
        
   
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
    steps = 12 ##CHANGE THIS ##
    time = np.linspace(0, T_end, num=steps)
    dt = time[1] - time[0]
    tt = 10 # sequence size
    time_history = time[0:tt]
    pred_len = 2

    g = 9.81
    v_l1 = 0.2
    v_l2 = 1.5

    src = np.array([np.random.rand(1)*time[0:-1] , (2*np.random.rand(1) * time[0:-1]) - (0.5 * 9.81* time[0:-1] * time[0:-1])])# random sample input [x1_targte, x2_target]
    src = src.transpose(1,0)

    # define sets
    model.time = pyo.Set(initialize=time)
    model.time_history = pyo.Set(initialize=time[0:tt])
    # print(time, time[0:-1])
    
    # define parameters
    def target_location_rule(M, t):
        return v_l1 * t #+ (np.random.rand(1)/30)
    model.loc1 = pyo.Param(model.time, rule=target_location_rule) 

    def target_location2_rule(M, t):
        return (v_l2*t) - (0.5 * g * (t**2)) + (np.random.rand(1)/30)
    model.loc2 = pyo.Param(model.time, rule=target_location2_rule) 

    bounds_target = (-3,3)
    # define variables
    model.x1 = pyo.Var(model.time, bounds = bounds_target ) # distance path
    #model.v1 = pyo.Var(bounds=(0,None)) # initial velocity of cannon ball

    model.x2 = pyo.Var(model.time, bounds = bounds_target) # height path
    #model.v2 = pyo.Var(bounds=(0,None)) # initial velocity of cannon ball

    #model.T = pyo.Var(within=model.time)# time when cannon ball hits target

    # define initial conditions
    model.x1_constr = pyo.Constraint(expr= model.x1[0] == 0) 
    model.x2_constr = pyo.Constraint(expr= model.x2[0] == 0) 
    
    # # define constraints
    # def v1_rule(M, t):
    #     return M.x1[t] == M.v1 * t
    # model.v1_constr = pyo.Constraint(model.time, rule=v1_rule) 

    # def v2_rule(M, t):
    #     return M.x2[t] == (M.v2 * t) - (0.5*g * (t**2))
    # model.v2_constr = pyo.Constraint(model.time, rule=v2_rule)

    # model.v1_pos_constr = pyo.Constraint(expr = model.v1 >= 0)
    # model.v2_pos_constr = pyo.Constraint(expr = model.v2 >= 0)

    # def loc1_rule(M, t):
    #     return M.loc1[t] == model.history_loc1[t]
    # model.loc1_constr = pyo.Constraint(model.time_history, rule=loc1_rule)

    # def loc2_rule(M, t):
    #     return M.loc2[t] == model.history_loc2[t]
    # model.loc2_constr = pyo.Constraint(model.time_history, rule=loc2_rule)

    # load trained transformer
    sequence_size = tt
    head_size = 1
    device = torch.device('cpu')
    tnn_path = ".\\trained_transformer\\toy_pytorch_model_3.pt" # dmodel 4, num heads 1, n ence 1, n dec 1, head dim 4, pred_len 2+1 
    tnn_model = TransformerModel(input_dim=2, output_dim =2, d_model=4, nhead=head_size, num_encoder_layers=1, num_decoder_layers=1)
    tnn_model.load_state_dict(torch.load(tnn_path, map_location=device))
    
    # Fix model solution
    input_x1 =   v_l1 * time  
    input_x2 =  (v_l2*time) - (0.5 * g * (time*time))
    
    model.fixed_loc_constraints = pyo.ConstraintList()
    for i,t in enumerate(model.time):
        model.fixed_loc_constraints.add(expr= input_x1[i] == model.x1[t])
        model.fixed_loc_constraints.add(expr= input_x2[i]  == model.x2[t])

    # get intermediate results dictionary for optimal input values
    input = np.array([[ [x1,x2] for x1,x2 in zip(input_x1, input_x2)]], dtype=np.float32)
    # print(input.shape)
    
    layer_names, parameters, _, enc_dec_count, layer_outputs_dict = extract_from_pretrained.get_pytorch_learned_parameters(tnn_model, input[0, 0:tt, :] , input[0, -3:, :], head_size, tt)
    
    # print(dict_outputs)
    #output_dense_4 = layer_outputs_dict["dense_4"]
    
    # unit test
    enc_input_name = "enc_input"
    dec_input_name = "dec_input"
    unittest.main() 

