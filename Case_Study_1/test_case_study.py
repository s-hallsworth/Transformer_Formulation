# External imports
import pyomo.environ as pyo
import numpy as np
import unittest
import os
import torch
from gurobipy import GRB
#from gurobi_ml import add_predictor_constr
from gurobi_machinelearning.src.gurobi_ml.add_predictor import add_predictor_constr

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

# Import from repo file
import transformer_b_flag as TNN
from MINLP_tnn.helpers.GUROBI_ML_helper import get_inputs_gurobipy_FFN
import MINLP_tnn.helpers.convert_pyomo as convert_pyomo
import MINLP_tnn.helpers.extract_from_pretrained as extract_from_pretrained
import transformers
import sys
sys.modules['transformers.src.transformers'] = transformers
from transformers.models.time_series_transformer.configuration_time_series_transformer import TimeSeriesTransformerConfig
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerForPrediction
# cloned transformers from: https://github.com/s-hallsworth/transformers.git

"""
Test each module of reactor TNN
"""
class TestTransformer(unittest.TestCase):
    # def test_input_TNN(self):
    #     m = opt_model.clone()
        
    #     transformer = TNN.Transformer( ".\\data\\reactor_config_huggingface.json", m) 
        
    #     enc_dim_1 = src.size(0)
    #     dec_dim_1 = tgt.size(0)
    #     transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
    #     transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
    #     transformer.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) 
    #     transformer.M.model_dims = pyo.Set(initialize= list(range(transformer.d_model)))
    #     transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
    
    #     bounds_target = (None, None)
    #     # Add TNN input vars
    #     transformer.M.enc_input = pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
    #     transformer.M.dec_input = pyo.Var(transformer.M.dec_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        
    #     # Add constraints to TNN encoder input
    #     m.tnn_input_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.enc_input.index_set()).split("*"): # get TNN enc input index sets
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.enc_space):
    #         for tnn_dim, dim in zip(indices[1], m.dims):
    #             print(tnn_index, tnn_dim, index, dim)
    #             m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, tnn_dim] == m.x_enc[index, dim])
                
    #     # Add constraints to TNN decoder input
    #     indices = []
    #     for set in str(transformer.M.dec_input.index_set()).split("*"):# get TNN dec input index sets
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.dec_space):
    #         for tnn_dim, dim in zip(indices[1], m.dims):
    #             m.tnn_input_constraints.add(expr= transformer.M.dec_input[tnn_index, tnn_dim]== m.x[index, dim])
                
                
    #     # Set objective: maximise amount of methanol at reactor outlet
    #     m.obj = pyo.Objective(
    #             expr = m.x[m.dec_space.last(), "CH3OH"], sense=-1
    #         )  # -1: maximize, +1: minimize (default)
        
    #     # Convert to gurobi
    #     gurobi_model, map_var , _ = convert_pyomo.to_gurobi(m)
            
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
                    
    #     if gurobi_model.status == GRB.INFEASIBLE:
    #         gurobi_model.computeIIS()
    #         gurobi_model.write("pytorch_model.ilp")
            
    #     # model input
    #     enc_input_name = "enc_input"
    #     dec_input_name = "dec_input"
    #     model_enc_input = np.array(optimal_parameters[enc_input_name])
    #     model_dec_input = np.array(optimal_parameters[dec_input_name])
        
    #     expected_enc_input = src.numpy().flatten()#(src.numpy() - np.array(states_min)) / ( np.array(states_max) - np.array(states_min))
    #     expected_dec_input = tgt.numpy().flatten()#(tgt.numpy() - np.array(states_min)) / ( np.array(states_max) - np.array(states_min))
        
    #     model_tgt = np.array(optimal_parameters["x"])
    #     expected_tgt = tgt.numpy().flatten()
        
    #     # Assertions
    #     # self.assertIsNone(np.testing.assert_array_equal(model_tgt.shape, expected_tgt.shape)) 
    #     # self.assertIsNone(np.testing.assert_array_equal(model_tgt, expected_tgt)) 
    #     # print("input tgt = expected input tgt")
        
        
    #     self.assertIsNone(np.testing.assert_array_equal(model_enc_input.shape, expected_enc_input.flatten().shape)) # pyomo input data and transformer input data must be the same shape
    #     self.assertIsNone(np.testing.assert_array_almost_equal(model_enc_input, expected_enc_input.flatten(), decimal = 7))             # both inputs must be equal
    #     print("input enc tnn = expected enc input tnn")
        
    #     # print(model_dec_input-expected_dec_input.flatten())
    #     # self.assertIsNone(np.testing.assert_array_equal(model_dec_input.shape, expected_dec_input.flatten().shape)) # pyomo input data and transformer input data must be the same shape
    #     # self.assertIsNone(np.testing.assert_array_almost_equal(model_dec_input, expected_dec_input.flatten(), decimal = 7))             # both inputs must be equal
    #     # print("input dec tnn = expected dec input tnn")
        
        
        
    # def test_encoder_TNN(self):
    #     m = opt_model.clone()
        
    #     transformer = TNN.Transformer( ".\\data\\reactor_config_huggingface.json", m, activation_dict) 
        
    #     enc_dim_1 = src.size(0)
    #     dec_dim_1 = tgt.size(0)
    #     transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
    #     transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
    #     transformer.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) 
    #     transformer.M.model_dims = pyo.Set(initialize= list(range(transformer.d_model)))
    #     transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
    
    #     bounds_target = (None, None)
    #     # Add TNN input vars
    #     transformer.M.enc_input = pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
    #     transformer.M.dec_input = pyo.Var(transformer.M.dec_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        
    #     # Add constraints to TNN encoder input
    #     m.tnn_input_constraints = pyo.ConstraintList()
    #     indices = []
    #     for set in str(transformer.M.enc_input.index_set()).split("*"): # get TNN enc input index sets
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.enc_space):
    #         for tnn_dim, dim in zip(indices[1], m.dims):
    #             m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, tnn_dim] == m.x_enc[index, dim])
                
    #     # Add constraints to TNN decoder input
    #     indices = []
    #     for set in str(transformer.M.dec_input.index_set()).split("*"):# get TNN dec input index sets
    #         indices.append( getattr(m, set) )
    #     for tnn_index, index in zip(indices[0], m.dec_space):
    #         for tnn_dim, dim in zip(indices[1], m.dims):
    #             m.tnn_input_constraints.add(expr= transformer.M.dec_input[tnn_index, tnn_dim]== m.x[index, dim])
                
                
    #     # ADD ENCODER COMPONENTS
    #     # Add Linear transform
    #     # Linear transform
    #     embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
    #     layer = "enc_linear_1"
    #     W_linear = parameters[layer,'W']
    #     b_linear = parameters[layer,'b'] 
    #     transformer.embed_input( "enc_input", layer, embed_dim, W_linear, b_linear)
        
    #     # # # Add positiona encoding
    #     layer = "enc_pos_encoding_1"
    #     b_pe = parameters[layer,'b']
    #     transformer.add_pos_encoding("enc_linear_1", layer, b_pe)
        
        
    #     # add norm1
    #     layer = "enc__layer_normalization_1"
    #     gamma1 = parameters["enc__layer_normalization_1", 'gamma']
    #     beta1 = parameters["enc__layer_normalization_1", 'beta']
        
    #     transformer.add_layer_norm("enc_pos_encoding_1", "enc_norm_1", gamma1, beta1)
        
    #     # Add encoder self attention layer
    #     layer = "enc__self_attention_1"
        
    #     W_q = parameters[layer,'W_q']
    #     W_k = parameters[layer,'W_k']
    #     W_v = parameters[layer,'W_v']
    #     W_o = parameters[layer,'W_o']
    #     b_q = parameters[layer,'b_q']
    #     b_k = parameters[layer,'b_k']
    #     b_v = parameters[layer,'b_v']
    #     b_o = parameters[layer,'b_o']
         
    #     transformer.add_attention( "enc_norm_1", layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, norm_softmax=False)
        
    #     # add res+norm2
    #     layer = "enc__layer_normalization_2"
    #     gamma1 = parameters[layer, 'gamma']
    #     beta1 = parameters[layer, 'beta']
        
    #     transformer.add_residual_connection("enc_norm_1", "enc__self_attention_1", f"{layer}__residual_1")
    #     transformer.add_layer_norm(f"{layer}__residual_1", "enc_norm_2", gamma1, beta1)
         
    #     # add ffn1
    #     ffn_parameter_dict = {}
    #     input_shape = parameters["enc__ffn_1"]['input_shape']
    #     ffn_params = transformer.get_ffn( "enc_norm_2", "enc__ffn_1", "enc__ffn_1", input_shape, parameters)
    #     ffn_parameter_dict["enc__ffn_1"] = ffn_params # ffn_params: nn, input_nn, output_nn

    #     # add res+norm3
    #     layer = "enc__layer_normalization_3"
    #     gamma1 = parameters[layer, 'gamma']
    #     beta1 = parameters[layer, 'beta']
        
    #     transformer.add_residual_connection("enc_norm_2", "enc__ffn_1", f"{layer}__residual_1")
    #     transformer.add_layer_norm(f"{layer}__residual_1", "enc_norm_3", gamma1, beta1)
        
    #     ## Encoder Layer 2:
        
    #     # Add residual
        
    #     # Add encoder self attention layer
    #     layer = "enc__self_attention_2"
    #     W_q = parameters[layer,'W_q']
    #     W_k = parameters[layer,'W_k']
    #     W_v = parameters[layer,'W_v']
    #     W_o = parameters[layer,'W_o']
    #     b_q = parameters[layer,'b_q']
    #     b_k = parameters[layer,'b_k']
    #     b_v = parameters[layer,'b_v']
    #     b_o = parameters[layer,'b_o']
    #     transformer.add_attention( "enc_norm_3", layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, norm_softmax=False)
        
    #     #add res+norm4
    #     layer = "enc__layer_normalization_4"
    #     gamma1 = parameters[layer, 'gamma']
    #     beta1 = parameters[layer, 'beta']
        
    #     transformer.add_residual_connection("enc_norm_1", "enc__self_attention_2", f"{layer}__residual_1")
    #     transformer.add_layer_norm(f"{layer}__residual_1", "enc_norm_4", gamma1, beta1)
        
        
    #     # add ffn2
    #     ffn_parameter_dict = {}
    #     input_shape = parameters["enc__ffn_2"]['input_shape']
    #     ffn_params = transformer.get_ffn( "enc_norm_4", "enc__ffn_2", "enc__ffn_2", input_shape, parameters)
    #     ffn_parameter_dict["enc__ffn_2"] = ffn_params # ffn_params: nn, input_nn, output_nn

    #     # add res+norm5
    #     layer = "enc__layer_normalization_5"
    #     gamma1 = parameters[layer, 'gamma']
    #     beta1 = parameters[layer, 'beta']
        
    #     transformer.add_residual_connection("enc_norm_4", "enc__ffn_2", f"{layer}__residual_1")
    #     transformer.add_layer_norm(f"{layer}__residual_1", "enc_norm_5", gamma1, beta1)
         
         
        
    #     # Set objective: maximise amount of methanol at reactor outlet
    #     m.obj = pyo.Objective(
    #             expr = m.x[m.dec_space.last(), "CH3OH"], sense=-1
    #         )  # -1: maximize, +1: minimize (default)
        
    #     # Convert to gurobi
    #     gurobi_model, map_var , _ = convert_pyomo.to_gurobi(m)
        
    #     # Add FNN1 to gurobi model
    #     for key, value in ffn_parameter_dict.items():
    #         nn, input_nn, output_nn = value
    #         input, output = get_inputs_gurobipy_FFN(input_nn, output_nn, map_var)
    #         pred_constr = add_predictor_constr(gurobi_model, nn, input, output)
        
    #     gurobi_model.update() # update gurobi model with FFN constraints
        
            
    #     # Optimize
    #     #gurobi_model.setParam('MIPFocus', 1) 
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
    #         gurobi_model.computeIIS()
    #         gurobi_model.write("pytorch_model.ilp")
            
    #     # input TNN
    #     model_enc_input = np.array(optimal_parameters["enc_input"])
    #     model_dec_input = np.array(optimal_parameters["dec_input"])
    #     expected_enc_input = src.numpy()
    #     expected_dec_input = tgt.numpy()

    #     self.assertIsNone(np.testing.assert_array_equal(model_enc_input.shape, expected_enc_input.flatten().shape)) # pyomo input data and transformer input data must be the same shape
    #     self.assertIsNone(np.testing.assert_array_almost_equal(model_enc_input, expected_enc_input.flatten(), decimal = 7))             # both inputs must be equal
        
    #     self.assertIsNone(np.testing.assert_array_equal(model_dec_input.shape, expected_dec_input.flatten().shape)) # pyomo input data and transformer input data must be the same shape
    #     self.assertIsNone(np.testing.assert_array_almost_equal(model_dec_input, expected_dec_input.flatten(), decimal = 7))             # both inputs must be equal
    #     print("input tnn = expected input tnn")
            
    #     # Linear
    #     enc_linear_1 = np.array(optimal_parameters["enc_linear_1"])
    #     enc_linear_1_expected = np.array(list(layer_outputs_dict['model.encoder.value_embedding.value_projection']))[0].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(enc_linear_1.shape, enc_linear_1_expected.shape)) 
    #     self.assertIsNone(np.testing.assert_array_almost_equal(enc_linear_1, enc_linear_1_expected, decimal = 7)) 
    #     print("Enc linear 1 = expected enc linear 1")
            
    #     # Positional Embedding
    #     enc_pe = np.array(optimal_parameters["enc_pos_encoding_1"])
    #     enc_pe_expected = np.array(list(layer_outputs_dict['model.encoder.embed_positions']))[0].flatten() + enc_linear_1_expected
        
    #     self.assertIsNone(np.testing.assert_array_equal(enc_pe.shape, enc_pe_expected.shape)) 
    #     self.assertIsNone(np.testing.assert_array_almost_equal(enc_pe, enc_pe_expected, decimal = 7)) 
    #     print("Enc PE = expected Enc PE")
        
    #     # Norm 1
    #     norm1 = np.array(optimal_parameters["enc_norm_1"])
    #     norm1_expected = np.array(list(layer_outputs_dict['model.encoder.layernorm_embedding']))[0].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(norm1.shape, norm1_expected.shape)) 
    #     self.assertIsNone(np.testing.assert_array_almost_equal(norm1, norm1_expected, decimal = 6)) 
    #     print("Enc Norm1= expected Enc Norm1")
        
    #     #Self attn
    #     self_attn_enc = np.array(optimal_parameters["enc__self_attention_1"])
    #     self_attn_expected_out = np.array(list(layer_outputs_dict['model.encoder.layers.0.self_attn.out_proj'])[0][0]).flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(self_attn_expected_out.shape, self_attn_enc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(self_attn_expected_out, self_attn_enc , decimal=6)) # compare value with transformer output
    #     print("Enc MHA output formulation == Enc MHA Trained TNN")  
        
    #     # Norm 2
    #     norm2 = np.array(optimal_parameters["enc_norm_2"])
    #     norm2_expected = np.array(list(layer_outputs_dict['model.encoder.layers.0.self_attn_layer_norm']))[0].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(norm2.shape, norm2_expected.shape)) 
    #     self.assertIsNone(np.testing.assert_array_almost_equal(norm2, norm2_expected, decimal = 5)) 
    #     print("Enc Norm2= expected Enc Norm2")
        
    #     # FFN 1 
    #     ffn1_enc = np.array(optimal_parameters["enc__ffn_1"])
    #     ffn1_expected = np.array(list(layer_outputs_dict['model.encoder.layers.0.fc2']))[0].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(ffn1_expected.shape, ffn1_enc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(ffn1_expected, ffn1_enc , decimal=5)) # compare value with transformer output
    #     print("Enc FFN1 formulation == Enc FNN1 Trained TNN") 
        
    #     # Norm 3
    #     norm3 = np.array(optimal_parameters["enc_norm_3"])
    #     norm3_expected = np.array(list(layer_outputs_dict['model.encoder.layers.0.final_layer_norm']))[0].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(norm3.shape, norm3_expected.shape)) 
    #     self.assertIsNone(np.testing.assert_array_almost_equal(norm3, norm3_expected, decimal = 5)) 
    #     print("Enc Norm3 = expected Enc Norm3")
        
    #     #ENc Layer 2: Self attn
    #     self_attn2_enc = np.array(optimal_parameters["enc__self_attention_2"])
    #     self_attn2_expected_out = np.array(list(layer_outputs_dict['model.encoder.layers.1.self_attn.out_proj'])[0][0]).flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(self_attn2_expected_out.shape, self_attn2_enc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(self_attn2_expected_out, self_attn2_enc , decimal=5)) # compare value with transformer output
    #     print("Enc MHA Layer 2 == expected Enc MHA Layer 2 ")  
        
    #     # Norm 4
    #     norm4 = np.array(optimal_parameters["enc_norm_4"])
    #     norm4_expected = np.array(list(layer_outputs_dict['model.encoder.layers.1.self_attn_layer_norm']))[0].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(norm4.shape, norm4_expected.shape)) 
    #     self.assertIsNone(np.testing.assert_array_almost_equal(norm4, norm4_expected, decimal = 4)) 
    #     print("Enc Norm4 = expected Enc Norm4")
        
    #     # FFN 2 
    #     ffn2_enc = np.array(optimal_parameters["enc__ffn_2"])
    #     ffn2_expected = np.array(list(layer_outputs_dict['model.encoder.layers.1.fc2']))[0].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(ffn2_expected.shape, ffn2_enc.shape)) # compare shape with transformer
    #     self.assertIsNone(np.testing.assert_array_almost_equal(ffn2_expected, ffn2_enc , decimal=4)) # compare value with transformer output
    #     print("Enc FFN2 formulation == Enc FNN2 Trained TNN")
        
    #     #Enc layer 2: norm final
    #     norm5 = np.array(optimal_parameters["enc_norm_5"])
    #     norm5_expected = np.array(list(layer_outputs_dict['model.encoder.layers.1.self_attn_layer_norm']))[0].flatten()
        
    #     self.assertIsNone(np.testing.assert_array_equal(norm5.shape, norm5_expected.shape)) 
    #     self.assertIsNone(np.testing.assert_array_almost_equal(norm5, norm5_expected, decimal = 5)) 
    #     print("Enc Output = expected Enc Output")
        
        
    def test_decoder_TNN(self):
        m = opt_model.clone()
        
        transformer = TNN.Transformer( ".\\data\\reactor_config_huggingface.json", m, activation_dict) 
        
        # Define tranformer
        enc_dim_1 = src.size(0)
        transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
        transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
    
        bounds_target = (None, None)
        # Add TNN input vars
        transformer.M.enc_input = pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        
        # Add constraints to TNN encoder input
        m.tnn_input_constraints = pyo.ConstraintList()
        indices = []
        for set in str(transformer.M.enc_input.index_set()).split("*"): # get TNN enc input index sets
            indices.append( getattr(m, set) )
        for tnn_index, index in zip(indices[0], m.enc_space):
            for tnn_dim, dim in zip(indices[1], m.dims):
                m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, tnn_dim] * (opt_model.states_max[dim] - opt_model.states_min[dim] ) == (m.x_enc[index, dim] - opt_model.states_min[dim] ))
                
        out = transformer.M.enc_input        
        # ADD ENCODER COMPONENTS
        # Add Linear transform
        # Linear transform
        embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
        layer = "enc_linear_1"
        W_linear = np.array(parameters[layer,'W'])
        b_linear = parameters[layer,'b']
        out = transformer.embed_input( "enc_input", layer, embed_dim, W_linear, b_linear)
        
        
        # # # Add positiona encoding
        layer = "enc_pos_encoding_1"
        b_pe = parameters[layer,'b']
        transformer.add_pos_encoding(out, layer, b_pe)
        
        
        # add norm1
        layer = "enc__layer_normalization_1"
        gamma1 = parameters["enc__layer_normalization_1", 'gamma']
        beta1 = parameters["enc__layer_normalization_1", 'beta']
        
        transformer.add_layer_norm("enc_pos_encoding_1", "enc_norm_1", gamma1, beta1)
        
        # Add encoder self attention layer
        layer = "enc__self_attention_1"
        
        W_q = parameters[layer,'W_q']
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']
        b_q = parameters[layer,'b_q']
        b_k = parameters[layer,'b_k']
        b_v = parameters[layer,'b_v']
        b_o = parameters[layer,'b_o']
         
        transformer.add_attention( "enc_norm_1", layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, norm_softmax=False)
        
        # add res+norm2
        layer = "enc__layer_normalization_2"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        transformer.add_residual_connection("enc_norm_1", "enc__self_attention_1", f"{layer}__residual_1")
        transformer.add_layer_norm(f"{layer}__residual_1", "enc_norm_2", gamma1, beta1)
         
        # add ffn1
        ffn_parameter_dict = {}
        input_shape = parameters["enc__ffn_1"]['input_shape']
        ffn_params = transformer.get_ffn( "enc_norm_2", "enc__ffn_1", "enc__ffn_1", input_shape, parameters)
        ffn_parameter_dict["enc__ffn_1"] = ffn_params # ffn_params: nn, input_nn, output_nn

        # add res+norm2
        layer = "enc__layer_normalization_3"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        transformer.add_residual_connection("enc_norm_2", "enc__ffn_1", f"{layer}__residual_1")
        transformer.add_layer_norm(f"{layer}__residual_1", "enc_norm_3", gamma1, beta1)
        
        ## Encoder Layer 2:
        # Add encoder self attention layer
        layer = "enc__self_attention_2"
        W_q = parameters[layer,'W_q']
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']
        b_q = parameters[layer,'b_q']
        b_k = parameters[layer,'b_k']
        b_v = parameters[layer,'b_v']
        b_o = parameters[layer,'b_o']
        transformer.add_attention( "enc_norm_3", layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, norm_softmax=False)
        
        #add res+norm2
        layer = "enc__layer_normalization_4"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        transformer.add_residual_connection("enc_norm_1", "enc__self_attention_2", f"{layer}__residual_1")
        transformer.add_layer_norm(f"{layer}__residual_1", "enc_norm_4", gamma1, beta1)
         
        # add ffn1
        ffn_parameter_dict = {}
        input_shape = parameters["enc__ffn_2"]['input_shape']
        ffn_params = transformer.get_ffn( "enc_norm_4", "enc__ffn_2", "enc__ffn_2", input_shape, parameters)
        ffn_parameter_dict["enc__ffn_2"] = ffn_params # ffn_params: nn, input_nn, output_nn

        # add res+norm2
        layer = "enc__layer_normalization_5"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        transformer.add_residual_connection("enc_norm_4", "enc__ffn_2", f"{layer}__residual_1")
        transformer.add_layer_norm(f"{layer}__residual_1", "enc_norm_5", gamma1, beta1)
         
        ## Decoder
        # Add constraints to TNN decoder input
        dec_dim_1 = tgt.size(0)
        transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
        transformer.M.dec_input = pyo.Var(transformer.M.dec_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        transformer.M.dec_output = pyo.Var(transformer.M.dec_time_dims,  transformer.M.input_dims, bounds=bounds_target)

        indices = []
        for set in str(transformer.M.dec_input.index_set()).split("*"):# get TNN dec input index sets
            indices.append( getattr(m, set) )

        # link decoder input value to variable storing states
        for tnn_index, index in zip(indices[0], m.dec_space):
            for tnn_dim, dim in zip(indices[1], m.dims):
                m.tnn_input_constraints.add(expr= transformer.M.dec_input[tnn_index, tnn_dim] * (opt_model.states_max[dim] - opt_model.states_min[dim] ) == (m.x[index, dim] - opt_model.states_min[dim] ))
        dec_in = transformer.M.dec_input
        
        # ## Dec Add Linear:
        embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
        layer = "dec_linear_1"
        W_linear = parameters[layer,'W']
        b_linear = parameters[layer,'b'] 
        
        layer = layer
        dec_in = transformer.embed_input( dec_in, layer, embed_dim, W_linear, b_linear)
        
        # Dec Add positiona encoding
        layer = "dec_pos_encoding_1"
        b_pe = parameters[layer,'b']
        
        layer = layer
        dec_in = transformer.add_pos_encoding(dec_in, layer, b_pe)
        
        # Dec Add norm1
        layer = "dec__layer_normalization_1"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        layer = "dec_norm_1"
        dec_in = transformer.add_layer_norm(dec_in, layer, gamma1, beta1)
        dec_norm_1 = dec_in
        
        # Dec Add decoder self attention layer
        layer = "dec__self_attention_1"
        W_q = parameters[layer,'W_q']
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']
        b_q = parameters[layer,'b_q']
        b_k = parameters[layer,'b_k']
        b_v = parameters[layer,'b_v']
        b_o = parameters[layer,'b_o']
        
        layer = layer
        dec_in = transformer.add_attention( dec_in, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, mask=True)
        
        # Dec add res+norm2
        layer = "dec__layer_normalization_2"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        layer = "d_LN_2"
        dec_in = transformer.add_residual_connection(dec_norm_1, dec_in, f"{layer}__res")
        
        layer = "dec_norm_2"
        dec_in = transformer.add_layer_norm(dec_in, layer, gamma1, beta1)
        dec_norm_2 = dec_in
        
        # Dec Cross Attn
        layer = "dec__multi_head_attention_1" 
        W_q = parameters["dec__multi_head_attention_1",'W_q'] # query from encoder
        W_k = parameters["dec__multi_head_attention_1",'W_k']
        W_v = parameters["dec__multi_head_attention_1",'W_v']
        W_o = parameters["dec__multi_head_attention_1",'W_o']
        
        b_q = parameters["dec__multi_head_attention_1",'b_q'] # query from encoder
        b_k = parameters["dec__multi_head_attention_1",'b_k']
        b_v = parameters["dec__multi_head_attention_1",'b_v']
        b_o = parameters["dec__multi_head_attention_1",'b_o']
        
        layer = layer 
        dec_in = transformer.add_attention( dec_in, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, cross_attn=True, encoder_output="enc_norm_5")

        
        # add res+norm3
        layer = "dec__layer_normalization_3"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        layer = "d_ln_3" 
        dec_in = transformer.add_residual_connection(dec_norm_2, dec_in, f"{layer}__res")
        
        layer = "dec_norm_3"
        dec_in = transformer.add_layer_norm(dec_in, layer, gamma1, beta1)
        dec_norm_3 = dec_in
        
        # add FFN
        nn_name = "dec__ffn_1"
        input_shape = parameters[nn_name]['input_shape']
        layer = nn_name
        ffn_params = transformer.get_ffn( dec_norm_3,layer, nn_name, input_shape, parameters)
        ffn_parameter_dict[nn_name] = ffn_params # ffn_params: nn, input_nn, output_nn
        dec_in = ffn_params[-1]
        
        # add Norm 4
        layer = "dec__layer_normalization_4"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        layer = "d_ln_4" 
        dec_in = transformer.add_residual_connection(dec_norm_3, dec_in, f"{layer}__res")
        
        layer = "dec_norm_4"
        dec_in = transformer.add_layer_norm(dec_in, layer, gamma1, beta1)
        dec_norm_4 = dec_in
        
        ##-- Decoder Layer 2:
        # Dec Add decoder self attention layer
        layer = "dec__self_attention_2"
        W_q = parameters[layer,'W_q']
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']
        b_q = parameters[layer,'b_q']
        b_k = parameters[layer,'b_k']
        b_v = parameters[layer,'b_v']
        b_o = parameters[layer,'b_o']
        
        layer = layer
        dec_in = transformer.add_attention( dec_in, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, mask=True)
        
        # Dec add res+norm2
        layer = "dec__layer_normalization_5"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        layer = "d_LN_5"
        dec_in = transformer.add_residual_connection(dec_norm_1, dec_in, f"{layer}__res") ## res to LN1
        
        layer = "dec_norm_5"
        dec_in = transformer.add_layer_norm(dec_in, layer, gamma1, beta1)
        dec_norm_5 = dec_in
        
        # Dec Cross Attn
        layer = "dec__multi_head_attention_2" 
        W_q = parameters[layer,'W_q'] # query from encoder
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']
        
        b_q = parameters[layer,'b_q'] # query from encoder
        b_k = parameters[layer,'b_k']
        b_v = parameters[layer,'b_v']
        b_o = parameters[layer,'b_o']
        
        layer = layer 
        dec_in = transformer.add_attention( dec_in, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, cross_attn=True, encoder_output="enc_norm_5")

        
        # add res+norm6
        layer = "dec__layer_normalization_6"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        layer = "d_ln_6" 
        dec_in = transformer.add_residual_connection(dec_norm_5, dec_in, f"{layer}__res")
        
        layer = "dec_norm_6"
        dec_in = transformer.add_layer_norm(dec_in, layer, gamma1, beta1)
        dec_norm_6 = dec_in
        
        # add FFN
        nn_name = "dec__ffn_2"
        input_shape = parameters[nn_name]['input_shape']
        layer = nn_name
        ffn_params = transformer.get_ffn( dec_norm_3,layer, nn_name, input_shape, parameters)
        ffn_parameter_dict[nn_name] = ffn_params # ffn_params: nn, input_nn, output_nn
        dec_in = ffn_params[-1]
        
        # add Norm 4
        layer = "dec__layer_normalization_7"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        layer = "d_ln_7" 
        dec_in = transformer.add_residual_connection(dec_norm_6, dec_in, f"{layer}__res")
        
        layer = "dec_norm_7"
        dec_in = transformer.add_layer_norm(dec_in, layer, gamma1, beta1)
        dec_norm_7 = dec_in
        
        # Linear transform 1
        transformer.M.dims_9 = pyo.Set(initialize= list(range(9)))
        embed_dim = transformer.M.dims_9
        
        layer = "linear_1" #degree of freedom
        W_linear = parameters[layer,'W']
        b_linear = parameters[layer,'b'] 
        out1 = transformer.embed_input( dec_in, layer, embed_dim, W_linear, b_linear)
        
        layer = "linear_2" #mean
        W_linear = parameters[layer,'W']
        b_linear = parameters[layer,'b'] 
        out2 = transformer.embed_input( dec_in, layer, embed_dim, W_linear, b_linear)
        
        layer = "linear_3" #scale
        W_linear = parameters[layer,'W']
        b_linear = parameters[layer,'b'] 
        out3 = transformer.embed_input( dec_in, layer, embed_dim, W_linear, b_linear)
        
        # transformer.M.out = pyo.Var(transformer.M.dec_time_dims,  embed_dim)
        # for i in out1.index_set():
        
        # W_linear = parameters[layer,'W']
        # #b_linear = parameters[layer,'b'] 
        # W = np.array(W_linear)
        # W = np.linalg.pinv(W) # get inverse
        # print(W.shape, W)
        
        # W_emb_dict = {
        #             (m.dims.at(s+1),embed_dim.at(d+1)): W[s][d]
        #             for s in range(len(m.dims))
        #             for d in range(len(embed_dim))
        #         } 
        # m.W_emb = pyo.Param(m.dims, embed_dim , initialize=W_emb_dict)
             
        for tnn_index, index in zip(indices[0], m.dec_space):
            for d, tnn_dim in enumerate(embed_dim):
                dim = m.dims.at(d+1)
                m.tnn_input_constraints.add(expr= out2[tnn_index, tnn_dim] * (opt_model.states_max[dim] - opt_model.states_min[dim] ) == (m.x[index, dim]- opt_model.states_min[dim] ) )
        
        
        ##----------------------------------------------------------------##
        ## Set objective: maximise amount of methanol at reactor outlet
        m.obj = pyo.Objective(
                expr = m.x[m.dec_space.last(), "CH3OH"], sense=-1
            )  # -1: maximize, +1: minimize (default)
        
        # Convert to gurobi
        gurobi_model, map_var , _ = convert_pyomo.to_gurobi(m)
        
        # Add FNN1 to gurobi model
        for key, value in ffn_parameter_dict.items():
            nn, input_nn, output_nn = value
            input, output = get_inputs_gurobipy_FFN(input_nn, output_nn, map_var)
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
          
        #     ##----------------------------------------------------------------##  
        # input TNN
        # model_enc_input = np.array(optimal_parameters["enc_input"])
        # #model_dec_input = np.array(optimal_parameters["dec_input"])
        # expected_enc_input = src.numpy()
        # expected_dec_input = tgt.numpy()

        # self.assertIsNone(np.testing.assert_array_equal(model_enc_input.shape, expected_enc_input.flatten().shape)) # pyomo input data and transformer input data must be the same shape
        # self.assertIsNone(np.testing.assert_array_almost_equal(model_enc_input, expected_enc_input.flatten(), decimal = 7))             # both inputs must be equal
        
        # # self.assertIsNone(np.testing.assert_array_equal(model_dec_input.shape, expected_dec_input.flatten().shape)) # pyomo input data and transformer input data must be the same shape
        # # self.assertIsNone(np.testing.assert_array_almost_equal(model_dec_input, expected_dec_input.flatten(), decimal = 7))             # both inputs must be equal
        # print("input tnn = expected input tnn")
        
        # # Positional Embedding
        # enc_pe = np.array(optimal_parameters["enc_pos_encoding_1"])
        # enc_linear_1_expected = np.array(list(layer_outputs_dict['model.encoder.value_embedding.value_projection']))[0].flatten()
        # enc_pe_expected = np.array(list(layer_outputs_dict['model.encoder.embed_positions']))[0].flatten() + enc_linear_1_expected
        
        
        # Enc output:
        norm5 = np.array(optimal_parameters["enc_norm_5"])
        norm5_expected = np.array(list(layer_outputs_dict['model.encoder.layers.1.self_attn_layer_norm']))[0].flatten()
        
        self.assertIsNone(np.testing.assert_array_equal(norm5.shape, norm5_expected.shape)) 
        self.assertIsNone(np.testing.assert_array_almost_equal(norm5, norm5_expected, decimal = 4)) 
        print("Enc Norm5 = expected Enc Norm5")
            
        
        #Dec Linear
        # t = 0
        # dec_linear_1 = np.array(optimal_parameters["dec_linear_1"])
        # dec_linear_1__expected = []
        # for x in list(layer_outputs_dict['model.decoder.value_embedding.value_projection']):
        #     if np.array(x).shape[1] ==  8:   
        #         dec_linear_1__expected = np.array(x.tolist())
        # dec_linear_1_expected = dec_linear_1__expected[0].flatten('C')

        # print(np.mean(dec_linear_1 - dec_linear_1_expected), max(dec_linear_1 - dec_linear_1_expected), min(dec_linear_1 - dec_linear_1_expected))
        # # self.assertIsNone(np.testing.assert_array_equal(dec_linear_1.shape, dec_linear_1_expected.shape)) 
        # # self.assertIsNone(np.testing.assert_array_almost_equal(dec_linear_1, dec_linear_1_expected, decimal = 3)) 
        # print("Dec linear 1 = expected Dec linear 1")
        
        # Positional Embedding
        # dec_expected = []
        # dec_pe = np.array(optimal_parameters["dec_pos_encoding_1"])
        # for x in list(layer_outputs_dict['model.decoder.embed_positions']):
        #     if np.array(x).shape[0] ==  8:   
        #         dec_expected = np.array(x)
        # dec_pe_expected= dec_expected.flatten('C') + dec_linear_1_expected 

        # print(np.mean(dec_pe - dec_pe_expected), max(dec_pe - dec_pe_expected), min(dec_pe - dec_pe_expected))
        # # self.assertIsNone(np.testing.assert_array_equal(dec_pe.shape, dec_pe_expected.shape)) 
        # # self.assertIsNone(np.testing.assert_array_almost_equal(dec_pe, dec_pe_expected, decimal = 3)) 
        # print("Dec PE = expected Dec PE")
        
        # Norm 1
        norm1 = np.array(optimal_parameters["dec_norm_1"])
        for x in list(layer_outputs_dict['model.decoder.layernorm_embedding']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        
        # self.assertIsNone(np.testing.assert_array_equal(norm1.shape, expected.shape)) 
        # self.assertIsNone(np.testing.assert_array_almost_equal(norm1, expected, decimal = 4)) 
        print(np.mean(norm1 - expected), max(norm1  - expected), min(norm1  - expected))
        print("Dec Norm1= expected Dec Norm1")
        
        #Self attn
        self_attn_dec = np.array(optimal_parameters["dec__self_attention_1"])
        for x in list(layer_outputs_dict['model.decoder.layers.0.self_attn.out_proj']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        
        # self.assertIsNone(np.testing.assert_array_equal(self_attn_expected_out.shape, self_attn_dec.shape)) # compare shape with transformer
        # self.assertIsNone(np.testing.assert_array_almost_equal(self_attn_expected_out, self_attn_dec , decimal=4)) # compare value with transformer output
        print(np.mean(self_attn_dec - expected), max(self_attn_dec  - expected), min(self_attn_dec - expected))
        print("Dec MHA output formulation == Dec MHA Trained TNN")
          
        # Norm 2
        norm2 = np.array(optimal_parameters["dec_norm_2"])
        for x in list(layer_outputs_dict['model.decoder.layers.0.self_attn_layer_norm']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        
        # self.assertIsNone(np.testing.assert_array_equal(norm2.shape, norm2_expected.shape)) 
        # self.assertIsNone(np.testing.assert_array_almost_equal(norm2, norm2_expected, decimal = 4)) 
        print(np.mean(norm2 - expected), max(norm2 - expected), min(norm2 - expected))
        print("Dec Norm2= expected Dec Norm2")
        
        #Cross attn
        self_cross_attn_dec = np.array(optimal_parameters["dec__multi_head_attention_1"])
        for x in list(layer_outputs_dict['model.decoder.layers.0.encoder_attn.out_proj']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        # self.assertIsNone(np.testing.assert_array_equal(self_cross_attn_expected_out.shape, self_cross_attn_dec.shape)) # compare shape with transformer
        # self.assertIsNone(np.testing.assert_array_almost_equal(self_cross_attn_expected_out, self_cross_attn_dec , decimal=4)) # compare value with transformer output
        print(np.mean(self_cross_attn_dec - expected), max(self_cross_attn_dec - expected), min(self_cross_attn_dec - expected))
        print("Dec Cross Attn output formulation == Dec Cross Attn Trained TNN")
        
        
        # FFN 1 
        ffn1_enc = np.array(optimal_parameters["dec__ffn_1"])
        for x in list(layer_outputs_dict['model.decoder.layers.0.fc2']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        
        # self.assertIsNone(np.testing.assert_array_equal(ffn1_expected.shape, ffn1_enc.shape)) # compare shape with transformer
        # self.assertIsNone(np.testing.assert_array_almost_equal(ffn1_expected, ffn1_enc , decimal=6)) # compare value with transformer output
        print(np.mean(ffn1_enc - expected), max(ffn1_enc - expected), min(ffn1_enc - expected))
        print("Dec FFN1 formulation == Dec FNN1 Trained TNN") 
        
        # Norm 3
        norm3 = np.array(optimal_parameters["dec_norm_4"])
        for x in list(layer_outputs_dict['model.decoder.layers.0.final_layer_norm']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        
        # self.assertIsNone(np.testing.assert_array_equal(ffn1_expected.shape, ffn1_enc.shape)) # compare shape with transformer
        # self.assertIsNone(np.testing.assert_array_almost_equal(ffn1_expected, ffn1_enc , decimal=6)) # compare value with transformer output
        print(np.mean(norm3 - expected), max(norm3 - expected), min(norm3 - expected))
        print("Dec Layer 1 Out formulation == Dec Layer 1 Out Trained TNN") 
        
        #----------- DECODER LAYER 2 ----------#   
        
        #Self attn
        self_attn_dec = np.array(optimal_parameters["dec__self_attention_2"])
        for x in list(layer_outputs_dict['model.decoder.layers.1.self_attn.out_proj']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        
        # self.assertIsNone(np.testing.assert_array_equal(self_attn_expected_out.shape, self_attn_dec.shape)) # compare shape with transformer
        # self.assertIsNone(np.testing.assert_array_almost_equal(self_attn_expected_out, self_attn_dec , decimal=4)) # compare value with transformer output
        print(np.mean(self_attn_dec - expected), max(self_attn_dec  - expected), min(self_attn_dec - expected))
        print("Dec MHA 2 formulation == Dec MHA 2 Trained TNN")
          
        # Norm 5
        norm2 = np.array(optimal_parameters["dec_norm_5"])
        for x in list(layer_outputs_dict['model.decoder.layers.0.self_attn_layer_norm']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        
        # self.assertIsNone(np.testing.assert_array_equal(norm2.shape, norm2_expected.shape)) 
        # self.assertIsNone(np.testing.assert_array_almost_equal(norm2, norm2_expected, decimal = 4)) 
        print(np.mean(norm2 - expected), max(norm2 - expected), min(norm2 - expected))
        print("Dec Norm 5 = expected Dec Norm 5")
        
        #Cross attn
        self_cross_attn_dec = np.array(optimal_parameters["dec__multi_head_attention_2"])
        for x in list(layer_outputs_dict['model.decoder.layers.1.encoder_attn.out_proj']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        # self.assertIsNone(np.testing.assert_array_equal(self_cross_attn_expected_out.shape, self_cross_attn_dec.shape)) # compare shape with transformer
        # self.assertIsNone(np.testing.assert_array_almost_equal(self_cross_attn_expected_out, self_cross_attn_dec , decimal=4)) # compare value with transformer output
        print(np.mean(self_cross_attn_dec - expected), max(self_cross_attn_dec - expected), min(self_cross_attn_dec - expected))
        print("Dec Cross Attn 2 formulation == Dec Cross Attn 2 Trained TNN")
        
        
        # FFN 1 
        ffn1_enc = np.array(optimal_parameters["dec__ffn_2"])
        for x in list(layer_outputs_dict['model.decoder.layers.1.fc2']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        
        # self.assertIsNone(np.testing.assert_array_equal(ffn1_expected.shape, ffn1_enc.shape)) # compare shape with transformer
        # self.assertIsNone(np.testing.assert_array_almost_equal(ffn1_expected, ffn1_enc , decimal=6)) # compare value with transformer output
        print(np.mean(ffn1_enc - expected), max(ffn1_enc - expected), min(ffn1_enc - expected))
        print("Dec FFN2 formulation == Dec FNN2 Trained TNN") 
        
        # Norm 3
        norm3 = np.array(optimal_parameters["dec_norm_7"])
        for x in list(layer_outputs_dict['model.decoder.layers.1.final_layer_norm']):
            if np.array(x).shape[1] ==  8:   
                dec_expected = np.array(x)
        expected= dec_expected.flatten('C') 
        
        # self.assertIsNone(np.testing.assert_array_equal(ffn1_expected.shape, ffn1_enc.shape)) # compare shape with transformer
        # self.assertIsNone(np.testing.assert_array_almost_equal(ffn1_expected, ffn1_enc , decimal=6)) # compare value with transformer output
        print(np.mean(norm3 - expected), max(norm3 - expected), min(norm3 - expected))
        print("Dec Layer 2 Out formulation == Dec Layer 2 Out Trained TNN") 
        
        # linear 
        dec_linear_2 = np.array(optimal_parameters["linear_2"])
        dec_linear_2_expected = np.array(list(layer_outputs_dict['parameter_projection.proj.1'])).flatten('C')

        print(np.mean(dec_linear_2 - dec_linear_2_expected), max(dec_linear_2 - dec_linear_2_expected), min(dec_linear_2 - dec_linear_2_expected))
        print("Out = expected Out")
        
        #---------#
        print("linear 2, ", dec_linear_2)
        print(" linear 2 ex, ", np.array(list(layer_outputs_dict['parameter_projection.proj.1'])).shape, dec_linear_2_expected)
        print()
        print(" param proj., ", np.array(list(layer_outputs_dict['parameter_projection'])).shape, list(layer_outputs_dict['parameter_projection']))
        
        
        # embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
        # layer = "dec_linear_1"
        # W_linear = parameters[layer,'W']
        # b_linear = parameters[layer,'b'] 
        
        # print(np.array(W_linear).shape)
        
        # layer = layer
        # dec_in = transformer.embed_input( dec_in, layer, embed_dim, W_linear, b_linear)
        
if __name__ == '__main__': 
    # load model
    train_tnn_path = ".\\trained_transformer\\case_study\\model_TimeSeriesTransformer_final.pth"
    
    # Model Configuration
    device = "cpu"
    NUMBER_OF_POINTS = 8
    CONTEXT_LENGTH = 3
    data_files = ["T", "P", "CO", "CO2", "H2", "CH4", "CH3OH", "H2O", "N2"]
    config = TimeSeriesTransformerConfig(
        prediction_length=NUMBER_OF_POINTS,
    )
    tnn_model = TimeSeriesTransformerForPrediction(config).to(device)
    
    tnn_model = torch.load(train_tnn_path, weights_only=False, map_location=torch.device('cpu'))
    tnn_model.config.prediction_length = NUMBER_OF_POINTS
    tnn_model.config.context_length=3
    tnn_model.config.embedding_dimension=60
    tnn_model.config.scaling=False
    tnn_model.config.lags_sequence=[0]
    tnn_model.config.num_time_features=1
    tnn_model.config.input_size=len(data_files)
    tnn_model.config.num_parallel_samples=1
    
    # TNN inputs
    # src = torch.ones(1, 1, len(data_files)) #input 1 point
    # tgt = torch.ones(1,  NUMBER_OF_POINTS, len(data_files)) # predict 
    src = torch.tensor( [[509.6634891767956, 64.21623862058296, 0.0399364819697806, 0.0766080670882077, 0.3901792052712248, 0.1744754753168476, 0.0033989528410583, 0.0006347082807082, 0.025494621375979]]) # from training data
    tgt = torch.tensor([
    [5.09663489e+02, 6.42162386e+01, 3.99364820e-02, 7.66080671e-02, 
     3.90179205e-01, 1.74475475e-01, 3.39895284e-03, 6.34708281e-04, 
     2.54946214e-02],
    [5.11175953e+02, 6.42149682e+01, 4.06188982e-02, 7.44947816e-02, 
     3.85204181e-01, 1.74475475e-01, 4.82982207e-03, 2.74799373e-03, 
     2.54946214e-02],
    [5.12050906e+02, 6.42136987e+01, 4.10101851e-02, 7.33666916e-02, 
     3.82602485e-01, 1.74475475e-01, 5.56662522e-03, 3.87608378e-03, 
     2.54946214e-02],
    [5.12778785e+02, 6.42124293e+01, 4.12625481e-02, 7.25857268e-02, 
     3.80764317e-01, 1.74475475e-01, 6.09522701e-03, 4.65704858e-03, 
     2.54946214e-02],
    [5.13453716e+02, 6.42111600e+01, 4.14279575e-02, 7.19886372e-02, 
     3.79303867e-01, 1.74475475e-01, 6.52690722e-03, 5.25413818e-03, 
     2.54946214e-02],
    [5.14104735e+02, 6.42098904e+01, 4.15321707e-02, 7.15075037e-02, 
     3.78068892e-01, 1.74475475e-01, 6.90382753e-03, 5.73527168e-03, 
     2.54946214e-02],
    [5.14740686e+02, 6.42086205e+01, 4.15888022e-02, 7.11112024e-02, 
     3.76993252e-01, 1.74475475e-01, 7.24349728e-03, 6.13157298e-03, 
     2.54946214e-02],
    [5.15366642e+02, 6.42073502e+01, 4.16078956e-02, 7.07778554e-02, 
     3.76031397e-01, 1.74475475e-01, 7.55775096e-03, 6.46491999e-03, 
     2.54946214e-02]])

    L_t = 8.0               # [m] length of the reactor
    z = np.linspace(0, L_t, NUMBER_OF_POINTS).reshape(-1,1,1) / L_t
    z = torch.from_numpy(z).to(device).permute(1, 0, 2)

    past_time_features =  z[:, 0:1].repeat(src.size(0), CONTEXT_LENGTH, 1).to(device).float()#torch.zeros_like(torch.linspace(-1, 0, CONTEXT_LENGTH).reshape(1, -1, 1).repeat(x_batch.size(0), 1, 1)).to(device)
    future_time_features = z.repeat(src.size(0), 1, 1).to(device).float() #torch.zeros_like(y_batch[..., 0]).unsqueeze(-1).to(device)
    past_values = src.repeat(1, CONTEXT_LENGTH, 1).to(device)
    past_observed_mask = torch.zeros_like(past_values).to(device)
    past_observed_mask[:, -1:, :] = 1
    
    hugging_face_dict = {}
    hugging_face_dict["past_values"] =  past_values
    hugging_face_dict["past_time_features"] = past_time_features
    hugging_face_dict["past_observed_mask"] = past_observed_mask
    hugging_face_dict["future_time_features"] = future_time_features

    # instantiate pyomo model component
    opt_model = pyo.ConcreteModel(name="(Reactor_TNN)")
    
    src = src.repeat(CONTEXT_LENGTH,1)
    src = torch.nn.functional.pad( src, (0,28 - len(data_files)), "constant", 0)
    tgt = torch.nn.functional.pad( tgt, (0,28 - len(data_files)), "constant", 0)
    padding = list(range(28 - len(data_files)))
    dims = data_files + padding

    space =  np.linspace(0, L_t, NUMBER_OF_POINTS)/ L_t
    start_time = space[0] - (CONTEXT_LENGTH - 1) * (space[1]-space[0])
    enc_space = np.linspace(start_time, space[0], CONTEXT_LENGTH)
    opt_model.enc_space = pyo.Set(initialize=enc_space)
    opt_model.dec_space = pyo.Set(initialize=space)
    opt_model.dims = pyo.Set(initialize=dims) # states: ["T", "P", "CO", "CO2", "H2", "CH4", "CH3OH", "H2O", "N2"]
    
    states_max = [569.952065200784, 71.49265445971363, 0.0534738227626869, 0.0839279358015094, 0.4739118921128102, 0.1961240582176027, 0.043617617295987, 0.0166983631358979, 0.0286116689671041] + [0] * (28 - len(data_files))# from training data
    states_max_dict = {}
    for d , val in zip(opt_model.dims, states_max):
        states_max_dict[d] = val
    opt_model.states_max = pyo.Param(opt_model.dims, initialize = states_max_dict)
    
    states_min  = [466.35539818346194, 57.31174829828023, 0.0172916368293674, 0.0552752589680291, 0.3095623691919211, 0.1604881777757451, 0.0028584153155807, 0.0006125105511711, 0.0234112567627298] + [0] * (28 - len(data_files))
    states_min_dict = {}
    for d , val in zip(opt_model.dims, states_min):
        states_min_dict[d] = val
    opt_model.states_min = pyo.Param(opt_model.dims, initialize = states_min_dict)
    
    # state var
    opt_model.x = pyo.Var(opt_model.dec_space, opt_model.dims) # state vars
    opt_model.x_enc = pyo.Var(opt_model.enc_space, opt_model.dims)

    # CO outlet  constraint
    opt_model.x[opt_model.dec_space.last(), "CO"].ub = 0.02
    
    # Temperature inlet constraints
    opt_model.x[opt_model.dec_space.first(), "T"].ub = 550
    opt_model.x[opt_model.dec_space.first(), "T"].lb = 450

    # Pressure inlet constraints
    opt_model.x[opt_model.dec_space.first(), "P"].ub = 68
    opt_model.x[opt_model.dec_space.first(), "P"].lb = 62

    # x bounds
    for s in opt_model.dec_space:
        for d, dim in enumerate(opt_model.dims):
            opt_model.x[s,dim].ub = 1.50* opt_model.states_max[dim]
            opt_model.x[s,dim].lb = 0.5 * opt_model.states_min[dim] 
    
    # x encoder constraints
    opt_model.x_enc_constraints = pyo.ConstraintList()
    for s in opt_model.enc_space:
        for dim in opt_model.dims:
            opt_model.x_enc_constraints.add(expr= opt_model.x_enc[s,dim] == opt_model.x[opt_model.dec_space.first(), dim])

    layer_names, parameters, _, enc_dec_count, layer_outputs_dict = extract_from_pretrained.get_hugging_learned_parameters(tnn_model, src , tgt, 2, hugging_face_dict)
     
    # __________________RMOVE_______________________##
    # fix inputs
    input = tgt.numpy()
    opt_model.x_fixed_constraints = pyo.ConstraintList()
    for s, space in enumerate(opt_model.dec_space):
        for d, dim in enumerate(opt_model.dims):
            if space < opt_model.dec_space.last():
                opt_model.x_fixed_constraints.add(expr= opt_model.x[space,dim] == input[s,d])
                
                
    # Set enhancement configuration
    ACTI_LIST_FULL = [
            "LN_var", "LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum",
             "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight",
            "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC", "MHA_attn_score", "MHA_output", 
            "RES_var", "MHA_softmax_env", "AVG_POOL_var", "embed_var"]

    activation_dict = {}
    for key in ACTI_LIST_FULL:
        activation_dict[key] = False

    ACTI = {}  
    ACTI["LN_I"] = {"list": ["LN_var"]}
    ACTI["LN_D"] = {"list": ["LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum"]}

    ACTI["MHA_I"] = {"list": ["MHA_attn_weight_sum", "MHA_attn_weight"]}
    ACTI["MHA_D"] = {"list": ["MHA_Q", "MHA_K", "MHA_V", "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_attn_score", "MHA_output" , "RES_var"]}
    ACTI["MHA_MC"] = {"list":[ "MHA_QK_MC", "MHA_WK_MC"]}

    #ACTI["RES_ALL"] = {"list":[ "RES_var"]}

    ACTI_Groups = 6
    LN_ALL = ACTI["LN_I"]["list"] + ACTI["LN_D"]["list"]
    MHA_ALL = ACTI["MHA_I"]["list"]  + ACTI["MHA_D"]["list"] + ACTI["MHA_MC"]["list"]
    #check all bounds and constraints in the lists
    #assert(len(ACTI["RES_ALL"]["list"])+ len(MHA_ALL)+ len(LN_ALL) == len(ACTI_LIST)) 


    combinations = [
        # [1 , 0, 1, 1, 1], #1
        # [1 , 0, 1, 1, 0], #2 -- best initial feas, no optimal
        # [1 , 0, 1, 0, 0], #3 --2nd best initial feas, relatively fast opt
        # [1 , 0, 0, 0, 0], #4
        # [1 , 0, 0, 1, 1], #5 -- 3rd best initial feas, slower opt
        1 , 0, 0, 1, 0 #6 -- slow initial feas, fastest solving
        #[0 , 0, 0, 0, 0],  #7
    ]
    combinations = [bool(val) for val  in combinations]

    ACTI["LN_I"]["act_val"], ACTI["LN_D"]["act_val"], ACTI["MHA_I"]["act_val"] , ACTI["MHA_D"]["act_val"], ACTI["MHA_MC"]["act_val"] = combinations

    for k, val in ACTI.items():
        for elem in val["list"]:
            activation_dict[elem] = val["act_val"] # set activation dict to new combi

    unittest.main() 
    # # instantiate transformer
    # transformer = TNN.Transformer( ".\\data\\reactor_config_huggingface.json", opt_model) 
    
    # # build optimization TNN
    # result =  transformer.build_from_hug_torch( tnn_model,sample_enc_input=src, sample_dec_input=src,enc_bounds = bounds_target , dec_bounds=bounds_target, Transformer='huggingface', default=False, hugging_face_dict=hugging_face_dict)
    # tnn_input_enc = getattr( opt_model, result[0][0])
    # tnn_input_dec = getattr( opt_model, result[0][1])
    # tnn_output = getattr( opt_model, result[-2])
    # ffn_parameter_dict = result[-1]
    
    # # Add constraints to TNN encoder input
    # opt_model.tnn_input_constraints = pyo.ConstraintList()
    # indices = []
    # for set in str(tnn_input_enc.index_set()).split("*"): # get TNN enc input index sets
    #     indices.append( getattr(opt_model, set) )
    # for tnn_index, index in zip(indices[0], opt_model.enc_space):
    #     for tnn_dim, dim in zip(indices[1], opt_model.dims):
    #         opt_model.tnn_input_constraints.add(expr= tnn_input_enc[tnn_index, tnn_dim]== opt_model.x[index, dim])
            
    # # Add constraints to TNN decoder input
    # indices = []
    # for set in str(tnn_input_dec.index_set()).split("*"):# get TNN dec input index sets
    #     indices.append( getattr(opt_model, set) )
    # for tnn_index, index in zip(indices[0], opt_model.dec_space):
    #     for tnn_dim, dim in zip(indices[1], opt_model.dims):
    #         opt_model.tnn_input_constraints.add(expr= tnn_input_dec[tnn_index, tnn_dim]== opt_model.x[index, dim])
    
    # # Add constraints to TNN output
    # for set in str(tnn_output.index_set()).split("*"):# get TNN ouput index sets
    #     indices.append( getattr(opt_model, set) )
    # for tnn_index, index in zip(indices[0], opt_model.dec_space):
    #     for tnn_dim, dim in zip(indices[1], opt_model.dims):
    #         opt_model.tnn_input_constraints.add(expr= tnn_output[tnn_index, tnn_dim]== opt_model.x[index, dim])
                    
    
    # # Convert to gurobi
    # gurobi_model, map_var , _ = convert_pyomo.to_gurobi(opt_model)
        
    # # Add FNNS to gurobi model
    # for key, value in ffn_parameter_dict.items():
    #     print(key, value)
    #     nn, input_nn, output_nn = value
    #     input, output = get_inputs_gurobipy_FFN(input_nn, output_nn, map_var)
    #     pred_constr = add_predictor_constr(gurobi_model, nn, input, output)
    
    # gurobi_model.update() # update gurobi model with FFN constraints
    
    # # Optimize
    # gurobi_model.optimize()

    # if gurobi_model.status == GRB.OPTIMAL:
    #     optimal_parameters = {}
    #     for v in gurobi_model.getVars():
    #         #print(f'var name: {v.varName}, var type {type(v)}')
    #         if "[" in v.varName:
    #             name = v.varname.split("[")[0]
    #             if name in optimal_parameters.keys():
    #                 optimal_parameters[name] += [v.x]
    #             else:
    #                 optimal_parameters[name] = [v.x]
    #         else:    
    #             optimal_parameters[v.varName] = v.x
                
    # if gurobi_model.status == GRB.INFEASIBLE:
    #     gurobi_model.computeIIS()
    #     gurobi_model.write("pytorch_model.ilp")
        
    # states = norm4_dec = np.array(optimal_parameters["x"])
    # print(data_files)
    # print("prediction", states)
    # print("expected", layer_outputs_dict['parameter_projection.proj.2'])