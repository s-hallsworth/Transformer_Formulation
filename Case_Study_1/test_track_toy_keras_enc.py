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

    def test_FFN2(self):
        print("======= FFN2 =======")
        
        # Define Test Case Params
        m = model.clone()
        seq_len = 10
        layer = 'mutli_head_attention_1'
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
        
        ffn_1_params = parameters['ffn_1']
        parameters['ffn_2']  = {'input_shape':ffn_1_params['input_shape']}#, "input": ffn_1_params['input']}
        parameters['ffn_2'] |= {'dense_2':  ffn_1_params['dense_2']}
        parameters['ffn_2'] |= {'dense_3':ffn_1_params['dense_3']}
        
        parameters['ffn_1']  = {'input_shape': ffn_1_params['input_shape']}
        parameters['ffn_1'] |= {'dense': ffn_1_params['dense']}
        parameters['ffn_1'] |= {'dense_1': ffn_1_params['dense_1']}
        
        
        # Define tranformer and execute 
        transformer = TNN.Transformer(config_file, m)  
        transformer.add_input_var("input_embed", dims=(seq_len, transformer.input_dim), bounds=(-3,3))
        transformer.add_layer_norm( "input_embed", "layer_norm", gamma1, beta1)
        transformer.add_attention( "layer_norm","attention_output", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        transformer.add_residual_connection("input_embed", "attention_output", "residual_1")
        transformer.add_layer_norm( "residual_1", "layer_norm_2", gamma2, beta2)
        nn, input_nn, output_nn = transformer.get_fnn("layer_norm_2", "ffn_1", "ffn_1", (seq_len,2), parameters)
        transformer.add_residual_connection("residual_1", "ffn_1", "residual_2")  
        transformer.add_avg_pool( "residual_2", "avg_pool")
        nn2, input_nn2, output_nn2 = transformer.get_fnn( "residual_2", "ffn_2", "ffn_2", (pred_len+1, 2), parameters)
        
        
        # add constraints to trained TNN input
        m.tnn_constraints = pyo.ConstraintList()
        indices = []
        for set in str(transformer.M.input_embed.index_set()).split("*"):
            indices.append( getattr(m, set) )
        for tnn_index, index in zip(indices[0], m.time_history):
            m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].first()]== m.history_loc1[index])
            m.tnn_constraints.add(expr= transformer.M.input_embed[tnn_index, indices[1].last()] == m.history_loc2[index]) 
            
        # add constraints to trained TNN output
        m.tnn_constraints = pyo.ConstraintList()
        indices = []
        for set in str(output_nn2.index_set()).split("*"): 
            indices.append( getattr(m, set) )
        out_index = 0
        for t_index, t in enumerate(m.time):
            index = t_index + 1 # 1 indexing
            
            if t >= m.time_history.last():
                out_index += 1
                print(out_index, t )
                m.tnn_constraints.add(expr= output_nn2[indices[0].at(out_index), indices[1].first()] == m.loc1[t])
                m.tnn_constraints.add(expr= output_nn2[indices[0].at(out_index), indices[1].last()]  == m.loc2[t])
        
        # Set objective
        m.obj = pyo.Objective(
            expr= sum((m.x1[t] - m.loc1[t])**2 + (m.x2[t] - m.loc2[t])**2 for t in m.time), sense=1
        )  # -1: maximize, +1: minimize (default)
            

        # # Convert to gurobipy
        gurobi_model, map_var, _ = convert_pyomo.to_gurobi(m)

        ## Add FNN1 to gurobi model
        input_1, output_1 = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
        pred_constr1 = add_predictor_constr(gurobi_model, nn, input_1, output_1)
        
        inputs_2, outputs_2 = get_inputs_gurobipy_FNN(input_nn2, output_nn2, map_var)
        pred_constr2 = add_predictor_constr(gurobi_model, nn2, inputs_2, outputs_2)
        gurobi_model.update()
        #pred_constr.print_stats()
        
        ## Optimizes
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
                    
        ## Check MHA output
        attention_output= np.array(optimal_parameters["attention_output"]) 
        MHA_output = np.array(layer_outputs_dict["multi_head_attention_1"]).flatten()
        self.assertIsNone(np.testing.assert_array_equal(attention_output.shape, MHA_output.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(attention_output, MHA_output , decimal=5)) # compare value with transformer output
        print("- MHA output formulation == MHA output model")
                    
        
        #Check outputs
        ffn_2_output = np.array([optimal_parameters["ffn_2"][0]])
        FFN_out = np.array(layer_outputs_dict["dense_4"])[0]
        print(ffn_2_output.shape, FFN_out.shape )
        print(ffn_2_output,  FFN_out)
        self.assertIsNone(np.testing.assert_array_equal(ffn_2_output.shape,  FFN_out.shape)) # compare shape with transformer
        self.assertIsNone(np.testing.assert_array_almost_equal(ffn_2_output,  FFN_out, decimal=5)) # compare value with transformer output
        print("- FFN2 output formulation == FFN2 output model")   
        
        print("Output: ", ffn_2_output)
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
    config_file = '.\\data\\toy_track_k_enc_config.json' 
    
    # define constants
    T_end = 0.0105
    steps = 19 ##CHANGE THIS ##
    time = np.linspace(0, T_end, num=steps)
    dt = time[1] - time[0]
    tt = 10 # sequence size
    time_history = time[0:tt]
    pred_len = 9

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
        return v_l1 * t
    model.history_loc1 = pyo.Param(model.time_history, rule=target_location_rule) 

    def target_location2_rule(M, t):
        return (v_l2*t) - (0.5 * g * (t**2))
    model.history_loc2 = pyo.Param(model.time_history, rule=target_location2_rule) 
    
    history_loc1 = np.array([v for k,v in model.history_loc1.items()])
    history_loc2 = np.array([v for k,v in model.history_loc2.items()])
    # print(history_loc1, history_loc2)

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

    # Fix model solution
    input_x1 =   v_l1 * time  
    input_x2 =  (v_l2*time) - (0.5 * g * (time*time))
    
    model.fixed_loc_constraints = pyo.ConstraintList()
    for i,t in enumerate(model.time):
        model.fixed_loc_constraints.add(expr= input_x1[i] == model.loc1[t])
        model.fixed_loc_constraints.add(expr= input_x2[i]  == model.loc2[t])


    # load trained transformer
    model_path = ".\\trained_transformer\\TNN_traj_enc.keras" # dmodel 4, num heads 1, n ence 1, n dec 1, head dim 4, pred_len 2+1 
    layer_names, parameters ,_ = extract_from_pretrained.get_learned_parameters(model_path)
    
    # get intermediate results dictionary for optimal input values
    input = np.array([[ [x1,x2] for x1,x2 in zip(input_x1, input_x2)]], dtype=np.float32)
    layer_outputs_dict = extract_from_pretrained.get_intermediate_values(model_path, input[:, 0:tt, :])

    
    # unit test
    unittest.main() 

