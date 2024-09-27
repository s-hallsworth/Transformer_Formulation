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
from gurobipy import Model, GRB
#from gurobi_ml import add_predictor_constr
from gurobi_machinelearning.src.gurobi_ml.add_predictor import add_predictor_constr

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

# Import from repo file
import transformer_b_flag_cuts as TNN
from helpers.GUROBI_ML_helper import get_inputs_gurobipy_FNN
from helpers.print_stats import solve_pyomo, solve_gurobipy
import helpers.convert_pyomo as convert_pyomo
from trained_transformer.Tmodel import TransformerModel
import helpers.extract_from_pretrained as extract_from_pretrained
from transformers.src.transformers.models.time_series_transformer.configuration_time_series_transformer import TimeSeriesTransformerConfig
from transformers.src.transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerForPrediction
# cloned transformers from: https://github.com/s-hallsworth/transformers.git
from gurobi_machinelearning.src.gurobi_ml.add_predictor import add_predictor_constr

"""
Test each module of reactor TNN
"""
class TestTransformer(unittest.TestCase):
    def test_TNN(self):
        m = opt_model.clone()
        
        transformer = TNN.Transformer( ".\\data\\reactor_config_huggingface.json", m) 
        
        enc_dim_1 = src.size(0)
        dec_dim_1 = tgt.size(0)
        transformer.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
        transformer.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
        transformer.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) 
        transformer.M.model_dims = pyo.Set(initialize= list(range(transformer.d_model)))
        transformer.M.input_dims = pyo.Set(initialize= list(range(transformer.input_dim)))
    
        bounds_target = (-1,1)
        # Add TNN input vars
        transformer.M.enc_input = pyo.Var(transformer.M.enc_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        transformer.M.dec_input = pyo.Var(transformer.M.dec_time_dims,  transformer.M.input_dims, bounds=bounds_target)
        
        # Add constraints to TNN encoder input
        m.tnn_input_constraints = pyo.ConstraintList()
        indices = []
        for set in str(transformer.M.enc_input.index_set()).split("*"): # get TNN enc input index sets
            indices.append( getattr(m, set) )
            print(set)
            
        print(indices)
        print(len(indices[0]), len(m.enc_space))
        print(len(indices[1]), len(m.dims))
        for tnn_index, index in zip(indices[0], m.enc_space):
            for tnn_dim, dim in zip(indices[1], m.dims):
                print(tnn_index, index)
                print(tnn_dim, dim)
                m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, tnn_dim] == m.x_scaled[index, dim])
                
        # Add constraints to TNN decoder input
        indices = []
        for set in str(transformer.M.dec_input.index_set()).split("*"):# get TNN dec input index sets
            indices.append( getattr(m, set) )
        for tnn_index, index in zip(indices[0], m.dec_space):
            for tnn_dim, dim in zip(indices[1], m.dims):
                m.tnn_input_constraints.add(expr= transformer.M.dec_input[tnn_index, tnn_dim]== m.x_scaled[index, dim])
                
                
        # Set objective: maximise amount of methanol at reactor outlet
        m.obj = pyo.Objective(
                expr = m.x[m.dec_space.last(), "CH3OH"], sense=-1
            )  # -1: maximize, +1: minimize (default)
        
        # Convert to gurobi
        gurobi_model, map_var , _ = convert_pyomo.to_gurobi(m)
            
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
            
        # model input
        enc_input_name = "enc_input"
        dec_input_name = "dec_input"
        print(optimal_parameters.keys())
        model_enc_input = np.array(optimal_parameters[enc_input_name])
        model_dec_input = np.array(optimal_parameters[dec_input_name])
        
        expected_enc_input = (src.numpy() - np.array(states_min)) / ( np.array(states_max) - np.array(states_min))
        expected_dec_input = (tgt.numpy() - np.array(states_min)) / ( np.array(states_max) - np.array(states_min))
        
        model_tgt = np.array(optimal_parameters["x"])
        model_tgt_scaled = np.array(optimal_parameters["x_scaled"])
        expected_tgt = tgt.numpy().flatten()
        
        # Assertions
        self.assertIsNone(np.testing.assert_array_equal(model_tgt.shape, expected_tgt.shape)) 
        self.assertIsNone(np.testing.assert_array_equal(model_tgt, expected_tgt)) 
        print("input tgt = expected input tgt")
        
        self.assertIsNone(np.testing.assert_array_equal(model_tgt_scaled.shape, expected_dec_input.flatten().shape)) 
        self.assertIsNone(np.testing.assert_array_almost_equal(model_tgt_scaled, expected_dec_input.flatten(), decimal = 7))
        print("input tgt scaled  = expected input tgt scaled")
        
        self.assertIsNone(np.testing.assert_array_equal(model_dec_input.shape, expected_dec_input.flatten().shape)) # pyomo input data and transformer input data must be the same shape
        self.assertIsNone(np.testing.assert_array_almost_equal(model_dec_input, expected_dec_input.flatten(), decimal = 7))             # both inputs must be equal
        print("input dectnn = expected dec input tnn")
        
        self.assertIsNone(np.testing.assert_array_equal(model_enc_input.shape, expected_enc_input.flatten().shape)) # pyomo input data and transformer input data must be the same shape
        self.assertIsNone(np.testing.assert_array_almost_equal(model_enc_input, expected_enc_input.flatten(), decimal = 7))             # both inputs must be equal
        
        
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
    
    space =  np.linspace(0, L_t, NUMBER_OF_POINTS)/ L_t
    opt_model.enc_space = pyo.Set(initialize=[space[0]])
    opt_model.dec_space = pyo.Set(initialize=space)
    opt_model.dims = pyo.Set(initialize=data_files) # states: ["T", "P", "CO", "CO2", "H2", "CH4", "CH3OH", "H2O", "N2"]
    
    states_max = [569.952065200784, 71.49265445971363, 0.0534738227626869, 0.0839279358015094, 0.4739118921128102, 0.1961240582176027, 0.043617617295987, 0.0166983631358979, 0.0286116689671041] # from training data
    states_max_dict = {}
    for d , val in zip(opt_model.dims, states_max):
        states_max_dict[d] = val
    opt_model.states_max = pyo.Param(opt_model.dims, initialize = states_max_dict)
    
    states_min  = [466.35539818346194, 57.31174829828023, 0.0172916368293674, 0.0552752589680291, 0.3095623691919211, 0.1604881777757451, 0.0028584153155807, 0.0006125105511711, 0.0234112567627298]
    states_min_dict = {}
    for d , val in zip(opt_model.dims, states_min):
        states_min_dict[d] = val
    opt_model.states_min = pyo.Param(opt_model.dims, initialize = states_min_dict)
    
    # state var
    opt_model.x = pyo.Var(opt_model.dec_space, opt_model.dims) # state vars
    opt_model.x_scaled = pyo.Var(opt_model.dec_space, opt_model.dims, bounds=(-1,1)) # min-max sclaed state variables
    
    # CO outlet  constraint
    opt_model.x[opt_model.dec_space.last(), "CO"].ub = 0.02
    # opt_model.CO_constr = pyo.Constraint(expr = opt_model.x[opt_model.dec_space.last(), "CO"] <= 0.02) 
    
    # Temperature inlet constraints
    opt_model.x[opt_model.dec_space.first(), "T"].ub = 550
    opt_model.x[opt_model.dec_space.first(), "T"].lb = 450
    # opt_model.T_in_ub_constr = pyo.Constraint(expr = opt_model.x[opt_model.dec_space.first(), "T"] <= 550) 
    # opt_model.T_in_lb_constr = pyo.Constraint(expr = opt_model.x[opt_model.dec_space.first(), "T"] >= 450) 
    
    # Pressure inlet constraints
    opt_model.x[opt_model.dec_space.first(), "P"].ub = 68
    opt_model.x[opt_model.dec_space.first(), "P"].lb = 62
    # opt_model.P_in_ub_constr = pyo.Constraint(expr = opt_model.x[opt_model.dec_space.first(), "P"] <= 68) 
    # opt_model.P_in_lb_constr = pyo.Constraint(expr = opt_model.x[opt_model.dec_space.first(), "P"] >= 62) 
    
    # x_scaled constraints
    opt_model.x_scaled_constraints = pyo.ConstraintList()
    for s in opt_model.dec_space:
        for dim in opt_model.dims:
            opt_model.x_scaled_constraints.add(expr= opt_model.x_scaled[s,dim] == (opt_model.x[s,dim] - opt_model.states_min[dim])/ (opt_model.states_max[dim] - opt_model.states_min[dim]))
            opt_model.x[s,dim].ub = 1.5 * opt_model.states_max[dim]
            opt_model.x[s,dim].lb = 0.5 * opt_model.states_min[dim] 
           
           
    layer_names, parameters, _, enc_dec_count, layer_outputs_dict = extract_from_pretrained.get_hugging_learned_parameters(tnn_model, src , tgt, 2, hugging_face_dict)
     
    # __________________RMOVE_______________________##
    # fix inputs
    input = tgt.numpy()
    print(tgt.shape, src.shape)
    opt_model.x_fixed_constraints = pyo.ConstraintList()
    for s, space in enumerate(opt_model.dec_space):
        for d, dim in enumerate(opt_model.dims):
            opt_model.x_fixed_constraints.add(expr= opt_model.x[space,dim] == input[s,d])

    unittest.main() 
    # # instantiate transformer
    # transformer = TNN.Transformer( ".\\data\\reactor_config_huggingface.json", opt_model) 
    
    # # build optimization TNN
    # result =  transformer.build_from_pytorch( tnn_model,sample_enc_input=src, sample_dec_input=src,enc_bounds = bounds_target , dec_bounds=bounds_target, Transformer='huggingface', default=False, hugging_face_dict=hugging_face_dict)
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
    #         opt_model.tnn_input_constraints.add(expr= tnn_input_enc[tnn_index, tnn_dim]== opt_model.x_scaled[index, dim])
            
    # # Add constraints to TNN decoder input
    # indices = []
    # for set in str(tnn_input_dec.index_set()).split("*"):# get TNN dec input index sets
    #     indices.append( getattr(opt_model, set) )
    # for tnn_index, index in zip(indices[0], opt_model.dec_space):
    #     for tnn_dim, dim in zip(indices[1], opt_model.dims):
    #         opt_model.tnn_input_constraints.add(expr= tnn_input_dec[tnn_index, tnn_dim]== opt_model.x_scaled[index, dim])
    
    # # Add constraints to TNN output
    # for set in str(tnn_output.index_set()).split("*"):# get TNN ouput index sets
    #     indices.append( getattr(opt_model, set) )
    # for tnn_index, index in zip(indices[0], opt_model.dec_space):
    #     for tnn_dim, dim in zip(indices[1], opt_model.dims):
    #         opt_model.tnn_input_constraints.add(expr= tnn_output[tnn_index, tnn_dim]== opt_model.x_scaled[index, dim])
                    
    
    # # Convert to gurobi
    # gurobi_model, map_var , _ = convert_pyomo.to_gurobi(opt_model)
        
    # # Add FNNS to gurobi model
    # for key, value in ffn_parameter_dict.items():
    #     print(key, value)
    #     nn, input_nn, output_nn = value
    #     input, output = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
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