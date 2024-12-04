# External imports
import pyomo.environ as pyo
import numpy as np
import os
import torch

# Import from repo file
import transformer_b_flag as TNN
import MINLP_tnn.helpers.extract_from_pretrained as extract_from_pretrained
import transformers, sys
sys.modules['transformers.src.transformers'] = transformers
from transformers.models.time_series_transformer.configuration_time_series_transformer import TimeSeriesTransformerConfig
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerForPrediction
# cloned transformers from: https://github.com/s-hallsworth/transformers.git
from gurobi_machinelearning.src.gurobi_ml.add_predictor import add_predictor_constr
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

"""
Setup for reactor case study experiments
"""

def reactor_problem(train_tnn_path):
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
    src = torch.ones(1, len(data_files)) #dummy input
    tgt = torch.ones(1,  NUMBER_OF_POINTS, len(data_files)) #dummy input 
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
    
    states_min = [466.35539818346194, 57.31174829828023, 0.0172916368293674, 0.0552752589680291, 0.3095623691919211, 0.1604881777757451, 0.0028584153155807, 0.0006125105511711, 0.0234112567627298] + [0] * (28 - len(data_files)) # from training data
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
    opt_model.x[opt_model.dec_space.first(), "T"].lb = 450 # outside of range

    # Pressure inlet constraints
    opt_model.x[opt_model.dec_space.first(), "P"].ub = 68
    opt_model.x[opt_model.dec_space.first(), "P"].lb = 62

    # x bounds
    for s in opt_model.dec_space:
        for d, dim in enumerate(opt_model.dims):

            if not opt_model.x[s,dim].lb is None:
                opt_model.x[s,dim].lb = max(opt_model.x[s,dim].lb, opt_model.states_min[dim]) #lower bound is min from training data
            else:
                 opt_model.x[s,dim].lb = opt_model.states_min[dim]
                 
            if not opt_model.x[s,dim].ub is None:
                opt_model.x[s,dim].ub = min(opt_model.x[s,dim].ub, opt_model.states_max[dim]) #lower bound is min from training data
            else:
                 opt_model.x[s,dim].ub =  opt_model.states_max[dim] #upper bound is max from training data
            
            print(s, dim, opt_model.x[s,dim].lb, opt_model.x[s,dim].ub)
    
    # x encoder constraints
    opt_model.x_enc_constraints = pyo.ConstraintList()
    for s in opt_model.enc_space:
        for dim in opt_model.dims:
            opt_model.x_enc_constraints.add(expr= opt_model.x_enc[s,dim] == opt_model.x[opt_model.dec_space.first(), dim])

    layer_names, parameters, _, enc_dec_count, layer_outputs_dict = extract_from_pretrained.get_hugging_learned_parameters(tnn_model, src , tgt, 2, hugging_face_dict)
    
    # Set objective: maximise amount of methanol at reactor outlet
    opt_model.obj = pyo.Objective(
            expr = opt_model.x[opt_model.dec_space.last(), "CH3OH"], sense=-1
        )  # -1: maximize, +1: minimize (default)

    return opt_model, parameters,layer_outputs_dict, src, tgt

def reactor_tnn(opt_model, parameters,layer_outputs_dict,activation_dict, config, src, tgt):
    m = opt_model.clone()
        
    transformer = TNN.Transformer(config, m, activation_dict) 
    
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
        
    transformer.add_attention( "enc_norm_1", layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, norm_softmax=True)
    
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
    transformer.add_attention( "enc_norm_3", layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, norm_softmax=True)
    
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
    dec_dim_1 = tgt.size(1)
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
    
    dec_in = transformer.add_attention( dec_in, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, mask=True, norm_softmax=True)
    
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
    dec_in = transformer.add_attention( dec_in, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, cross_attn=True, encoder_output="enc_norm_5", norm_softmax=True)

    
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

    dec_in = transformer.add_attention( dec_in, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, mask=True, norm_softmax=True)
    
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
    dec_in = transformer.add_attention( dec_in, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, cross_attn=True, encoder_output="enc_norm_5", norm_softmax=True)

    
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
    
    # link decoder output to input
    for tnn_index, index in zip(indices[0], m.dec_space):
        for d, tnn_dim in enumerate(embed_dim):
            dim = m.dims.at(d+1)
            m.tnn_input_constraints.add(expr= out2[tnn_index, tnn_dim] * (opt_model.states_max[dim] - opt_model.states_min[dim] ) == (m.x[index, dim]- opt_model.states_min[dim] ) )
    
    return m, ffn_parameter_dict, layer_outputs_dict, transformer