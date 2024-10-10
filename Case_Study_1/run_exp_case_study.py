import pyomo.environ as pyo
import numpy as np
import os
import torch
from gurobipy import  GRB
# from gurobi_ml import add_predictor_constr
from gurobi_machinelearning.src.gurobi_ml.add_predictor import add_predictor_constr

# Import from repo files
import transformer_b_flag as TNN
import helpers.extract_from_pretrained as extract_from_pretrained
from helpers.print_stats import save_gurobi_results
import helpers.convert_pyomo as convert_pyomo
from helpers.combine_csv import combine
from helpers.GUROBI_ML_helper import get_inputs_gurobipy_FNN
import transformers, sys
sys.modules['transformers.src'] = transformers
sys.modules['transformers.src.transformers'] = transformers
from transformers.models.time_series_transformer.configuration_time_series_transformer import TimeSeriesTransformerConfig
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerForPrediction
# cloned transformers from: https://github.com/s-hallsworth/transformers.git
# cd transformers 
# pip install -e .

# turn off floating-point round-off
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' 

# Set Up
TESTING = False # fix TNN input for testing (faster solve)
combine_files = not TESTING
TL = 60 # time limit
REP = 1 # number of repetitions of each scenario
NAME = "reactor_cs"
SOLVER = "gurobi"
FRAMEWORK = "gurobipy"
exp_name = "Reactor"
config_file = ".\\data\\reactor_config_huggingface.json" 
model_path = ".\\trained_transformer\\case_study\\model_TimeSeriesTransformer_final.pth"
PATH =  f".\\Experiments\\{exp_name}"+"\\"

# Store Summary TNN Architecture info
tnn_config = {}
tnn_config["Num Enc"] = 2
tnn_config["Num Dec"] = 2
tnn_config["Num Res"] = 10
tnn_config["Num LN"]  = 12
tnn_config["Num AVGP"] = 0
tnn_config["Num Dense"] = 15
tnn_config["Num ReLu"] = 0
tnn_config["Num SiLU"] = 4
tnn_config["Num Attn"] = 6
tnn_config["Config File"] = config_file

# Model Configuration
device = "cpu"
NUMBER_OF_POINTS = 8
CONTEXT_LENGTH = 3
data_files = ["T", "P", "CO", "CO2", "H2", "CH4", "CH3OH", "H2O", "N2"]
config = TimeSeriesTransformerConfig(
    prediction_length=NUMBER_OF_POINTS,
)
tnn_model = TimeSeriesTransformerForPrediction(config).to(device)
tnn_model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
tnn_model.config.prediction_length = NUMBER_OF_POINTS
tnn_model.config.context_length=3
tnn_model.config.embedding_dimension=60
tnn_model.config.scaling=False
tnn_model.config.lags_sequence=[0]
tnn_model.config.num_time_features=1
tnn_model.config.input_size=len(data_files)
tnn_model.config.num_parallel_samples=1

# Set values of hugging face dict for hook
L_t = 8.0               # [m] length of the reactor
z = np.linspace(0, L_t, NUMBER_OF_POINTS).reshape(-1,1,1) / L_t
z = torch.from_numpy(z).to(device).permute(1, 0, 2)

src = torch.ones(1, 1, len(data_files)) # supply random smaple input
tgt = torch.ones(1,  NUMBER_OF_POINTS, len(data_files)) # supply random smaple input
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

src = src.repeat(1, CONTEXT_LENGTH,1)
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
        opt_model.x[s,dim].ub = 1.5 * opt_model.states_max[dim]
        opt_model.x[s,dim].lb = 0.5 * opt_model.states_min[dim] 

# x encoder constraints
opt_model.x_enc_constraints = pyo.ConstraintList()
for s in opt_model.enc_space:
    for dim in opt_model.dims:
        opt_model.x_enc_constraints.add(expr= opt_model.x_enc[s,dim] == opt_model.x[opt_model.dec_space.first(), dim])

layer_names, parameters, _, enc_dec_count, layer_outputs_dict = extract_from_pretrained.get_hugging_learned_parameters(tnn_model, src , tgt, 2, hugging_face_dict)
    


# ##------ Fix model solution for TESTING ------##
if TESTING:
    REP = 1
    

# ## --------------------------------##


# Initially all constraints deactivated
ACTI_LIST = [
            "LN_var", "LN_mean", "LN_num", "LN_num_squ", "LN_denom", "LN_num_squ_sum",
             "MHA_Q", "MHA_K", "MHA_V", "MHA_attn_weight_sum", "MHA_attn_weight",
            "MHA_compat", "MHA_compat_exp", "MHA_compat_exp_sum", "MHA_QK_MC", "MHA_WK_MC", "MHA_attn_score", "MHA_output", 
            "RES_var"] #names of bounds and cuts to activate
                # "MHA_softmax_env"<- removed from list: should be dynamic
                # "AVG_POOL_var" <- no avg pool
                #  "embed_var" <- no embed
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
ACTI["LN_D"] = {"list": ["LN_num", "LN_num_squ", "LN_denom"]}

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
    #[1 , 0, 1, 1, 1], #1
    [1 , 0, 1, 1, 0], #2 -- fastest feasible soln
    [1 , 0, 1, 0, 0], #3 -- good trade off fast feasible and solve time
    #[1 , 0, 0, 0, 0], #4 -- smallest opt gap
    #[1 , 0, 0, 1, 1], #5
    [1 , 0, 0, 1, 0], #6 -- fastest convergance
    #[0 , 0, 0, 0, 0]  #7
]
combinations = [[bool(val) for val in sublist] for sublist in combinations]

# for each experiment repetition
for r in range(REP):
        
    for c, combi in enumerate(combinations):# for each combination of constraints/bounds
        experiment_name = f"{exp_name}_r{r+1}_c{c+1}"
        # activate constraints
        ACTI["LN_I"]["act_val"], ACTI["LN_D"]["act_val"], ACTI["MHA_I"]["act_val"] , ACTI["MHA_D"]["act_val"], ACTI["MHA_MC"]["act_val"] = combi

        for k, val in ACTI.items():
            for elem in val["list"]:
                activation_dict[elem] = val["act_val"] # set activation dict to new combi
        tnn_config["Activated Bounds/Cuts"] = activation_dict # save act config
        print(activation_dict)
        # clone optimization model
        m = opt_model.clone()
    
        #init and activate constraints
        transformer = TNN.Transformer(config_file, m, activation_dict)  
        
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
                m.tnn_input_constraints.add(expr= transformer.M.enc_input[tnn_index, tnn_dim] == m.x_enc[index, dim])
                
        # ADD ENCODER COMPONENTS
        # Add Linear transform
        # Linear transform
        embed_dim = transformer.M.model_dims # embed from current dim to self.M.model_dims
        layer = "enc_linear_1"
        W_linear = parameters[layer,'W']
        b_linear = parameters[layer,'b'] 
        transformer.embed_input( "enc_input", layer, embed_dim, W_linear, b_linear)
        
        # # # Add positiona encoding
        layer = "enc_pos_encoding_1"
        b_pe = parameters[layer,'b']
        transformer.add_pos_encoding("enc_linear_1", layer, b_pe)
        
        
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
         
        transformer.add_attention( "enc_norm_1", layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        
        # add res+norm2
        layer = "enc__layer_normalization_2"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        transformer.add_residual_connection("enc_norm_1", "enc__self_attention_1", f"{layer}__residual_1")
        transformer.add_layer_norm(f"{layer}__residual_1", "enc_norm_2", gamma1, beta1)
         
        # add ffn1
        ffn_parameter_dict = {}
        input_shape = parameters["enc__ffn_1"]['input_shape']
        ffn_params = transformer.get_fnn( "enc_norm_2", "enc__ffn_1", "enc__ffn_1", input_shape, parameters)
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
        transformer.add_attention( "enc_norm_3", layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        
        #add res+norm2
        layer = "enc__layer_normalization_4"
        gamma1 = parameters[layer, 'gamma']
        beta1 = parameters[layer, 'beta']
        
        transformer.add_residual_connection("enc_norm_1", "enc__self_attention_2", f"{layer}__residual_1")
        transformer.add_layer_norm(f"{layer}__residual_1", "enc_norm_4", gamma1, beta1)
         
        # add ffn1
        ffn_parameter_dict = {}
        input_shape = parameters["enc__ffn_2"]['input_shape']
        ffn_params = transformer.get_fnn( "enc_norm_4", "enc__ffn_2", "enc__ffn_2", input_shape, parameters)
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
                m.tnn_input_constraints.add(expr= transformer.M.dec_input[tnn_index, tnn_dim] == m.x[index, dim])
        dec_in = transformer.M.dec_input
        
        ## Dec Add Linear:
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
        dec_in = transformer.add_attention( dec_in, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        
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
        ffn_params = transformer.get_fnn( dec_norm_3,layer, nn_name, input_shape, parameters)
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
        dec_in = transformer.add_attention( dec_in, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        
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
        ffn_params = transformer.get_fnn( dec_norm_3,layer, nn_name, input_shape, parameters)
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
        layer = "linear_1"
        W_linear = parameters[layer,'W']
        b_linear = parameters[layer,'b'] 
        out = transformer.embed_input( dec_in, layer, embed_dim, W_linear, b_linear)
        
        # Linear transform 2
        embed_dim = transformer.M.dims_9
        layer = "linear_2"
        W_linear = parameters[layer,'W']
        b_linear = parameters[layer,'b'] 
        out = transformer.embed_input( dec_in, layer, embed_dim, W_linear, b_linear)
        
        
        # Linear transform 3
        embed_dim = transformer.M.dims_9 
        layer = "linear_3"
        W_linear = parameters[layer,'W']
        b_linear = parameters[layer,'b'] 
        out = transformer.embed_input( dec_in, layer, embed_dim, W_linear, b_linear)
        
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
            input, output = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
            pred_constr = add_predictor_constr(gurobi_model, nn, input, output)
        
        gurobi_model.update() # update gurobi model with FFN constraints
        
        
        ## Optimizes
        # gurobi_model.setParam('DualReductions',0)
        gurobi_model.setParam('MIPFocus',1)
        gurobi_model.setParam('LogFile', PATH+f'Logs\\{experiment_name}.log')
        gurobi_model.setParam('TimeLimit', TL) 
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
                    
            
        # save results
        tnn_config["Enc Seq Len"] = transformer.N
        tnn_config["Pred Len"] = dec_dim_1
        tnn_config["Overlap"] = 1
        tnn_config["TNN Model Dims"] = transformer.d_model
        tnn_config["TNN Head Dims"] = transformer.d_k
        tnn_config["TNN Head Size"] = transformer.d_H
        tnn_config["TNN Input Dim"] = transformer.input_dim
        tnn_config["Config"] = c+1

        if not TESTING:
            save_gurobi_results(gurobi_model, PATH+experiment_name, experiment_name, r+1, tnn_config)

if combine_files:            
    output_filename = f'{exp_name}.csv'
    combine(PATH, output_filename)