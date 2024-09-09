import pyomo.environ as pyo
import numpy as np
import torch
import math
from pyomo import dae
import json
import os
from helpers.extract_from_pretrained import get_pytorch_learned_parameters
from omlt import OmltBlock
from omlt.neuralnet import NetworkDefinition, ReluBigMFormulation
from omlt.io.keras import keras_reader
import omlt
import helpers.OMLT_helper 
import helpers.GUROBI_ML_helper as GUROBI_ML_helper

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

# def activate_envelope_att(model):
#         model.constr_convex.deactivate()
#         model.constr_concave.deactivate() 
#         model.constr_convex_tp.deactivate()
#         model.constr_convex_tp_sct.deactivate()
#         model.constr_concave_tp.deactivate()
#         model.constr_concave_tp_sct.deactivate()

#         if model.s_cv == 0: # --> convex region onlyt
#             model.constr_convex.activate()
#         elif model.s_cc == 0: # --> concave region only
#             model.constr_concave.activate() 
#         else: # in both regions
#             if model.t_cv == 0: # --> if x <= x_cv_tiepoint -->convex region
#                 model.constr_convex_tp.activate()
#             elif model.t_cv == 1: # -->concave region
#                 model.constr_convex_tp_sct.activate()
                
#             if model.t_cc == 0: # --> if x >= x_cc_tiepoint -->concave region
#                 model.constr_concave_tp.activate()
#             elif model.t_cc == 1:# --> convex region
#                 model.constr_concave_tp_sct.activate()

class Transformer:
    """ A Time Series Transformer based on Vaswani et al's "Attention is All You Need" paper."""
    def __init__(self, config_file, opt_model):
        
        self.M = opt_model
        # # time set
        # time_input = getattr( self.M, time_var_name)
        
         # get hyper params
        with open(config_file, "r") as file:
            config = json.load(file)

        self.N = config['hyper_params']['N'] # sequence length
        self.d_model = config['hyper_params']['d_model'] # embedding dimensions of model
        self.d_k = config['hyper_params']['d_k']
        self.d_H = config['hyper_params']['d_H']
        self.input_dim = config['hyper_params']['input_dim']
        
        file.close()
        
        # additional parameters
        self.transformer_pred = [0, 0]
        self.input_array = []
        self.epsilon = 1e-7
        
        # # initialise set of model dims
        # if not hasattr( self.M, "model_dims"):
        #     if self.d_model > 1:
        #         str_array = ["{}".format(x) for x in range(0, self.d_model)]
        #         self.M.model_dims = pyo.Set(initialize=str_array)
        #     else:
        #         self.M.model_dims = pyo.Set(initialize=[str(0)])
    
    def build_from_pytorch(self, pytorch_model, sample_enc_input, sample_dec_input, enc_bounds = None , dec_bounds = None):
        """ Builds transformer formulation for a trained pytorchtransfomrer model with and enocder an decoder """
        
        # Get learned parameters
        layer_names, parameters, _, enc_dec_count, _ = get_pytorch_learned_parameters(pytorch_model, sample_enc_input, sample_dec_input ,self.d_H, self.N)
        input_var_name, output_var_name, ffn_parameter_dict = self.__build_layers( layer_names, parameters, enc_dec_count , enc_bounds, dec_bounds)
        
        return [input_var_name, output_var_name, ffn_parameter_dict]
    
    def __add_linear(self, parameters, layer, input_name, embed_dim, layer_index=""):
        
        W_linear = parameters[layer,'W']
        try:
            b_linear = parameters[layer,'b']
        except:
            b_linear = None
            
        if "enc" in input_name and not "enc" in layer:
            output_name = "enc_"+layer+f"{layer_index}"
            
        elif "dec" in input_name and not "dec" in layer:
            output_name = "dec_"+layer+f"{layer_index}"
        else:
            output_name = layer
                        
        if not b_linear is None:    
            self.embed_input( input_name, output_name, embed_dim, W_linear, b_linear) 
        else:
            self.embed_input( input_name, output_name, embed_dim, W_linear)
            
        # return name of input to next layer
        return output_name
    
    def __add_ffn(self, parameters,ffn_parameter_dict, layer, input_name):

        input_shape = parameters[layer]['input_shape']
        ffn_params = self.get_fnn( input_name, layer, layer, input_shape, parameters)

        ffn_parameter_dict[layer] = ffn_params #.append(ffn_params)
        # return name of input to next layer
        return layer, ffn_parameter_dict
    
    def __add_layer_norm(self, parameters, layer, input_name, layer_index=""):
        gamma = parameters[layer, 'gamma']
        beta  = parameters[layer, 'beta']
        dict_gamma = {(v): val for v,val in zip( self.M.model_dims, gamma)}
        dict_beta  = {(v): val for v,val in zip( self.M.model_dims, beta)}
        
        if "enc" in input_name and not "enc" in layer:
            output_name = "enc_"+layer+f"{layer_index}"
            
        elif "dec" in input_name and not "dec" in layer:
            output_name = "dec_"+layer+f"{layer_index}"
        else:
            output_name = layer
        
        # define new gamma and beta params
        if not hasattr(self.M, f"{layer}_gamma"):
            setattr( self.M, f"{layer}_gamma", pyo.Param( self.M.model_dims, initialize = dict_gamma))
            setattr( self.M, f"{layer}_beta", pyo.Param( self.M.model_dims, initialize = dict_beta))
        
        # add layer normalization layer
        self.add_layer_norm( input_name, output_name, f"{layer}_gamma", f"{layer}_beta")
        
        # return name of input to next layer
        return output_name
    
    def __add_cross_attn(self, parameters, layer, input_name, enc_output_name):
        W_q = parameters["enc__self_attention_1",'W_q']
        W_k = parameters[layer,'W_k']
        W_v = parameters[layer,'W_v']
        W_o = parameters[layer,'W_o']
        
        try:
            b_q = parameters[layer,'b_q']
            b_k = parameters[layer,'b_k']
            b_v = parameters[layer,'b_v']
            b_o = parameters[layer,'b_o']
            
            self.add_attention( input_name, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, cross_attn=True, encoder_output=enc_output_name)
        except: # no bias values found
            self.add_attention( input_name, layer, W_q, W_k, W_k, W_o, cross_attn=True, encoder_output=enc_output_name)
        
        # return name of input to next layer
        return layer
    
    def __add_self_attn(self, parameters, layer, input_name):
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
                b_q = None
        
        if not b_q is None:     
            self.add_attention( input_name, layer, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        else:
            self.add_attention( input_name, layer, W_q, W_k, W_k, W_o)
            
        # return name of input to next layer
        return layer
    
    def __add_encoder_layer(self, parameters, layer, input_name, enc_layer, ffn_parameter_dict):
        embed_dim = self.M.model_dims 
        
        input_name_1 = self.__add_self_attn(parameters, f"enc__self_attention_1", input_name)
        self.add_residual_connection(input_name, input_name_1, f"{layer}__{enc_layer}_residual_1")
        input_name_2 = self.__add_layer_norm(parameters, "enc__layer_normalization_1", f"{layer}__{enc_layer}_residual_1", enc_layer)
        
        input_name, ffn_parameter_dict = self.__add_ffn(parameters, ffn_parameter_dict, "enc__ffn_1", input_name_2) # add ReLU ANN
        
        
        self.add_residual_connection(input_name, input_name_2, f"{layer}__{enc_layer}_residual_2")
        input_name = self.__add_layer_norm(parameters, "enc__layer_normalization_2", f"{layer}__{enc_layer}_residual_2", enc_layer)
        
        # return name of input to next layer
        return input_name, ffn_parameter_dict
    
    def __add_decoder_layer(self, parameters, layer, input_name, dec_layer, ffn_parameter_dict, enc_output_name):
        embed_dim = self.M.model_dims 
        
        input_name_1 = self.__add_self_attn(parameters, f"dec__self_attention_1", input_name)
        self.add_residual_connection(input_name, input_name_1, f"{layer}__{dec_layer}_residual_1")
        input_name_2 = self.__add_layer_norm(parameters, "dec__layer_normalization_1", f"{layer}__{dec_layer}_residual_1", dec_layer)
        
        input_name = self.__add_cross_attn(parameters, "dec__mutli_head_attention_1", input_name_2, enc_output_name)
        self.add_residual_connection(input_name, input_name_2, f"{layer}__{dec_layer}_residual_2")
        input_name_3 = self.__add_layer_norm(parameters, "dec__layer_normalization_2", f"{layer}__{dec_layer}_residual_2", dec_layer)
        
        input_name, ffn_parameter_dict = self.__add_ffn(parameters, ffn_parameter_dict, "dec__ffn_1", input_name_3) # add ReLU ANN
        
        self.add_residual_connection(input_name, input_name_3, f"{layer}__{dec_layer}_residual_3")
        input_name = self.__add_layer_norm(parameters, "dec__layer_normalization_3", f"{layer}__{dec_layer}_residual_3", dec_layer)
        
        # return name of input to next layer
        return input_name, ffn_parameter_dict
    
    def __build_layers(self, layer_names, parameters, enc_dec_count, enc_bounds, dec_bounds):
        """_summary_
        Adds transformer based on default pytorch transformer configuration 
        See https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        
        **NB**: 
        - extracting transformer layers from "summary" function does not include residual layers
        
        Default architecture:
        - transfromer:
            - encoder: # for norm_first=False (default)
                x  = src
                x  = self_attn(x)
                x  = dropout(x)
                x1 = norm_1( src + x)
                
                x  = linear_1(x1)
                x  = activation(x)
                x  = dropout_1(x)
                x  = linear_2(x)
                x  = dropout_2(x)
                x  = norm_2( x1 + x)
                
            - decoder: # for norm_first=False (default)
                x  = src
                x  = self_attn(x)
                x  = dropout(x)
                x1 = norm_1( src + x)
                
                x  = multihead_attn(x1)
                x  = dropout_2(x)
                x2 = norm_2( x1 + x)
                
                x  = linear_1(x)
                x  = activation(x)
                x  = dropout(x)
                x  = linear_2(x)
                x  = dropout_3(x)
                x  = norm_3( x2 + x)
        
        

        Args:
            layer_names (list): names of layers in transformer model
            parameters (dict): transformer model's learned parameters
            enc_bounds ((float,float), optional): upper, lower bound on encoder input (choose a logical value based on training data)
            dec_bounds ((float,float), optional): upper, lower bound on decoder input (choose a logical value based on training data)
        """
        ffn_parameter_dict = {}
        
        # define sets for input params
        enc_dim_1 = self.N 
        dec_dim_1 = self.N 
        self.M.enc_time_dims  = pyo.Set(initialize= list(range(enc_dim_1)))
        self.M.dec_time_dims  = pyo.Set(initialize= list(range(dec_dim_1)))
        self.M.dec_time_dims_param =  pyo.Set(initialize= list(range(dec_dim_1))) # - 2
        self.M.model_dims = pyo.Set(initialize= list(range(self.d_model)))
        self.M.input_dims = pyo.Set(initialize= list(range(self.input_dim)))
        enc_flag = False
        dec_flag = False
        
        for l, layer in enumerate(layer_names):
            print("layer iteration", layer, enc_flag, dec_flag)
            
            
            if l == 0: #input layer
                self.M.enc_input= pyo.Var(self.M.enc_time_dims,  self.M.input_dims, bounds=enc_bounds)
                enc_input_name = "enc_input"
                
                self.M.dec_input = pyo.Var(self.M.enc_time_dims,  self.M.input_dims, bounds=dec_bounds)
                dec_input_name = "dec_input"
                   
            if "enc" in layer:
                if not enc_flag:
                    enc_flag = True
                    # add enocder layers
                    for enc_layer in range(enc_dec_count[0]):
                        enc_input_name, ffn_parameter_dict = self.__add_encoder_layer(parameters, layer, enc_input_name, enc_layer, ffn_parameter_dict) 
                        
                    # normalize output of final layer    
                    enc_input_name = self.__add_layer_norm(parameters, "enc_layer_normalization_1", enc_input_name)
                
            elif "dec" in layer:
                if not dec_flag:
                    dec_flag = True
                    
                    # add decoder layers
                    for dec_layer in range(enc_dec_count[1]):
                        dec_input_name, ffn_parameter_dict  = self.__add_decoder_layer(parameters, layer, dec_input_name, dec_layer, ffn_parameter_dict, enc_input_name)
                        
                    # normalize output of final layer    
                    dec_input_name = self.__add_layer_norm(parameters, "dec_layer_normalization_1", dec_input_name)
                     
            elif "layer_norm" in layer:
                if l == len(layer_names)-1: #if final layer, only apply on decoder
                    dec_input_name = self.__add_layer_norm(parameters, layer, dec_input_name)
                else: 
                    enc_input_name = self.__add_layer_norm(parameters, layer, enc_input_name)
                    dec_input_name = self.__add_layer_norm(parameters, layer, dec_input_name)
                
            elif "linear" in layer:
                if l == len(layer_names)-1: 
                    embed_dim = self.M.input_dims # if last layer is linear, embed output dim = TNN input dim
                    dec_input_name = self.__add_linear( parameters, layer, dec_input_name, embed_dim)
                else:
                    embed_dim = self.M.model_dims # embed from current dim to self.M.model_dims
                    enc_input_name = self.__add_linear( parameters, layer, enc_input_name, embed_dim)
                    dec_input_name = self.__add_linear( parameters, layer, dec_input_name, embed_dim)
            
            elif "ffn" in layer:
                if l == len(layer_names)-1: 
                    dec_input_name,ffn_parameter_dict = self.__add_ffn(parameters,ffn_parameter_dict, layer, dec_input_name)
                else:
                    enc_input_name,ffn_parameter_dict = self.__add_ffn(parameters,ffn_parameter_dict, layer, enc_input_name)
                    dec_input_name,ffn_parameter_dict = self.__add_ffn(parameters,ffn_parameter_dict, layer, dec_input_name)
        
        #return [[encoder input name, decoder input name], transformer output name, ffn parameters dictionary]
        return [["enc_input","dec_input"], dec_input_name , ffn_parameter_dict] 

    def embed_input(self, input_var_name, embed_var_name, embed_dim_2, W_emb=None, b_emb = None):
        """
        Embed the feature dimensions of input
        """
        if not hasattr(self.M, "embed_constraints"):
            self.M.embed_constraints = pyo.ConstraintList()
            
        input_var = getattr(self.M, input_var_name)
        
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( self.M, set) )

            # define embedding var
            if not hasattr(self.M, embed_var_name):
                setattr(self.M, embed_var_name, pyo.Var(indices[0], embed_dim_2 , within=pyo.Reals, initialize= 0))
                embed_var = getattr(self.M, embed_var_name)   
            else:
                raise ValueError('Attempting to overwrite variable')
            
            if W_emb is None:
                for index, index_input in zip( embed_var.index_set(), set_var):
                    self.M.embed_constraints.add(embed_var[index] == input_var[index_input])
                    if isinstance(input_var, pyo.Var):
                        if not input_var[index_input].ub is None:
                            embed_var[index].ub = input_var[index_input].ub
                        if not input_var[index_input].lb is None:
                            embed_var[index].lb = input_var[index_input].lb
                        
                    elif isinstance(input_var, pyo.Param):
                        embed_var[index].ub = input_var[index_input]
                        embed_var[index].lb = input_var[index_input]         
            else: # w_emb has a value
                # Create weight variable
                W_emb_dict = {
                    (indices[1].at(s+1),embed_dim_2 .at(d+1)): W_emb[d][s]
                    for s in range(len(indices[1]))
                    for d in range(len(embed_dim_2))
                }
                setattr(self.M, embed_var_name+"_W_emb", pyo.Param(indices[1], embed_dim_2 , initialize=W_emb_dict))
                W_emb= getattr(self.M, embed_var_name+"_W_emb")   
                
                if b_emb:
                    # Create bias variable
                    b_emb_dict = {
                        (embed_dim_2.at(d+1)): b_emb[d]
                        for d in range(len(embed_dim_2))
                    }
                    
                    setattr(self.M, embed_var_name+"_b_emb", pyo.Param(embed_dim_2 , initialize=b_emb_dict))
                    b_emb= getattr(self.M, embed_var_name+"_b_emb")  
                
                    for d in embed_dim_2 :
                        for t in indices[0]:
                            self.M.embed_constraints.add(embed_var[t, d] 
                                                    == sum(input_var[t,s] * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                                                    )
                            if isinstance(input_var, pyo.Var):
                                if not input_var[t,indices[1].first()].ub is None and not input_var[t,indices[1].first()].lb is None:
                                    embed_var[t, d].ub = sum(max(input_var[t,s].ub * W_emb[s,d], input_var[t,s].lb * W_emb[s,d]) for s in indices[1]) +  b_emb[d]
                                    embed_var[t, d].lb = sum(min(input_var[t,s].ub * W_emb[s,d], input_var[t,s].lb * W_emb[s,d]) for s in indices[1]) +  b_emb[d]
                                    
                            elif isinstance(input_var, pyo.Param):
                                embed_var[t, d].ub = sum(input_var[t,s] * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                                embed_var[t, d].lb = sum(input_var[t,s] * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                else:
                    for d in embed_dim_2 :
                        for t in indices[0]:
                            self.M.embed_constraints.add(embed_var[t, d] 
                                                    == sum(input_var[t,s] * W_emb[s,d] for s in indices[1])
                                                    )
                            if isinstance(input_var, pyo.Var):
                                if not input_var[t,indices[1].first()].ub is None and not input_var[t,indices[1].first()].lb is None:
                                    embed_var[t, d].ub = sum(max(input_var[t,s].ub * W_emb[s,d], input_var[t,s].lb * W_emb[s,d]) for s in indices[1])
                                    embed_var[t, d].lb = sum(min(input_var[t,s].ub * W_emb[s,d], input_var[t,s].lb * W_emb[s,d]) for s in indices[1])
                                    
                            elif isinstance(input_var, pyo.Param):
                                embed_var[t, d].ub = sum(input_var[t,s] * W_emb[s,d] for s in indices[1])
                                embed_var[t, d].lb = sum(input_var[t,s] * W_emb[s,d] for s in indices[1])
        else:
            raise ValueError('Input value must be indexed')
        
    def add_layer_norm(self, input_var_name, layer_norm_var_name, gamma= None, beta = None, std=None):  # non-linear
        """
        Normalization over the sequennce of input
        """
        if not hasattr( self.M, "layer_norm_constraints"):
            self.M.layer_norm_constraints = pyo.ConstraintList()
        
        # get input
        input_var = getattr( self.M, input_var_name)
        
        # determine indices of input
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( self.M, set) )
                
            time_dim = indices[0]
            model_dims = indices[1]
        else:
            raise ValueError('Input value must be indexed (time, model_dim)')
                
        
        # Initialize variables
        if not hasattr( self.M, layer_norm_var_name):
            # define layer norm output var
            setattr( self.M, layer_norm_var_name, pyo.Var(time_dim, model_dims, within=pyo.Reals))
            layer_norm_var = getattr( self.M, layer_norm_var_name)
            
            # define calculation variables
            sum_name = 'sum_'+ layer_norm_var_name
            setattr( self.M, sum_name, pyo.Var(time_dim, within=pyo.Reals))
            sum_t = getattr( self.M, sum_name)
            
            variance_name = 'variance_'+ layer_norm_var_name
            setattr( self.M, variance_name, pyo.Var(time_dim, within=pyo.Reals))
            variance = getattr( self.M, variance_name)
            
            div_name = 'div_'+ layer_norm_var_name
            setattr( self.M, div_name, pyo.Var(time_dim, model_dims, within=pyo.Reals))
            div = getattr( self.M, div_name)
            
            denominator_name = 'denominator_'+ layer_norm_var_name
            setattr( self.M, denominator_name, pyo.Var(time_dim, within=pyo.Reals))
            denominator = getattr( self.M, denominator_name)
            
            denominator_abs_name = 'denominator_abs_'+ layer_norm_var_name
            setattr( self.M, denominator_abs_name, pyo.Var(time_dim, within=pyo.NonNegativeReals, bounds=(0,None)))
            denominator_abs = getattr( self.M, denominator_abs_name)
            
            numerator_name = 'numerator_'+ layer_norm_var_name
            setattr( self.M, numerator_name, pyo.Var(time_dim, model_dims, within=pyo.Reals))
            numerator = getattr( self.M, numerator_name)

            numerator_scaled_name = 'numerator_scaled_'+ layer_norm_var_name
            setattr( self.M, numerator_scaled_name, pyo.Var(time_dim, model_dims, within=pyo.Reals))
            numerator_scaled = getattr( self.M, numerator_scaled_name)
            
            numerator_squared_name = 'numerator_squared_'+ layer_norm_var_name
            setattr( self.M, numerator_squared_name, pyo.Var(time_dim, model_dims, within=pyo.Reals, bounds=(0,None)))
            numerator_squared = getattr( self.M, numerator_squared_name)
              
            numerator_squared_sum_name = 'numerator_squared_sum_'+ layer_norm_var_name
            setattr( self.M, numerator_squared_sum_name, pyo.Var(time_dim, within=pyo.Reals, bounds=(0,None)))
            numerator_squared_sum = getattr( self.M, numerator_squared_sum_name)
              
        else:
            raise ValueError('Attempting to overwrite variable')

        # Add constraints for layer norm
        # if self.d_model == 1:
        #     return
            
        for t in time_dim: 
            self.M.layer_norm_constraints.add(expr= sum_t[t] == sum(input_var[t, d] for d in model_dims) )
            
            # Constraints for each element in sequence
            for d in model_dims:  
                self.M.layer_norm_constraints.add(expr= numerator[t,d] == input_var[t, d] - ((1/ self.d_model) *sum_t[t]))
                self.M.layer_norm_constraints.add(expr= numerator_squared[t,d] == numerator[t,d]**2)
                
                self.M.layer_norm_constraints.add(expr= numerator_squared_sum[t] == sum(numerator_squared[t,d_prime] for d_prime in model_dims))
                self.M.layer_norm_constraints.add(expr= variance[t] * self.d_model == numerator_squared_sum[t])
                
                #self.M.layer_norm_constraints.add(expr= denominator[t] **2 == variance[t] )     ##IF SCIP SOLVER
                ## FOR SCIP or GUROBI: determine abs(denominator)
                self.M.layer_norm_constraints.add(expr= denominator[t] <= denominator_abs[t]) 
                self.M.layer_norm_constraints.add(expr= denominator[t]*denominator[t] == denominator_abs[t] * denominator_abs[t]) 
                
                self.M.layer_norm_constraints.add(expr= variance[t] == (denominator[t] * denominator_abs[t] ) )
                
                self.M.layer_norm_constraints.add(expr= div[t,d] * denominator[t] == numerator[t,d] )
                
                if gamma and beta:
                    self.M.layer_norm_constraints.add(expr= numerator_scaled[t,d] == getattr( self.M, gamma)[d] * div[t,d])
                    self.M.layer_norm_constraints.add(expr=layer_norm_var[t, d] == numerator_scaled[t,d] + getattr( self.M, beta)[d])
                    layer_norm_var[t, d].ub = getattr( self.M, beta)[d] + 4*getattr( self.M, gamma)[d]
                    layer_norm_var[t, d].lb = getattr( self.M, beta)[d] - 4*getattr( self.M, gamma)[d]
                    
                else: 
                    self.M.layer_norm_constraints.add(expr= numerator_scaled[t,d] == div[t,d])
                    self.M.layer_norm_constraints.add(expr=layer_norm_var[t, d] == numerator_scaled[t,d])
                    layer_norm_var[t, d].ub = 4
                    layer_norm_var[t, d].lb = -4
                    
                #Add bounds
                if std:
                    denominator[t].ub = std
                    denominator[t].lb = -std
                    
                div[t,d].ub = 4 #range of normalized values assuming normal distribution
                div[t,d].lb = -4
                    
                if input_var[t, d].ub and input_var[t, d].lb:
                    mean_u = (sum(input_var[t, d_prime].ub for d_prime in model_dims)/ self.d_model )
                    mean_l = (sum(input_var[t, d_prime].lb for d_prime in model_dims)/ self.d_model )
                    numerator[t,d].ub = input_var[t, d].ub - mean_l
                    numerator[t,d].lb = input_var[t, d].lb - mean_u
                    numerator_squared[t,d].ub = max(numerator[t,d].ub**2, numerator[t,d].lb**2) 
                    
                    if not std :
                        denominator[t].ub = abs( max(input_var[t,:].ub) - min(input_var[t,:].lb)) #standard deviation
                        denominator[t].lb = - abs( max(input_var[t,:].ub) - min(input_var[t,:].lb))#/8
                numerator_squared[t,d].lb = 0
            if input_var[t, d].ub and input_var[t, d].lb:
                numerator_squared_sum[t].ub = sum( (numerator_squared[t,d_prime].ub) for d_prime in model_dims) 
            numerator_squared_sum[t].lb = 0
            
        
    def add_attention(self, input_var_name, output_var_name, W_q, W_k, W_v, W_o, b_q = None, b_k = None, b_v = None, b_o = None, cross_attn=False, encoder_output=None):
        """
        Multihead attention between each element of embedded sequence
        
        Uses the pyo.exp() function to calculate softmax. 
        This is compatible with gurobi which allows for the outer approximation of the function to be calculated
        """
        
        # get input
        input_var = getattr( self.M, input_var_name)
        
        # determine indices of input
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( self.M, set) )
                
            time_dim = indices[0]
            model_dims = indices[1]
            W_dim_1_kv = indices[1] # kv dim same as q
            res_dim_1_kv = indices[0]
        else:
            raise ValueError('Input value must be indexed (time, model_dim)')
        
        # Check for cross attention between encoder and decoder
        if cross_attn and not encoder_output is None:
            encoder_output_var = getattr( self.M, encoder_output)
            if encoder_output_var.is_indexed():
                set_var = encoder_output_var.index_set()
                indices = []
                for set in str(set_var).split("*"):
                    indices.append( getattr( self.M, set) )
                W_dim_1_kv  = indices[1] # Weights k,v dim based on enc dim but Weight q dim based on decoder
                res_dim_1_kv = indices[0] # K and V first dim
            else:
                raise ValueError(f'{encoder_output} must be indexed (time, model_dim)')
        
        # define variables and parameters of this layer
        if not hasattr( self.M, output_var_name):
            setattr( self.M, output_var_name, pyo.Var(time_dim, model_dims , within=pyo.Reals))
            attention_output = getattr( self.M, output_var_name)
            
            setattr( self.M, "Block_"+output_var_name, pyo.Block())
            MHA_Block  = getattr( self.M, "Block_"+output_var_name)
            
            MHA_Block.attention_constraints = pyo.ConstraintList()
            MHA_Block.constr_convex = pyo.ConstraintList()
            MHA_Block.constr_concave = pyo.ConstraintList()
            MHA_Block.constr_convex_tp = pyo.ConstraintList()
            MHA_Block.constr_convex_tp_sct = pyo.ConstraintList()
            MHA_Block.constr_concave_tp = pyo.ConstraintList()
            MHA_Block.constr_concave_tp_sct = pyo.ConstraintList()
        else:
            raise ValueError('Attempting to overwrite variable')

        # define sets, vars
        MHA_Block.heads = pyo.RangeSet(1, self.d_H)
        MHA_Block.k_dims = pyo.RangeSet(1, self.d_k)

        W_q_dict = {
            (D, H, K): W_q[d][h][k]
            for d,D in enumerate(model_dims )
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.k_dims)
        }
        W_k_dict = {
            (D, H, K): W_k[d][h][k]
            for d,D in enumerate(W_dim_1_kv)
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.k_dims)
        }
        W_v_dict = {
            (D, H, K): W_v[d][h][k]
            for d,D in enumerate(W_dim_1_kv )
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.k_dims)
        }
        W_o_dict = {
            (D, H, K): W_o[h][k][d]
            for d,D in enumerate(model_dims )
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.k_dims)
        }
 
        MHA_Block.W_q = pyo.Param(model_dims ,MHA_Block.heads,MHA_Block.k_dims, initialize=W_q_dict, mutable=False)
        MHA_Block.W_k = pyo.Param(W_dim_1_kv ,MHA_Block.heads,MHA_Block.k_dims, initialize=W_k_dict, mutable=False)
        MHA_Block.W_v = pyo.Param(W_dim_1_kv ,MHA_Block.heads,MHA_Block.k_dims, initialize=W_v_dict, mutable=False)
        MHA_Block.W_o = pyo.Param(model_dims ,MHA_Block.heads,MHA_Block.k_dims, initialize=W_o_dict, mutable=False)
        
        if not b_q is None:
            b_q_dict = {
                        (h, k): b_q[h-1][k-1]
                        for h in MHA_Block.heads
                        for k in MHA_Block.k_dims
                       }
            MHA_Block.b_q = pyo.Param(MHA_Block.heads, MHA_Block.k_dims, initialize=b_q_dict, mutable=False)
            
        if not b_k is None:
            b_k_dict = {
                        (h, k): b_k[h-1][k-1]
                        for h in MHA_Block.heads
                        for k in MHA_Block.k_dims
                       }
            MHA_Block.b_k = pyo.Param(MHA_Block.heads, MHA_Block.k_dims, initialize=b_k_dict, mutable=False)
            
        if not b_v is None: 
            b_v_dict = {
                        (h, k): b_v[h-1][k-1]
                        for h in MHA_Block.heads
                        for k in MHA_Block.k_dims
                       }
            MHA_Block.b_v = pyo.Param(MHA_Block.heads, MHA_Block.k_dims, initialize=b_v_dict, mutable=False)
            
        if not b_o is None:
            b_o_dict = {(d): val for d, val in zip(model_dims , b_o) }
            MHA_Block.b_o = pyo.Param(model_dims , initialize=b_o_dict, mutable=False)
            

        MHA_Block.Q = pyo.Var(MHA_Block.heads, time_dim, MHA_Block.k_dims, within=pyo.Reals) 
        MHA_Block.K = pyo.Var(MHA_Block.heads, res_dim_1_kv, MHA_Block.k_dims, within=pyo.Reals)
        MHA_Block.V = pyo.Var(MHA_Block.heads, res_dim_1_kv, MHA_Block.k_dims, within=pyo.Reals) 

        MHA_Block.QK = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, MHA_Block.k_dims, within=pyo.Reals) 
        MHA_Block.compatibility = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals) 
        scale = np.sqrt(self.d_k) 
        
        MHA_Block.compatibility_exp = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.NonNegativeReals, bounds=(0,None)) # range: 0-->inf, initialize=init_compatibility_exp)
        MHA_Block.compatibility_exp_sum = pyo.Var(MHA_Block.heads, time_dim, within=pyo.NonNegativeReals, bounds=(0,None)) #, initialize=init_compatibility_sum)
        MHA_Block.tie_point_cc = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tie_point_cv = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tie_point_cc_prime = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tie_point_cv_prime = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tp_cv_mult_1 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tp_cv_mult_2 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tp_cc_mult_1 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tp_cc_mult_2 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        
        BigM_s = 0.5
        BigM_t = 1
        MHA_Block.sct = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals, bounds=(0,1))
        
        MHA_Block.s_cv= pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Binary)
        MHA_Block.t_cv= pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Binary)
        
        MHA_Block.s_cc= pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Binary)
        MHA_Block.t_cc= pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Binary)
        
        MHA_Block.tp_cv =pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Binary)
        MHA_Block.tp_cc =pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Binary)

        MHA_Block.attention_weight_cc = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals, bounds=(0,1))
        MHA_Block.attention_weight_x_cc_prime = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals, bounds=(0,1))
        MHA_Block.attention_weight_x_cc= pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals, bounds=(0,1))
        
        MHA_Block.attention_weight_cv = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals, bounds=(0,1))
        MHA_Block.attention_weight_x_cv_prime = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals, bounds=(0,1))
        MHA_Block.attention_weight_x_cv = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals, bounds=(0,1))
        
        MHA_Block.attention_weight = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals, bounds=(0,1))  # softmax ( (Q * K)/sqrt(d_k) )
        MHA_Block.tp_cv_sct = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals, bounds=(0,1))
        MHA_Block.tp_cv_sct_mult_1 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tp_cv_sct_mult_2 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tp_cv_sct_mult_1_2 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tp_cv_sct_mult_3 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        
        MHA_Block.tp_cc_sct = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals, bounds=(0,1))
        MHA_Block.tp_cc_sct_mult_1 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tp_cc_sct_mult_2 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tp_cc_sct_mult_1_2 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        MHA_Block.tp_cc_sct_mult_3 = pyo.Var(MHA_Block.heads, time_dim, res_dim_1_kv, within=pyo.Reals)
        
        MHA_Block.attention_score = pyo.Var(
            MHA_Block.heads, time_dim, MHA_Block.k_dims, within=pyo.Reals
        )  # softmax ( (Q * K)/sqrt(d_k) ) * V
        MHA_Block.attWK = pyo.Var(MHA_Block.heads, time_dim, MHA_Block.k_dims, res_dim_1_kv , within=pyo.Reals)
        
        for h in MHA_Block.heads:
            # Check if multihead attention or self attention
            if cross_attn and not encoder_output is None:
                input = encoder_output_var# calculate K and V from output of encoder
            else:
                input = input_var
            
            # Define K and V
            for n in res_dim_1_kv:
                    for k in MHA_Block.k_dims:

                        # constraints for Key
                        if not b_k is None:
                            MHA_Block.attention_constraints.add(
                            expr=MHA_Block.K[h, n, k]
                            == sum(input[n, d] * MHA_Block.W_k[d, h, k] for d in W_dim_1_kv ) + MHA_Block.b_k[h,k]
                            )  
                            #Add bounds
                            MHA_Block.K[h, n, k].ub = sum( max(input[n,d].ub * MHA_Block.W_k[d, h, k], input[n,d].lb * MHA_Block.W_k[d, h, k])  for d in W_dim_1_kv ) + MHA_Block.b_k[h,k]
                            MHA_Block.K[h, n, k].lb = sum( min(input[n,d].ub * MHA_Block.W_k[d, h, k], input[n,d].lb * MHA_Block.W_k[d, h, k])  for d in W_dim_1_kv ) + MHA_Block.b_k[h,k]
                            # if k_bound_1 < k_bound_2: 
                            #     MHA_Block.K[h, n, k].ub = k_bound_2
                            #     MHA_Block.K[h, n, k].lb = k_bound_1
                            # else:
                            #     MHA_Block.K[h, n, k].ub = k_bound_1
                            #     MHA_Block.K[h, n, k].lb = k_bound_2
                            
                        else: 
                            MHA_Block.attention_constraints.add(
                                expr=MHA_Block.K[h, n, k]
                                == sum(input[n, d] * MHA_Block.W_k[d, h, k] for d in W_dim_1_kv)
                            )
                            #Add bounds
                            
                            MHA_Block.K[h, n, k].ub = sum( max(input[n,d].ub * MHA_Block.W_k[d, h, k], input[n,d].lb * MHA_Block.W_k[d, h, k])  for d in W_dim_1_kv ) 
                            MHA_Block.K[h, n, k].lb = sum( min(input[n,d].ub * MHA_Block.W_k[d, h, k], input[n,d].lb * MHA_Block.W_k[d, h, k])  for d in W_dim_1_kv ) 
                            # if k_bound_1 < k_bound_2: 
                            #     MHA_Block.K[h, n, k].ub = k_bound_2
                            #     MHA_Block.K[h, n, k].lb = k_bound_1
                            # else:
                            #     MHA_Block.K[h, n, k].ub = k_bound_1
                            #     MHA_Block.K[h, n, k].lb = k_bound_2
                            
                        # constraints for Value    
                        if not b_v is None:
                            MHA_Block.attention_constraints.add(
                            expr=MHA_Block.V[h, n, k]
                            == sum(input[n, d] * MHA_Block.W_v[d, h, k] for d in W_dim_1_kv) + MHA_Block.b_v[h,k]
                            )  
                            #Add bounds
                            
                            MHA_Block.V[h, n, k].ub = sum( max(input[n,d].ub * MHA_Block.W_v[d, h, k], input[n,d].lb * MHA_Block.W_v[d, h, k])  for d in W_dim_1_kv ) + MHA_Block.b_v[h,k]
                            MHA_Block.V[h, n, k].lb = sum( min(input[n,d].ub * MHA_Block.W_v[d, h, k], input[n,d].lb * MHA_Block.W_v[d, h, k])  for d in W_dim_1_kv ) + MHA_Block.b_v[h,k]
                            # if v_bound_1 < v_bound_2: 
                            #     MHA_Block.V[h, n, k].ub = v_bound_2
                            #     MHA_Block.V[h, n, k].lb = v_bound_1
                            # else:
                            #     MHA_Block.V[h, n, k].ub = v_bound_1
                            #     MHA_Block.V[h, n, k].lb = v_bound_2
                            
                        else: 
                            MHA_Block.attention_constraints.add(
                                expr=MHA_Block.V[h, n, k]
                                == sum(input[n, d] * MHA_Block.W_v[d, h, k] for d in W_dim_1_kv ) 
                            )
                            #Add bounds     
                            MHA_Block.V[h, n, k].ub = sum( max(input[n,d].ub * MHA_Block.W_v[d, h, k], input[n,d].lb * MHA_Block.W_v[d, h, k])  for d in W_dim_1_kv )
                            MHA_Block.V[h, n, k].lb = sum( min(input[n,d].ub * MHA_Block.W_v[d, h, k], input[n,d].lb * MHA_Block.W_v[d, h, k])  for d in W_dim_1_kv )
                            # if v_bound_1 < v_bound_2: 
                            #     MHA_Block.V[h, n, k].ub = v_bound_2
                            #     MHA_Block.V[h, n, k].lb = v_bound_1
                            # else:
                            #     MHA_Block.V[h, n, k].ub = v_bound_1
                            #     MHA_Block.V[h, n, k].lb = v_bound_2
                                
            for n in time_dim:
                    for k in MHA_Block.k_dims:
                        
                        # constraints for Query
                        if not b_q is None:
                            MHA_Block.attention_constraints.add(
                            expr=MHA_Block.Q[h, n, k]
                            == sum(input_var[n,d] * MHA_Block.W_q[d, h, k] for d in model_dims ) + MHA_Block.b_q[h,k] 
                            )  
                            
                            #Add bounds
                            MHA_Block.Q[h, n, k].ub = sum( max(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model_dims ) + MHA_Block.b_q[h,k]
                            MHA_Block.Q[h, n, k].lb = sum( min(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model_dims ) + MHA_Block.b_q[h,k]
                            # if q_bound_1 < q_bound_2: 
                            #     MHA_Block.Q[h, n, k].ub = q_bound_2
                            #     MHA_Block.Q[h, n, k].lb = q_bound_1
                            # else:
                            #     MHA_Block.Q[h, n, k].ub = q_bound_1
                            #     MHA_Block.Q[h, n, k].lb = q_bound_2
                        else: 
                            MHA_Block.attention_constraints.add(
                                expr=MHA_Block.Q[h, n, k]
                                == sum(input_var[n, d] * MHA_Block.W_q[d, h, k] for d in model_dims )
                            )
                            #Add bounds
                            MHA_Block.Q[h, n, k].ub = sum( max(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model_dims )
                            MHA_Block.Q[h, n, k].lb = sum( min(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model_dims )
                            # if q_bound_1 < q_bound_2: 
                            #     MHA_Block.Q[h, n, k].ub = q_bound_2
                            #     MHA_Block.Q[h, n, k].lb = q_bound_1
                            # else:
                            #     MHA_Block.Q[h, n, k].ub = q_bound_1
                            #     MHA_Block.Q[h, n, k].lb = q_bound_2
                        
                        
            

                        # attention score = sum(attention_weight * V)
                        MHA_Block.attention_constraints.add(
                            expr=MHA_Block.attention_score[h, n, k]
                            == sum( MHA_Block.attWK[h, n, k, n2] for n2 in res_dim_1_kv)
                        )   
                        
                        # MHA_Block.attention_constraints.add(
                        #     expr=MHA_Block.attention_score[h, n, k]
                        #     == sum(
                        #         MHA_Block.attention_weight[h, n, n2] * MHA_Block.V[h, n2, k]
                        #         for n2 in res_dim_1_kv
                        #     )
                        # )
                        
                        
                    for p in res_dim_1_kv:
                        # compatibility sqrt(Q * K) across all pairs of elements
                        MHA_Block.attention_constraints.add(
                            expr=MHA_Block.compatibility[h, n, p] *scale
                            == sum(MHA_Block.QK[h, n, p, k] for k in MHA_Block.k_dims)
                        ) 
                        # MHA_Block.attention_constraints.add(
                        #     expr=MHA_Block.compatibility[h, n, p] *scale
                        #     == sum(MHA_Block.Q[h, n, k] * (MHA_Block.K[ h, p, k] )for k in MHA_Block.k_dims)
                        # )  
                        
                        # exp(compatibility)
                        MHA_Block.attention_constraints.add(expr= pyo.exp(MHA_Block.compatibility[h,n,p]) == MHA_Block.compatibility_exp[h, n, p] )
                        
                        
                        
                    # sum over exp(compatbility)
                    MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility_exp_sum[h, n] == sum(MHA_Block.compatibility_exp[h, n, p] for p in res_dim_1_kv))
                    
                    # sum over softmax = 1    
                    MHA_Block.attention_constraints.add(
                        expr=sum(MHA_Block.attention_weight[h, n, n_prime] for n_prime in res_dim_1_kv) == 1
                    )
                    
                    for n2 in res_dim_1_kv:
                        MHA_Block.attention_weight[h, n, n2].ub = 1
                        MHA_Block.attention_weight[h, n, n2].lb = 0
                
                        # attention weights softmax(compatibility)   
                        MHA_Block.attention_constraints.add(
                            expr=MHA_Block.attention_weight[h, n, n2] * MHA_Block.compatibility_exp_sum[h, n]
                            == MHA_Block.compatibility_exp[h, n, n2]) 
                        
                    
                    
            #Add bounds            
            for n in time_dim:
                for p in res_dim_1_kv:
                    MHA_Block.compatibility[h,n,p].ub = (1/scale ) * (sum(max(MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].lb,
                                                                                  MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].ub, 
                                                                                  MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].lb, 
                                                                                  MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].ub) for k in MHA_Block.k_dims) )
                    MHA_Block.compatibility[h,n,p].lb = (1/scale ) * (sum(min(MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].lb, 
                                                                                  MHA_Block.Q[h, n, k].lb * MHA_Block.K[ h, p, k].ub, 
                                                                                  MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].lb, 
                                                                                  MHA_Block.Q[h, n, k].ub * MHA_Block.K[ h, p, k].ub) for k in MHA_Block.k_dims) )
                    
                    
                    MHA_Block.compatibility_exp[h,n,p].ub = math.exp(MHA_Block.compatibility[h,n,p].ub)
                    MHA_Block.compatibility_exp[h,n,p].lb = max(0, 1 + MHA_Block.compatibility[h,n,p].lb)
                    
                    for k in MHA_Block.k_dims:
                        MHA_Block.attention_constraints.add(expr=MHA_Block.QK[h, n, p, k] == MHA_Block.Q[h, n, k] * MHA_Block.K[ h, p, k])
                        MHA_Block.attention_constraints.add(expr= MHA_Block.attWK[h, n, k, p] == MHA_Block.attention_weight[h, n, p] * MHA_Block.V[h, p, k])
                            
                        
                        self.__McCormick_bb(MHA_Block.QK[h, n, p, k], MHA_Block.Q[h, n, k], MHA_Block.K[ h, p, k]) # add McCromick Envelope
                        self.__McCormick_bb(MHA_Block.attWK[h, n, k, p], MHA_Block.attention_weight[h, n, p], MHA_Block.V[h, p, k]) # add McCromick Envelope
                
                for k in MHA_Block.k_dims:
                    # MHA_Block.attention_constraints.add(
                    #     expr=MHA_Block.attention_score[h, n, k]
                    #     == sum(
                    #         MHA_Block.attention_weight[h, n, n2] * MHA_Block.V[h, n2, k]
                    #         for n2 in res_dim_1_kv
                    #     )
                    # )
                    MHA_Block.attention_score[h, n, k].ub = (sum(max(MHA_Block.attention_weight[h, n, n2].lb * MHA_Block.V[h, n2, k].lb,
                                                                    MHA_Block.attention_weight[h, n, n2].lb * MHA_Block.V[h, n2, k].ub, 
                                                                    MHA_Block.attention_weight[h, n, n2].ub * MHA_Block.V[h, n2, k].lb, 
                                                                    MHA_Block.attention_weight[h, n, n2].ub * MHA_Block.V[h, n2, k].ub) 
                                                            for n2 in res_dim_1_kv) )
                    MHA_Block.attention_score[h, n, k].lb = (sum(min(MHA_Block.attention_weight[h, n, n2].lb * MHA_Block.V[h, n2, k].lb, 
                                                                    MHA_Block.attention_weight[h, n, n2].lb * MHA_Block.V[h, n2, k].ub, 
                                                                    MHA_Block.attention_weight[h, n, n2].ub * MHA_Block.V[h, n2, k].lb, 
                                                                    MHA_Block.attention_weight[h, n, n2].ub * MHA_Block.V[h, n2, k].ub) 
                                                            for n2 in res_dim_1_kv) )
                    
                MHA_Block.compatibility_exp_sum[h, n].ub = sum( MHA_Block.compatibility_exp[h,n,p].ub for p in time_dim) 
                MHA_Block.compatibility_exp_sum[h, n].lb = sum( MHA_Block.compatibility_exp[h,n,p].lb for p in time_dim)       
                ##############-----------------------------------############    
                #for p in res_dim_1_kv:    
                    # # f(x) >= f_cv(x): attention weight >= convex envelope
                    # MHA_Block.attention_constraints.add(
                    #     MHA_Block.attention_weight[h, n, p]  >= MHA_Block.attention_weight_cv[h, n, p]
                    # )
                    # # f(x) <= f_cc(x): attention weight <= concave envelope
                    # MHA_Block.attention_constraints.add(
                    #     MHA_Block.attention_weight[h, n, p]  <= MHA_Block.attention_weight_cc[h, n, p]
                    # )
           
                    # # Constraints for Concave/convex envelope
                    # # set convex aux var -- s=0: f(x_UB) <= 0.5 --> convex zone, s=1: f(x_UB) >= 0.5 --> concave zone
                    # MHA_Block.attention_constraints.add(
                    #     expr= MHA_Block.attention_weight[h, n, p].ub <= 0.5  + (BigM_s * MHA_Block.s_cv[h,n,p])
                    # )
                    
                    # # MHA_Block.attention_constraints.add(
                    # #     expr= MHA_Block.attention_weight[h, n, p].ub >= 0.5  - (BigM_s * (1 - MHA_Block.s_cv[h,n,p]))
                    # # )
                    # MHA_Block.attention_constraints.add(
                    #     expr= MHA_Block.attention_weight[h, n, p].ub - 0.5 + BigM_s >= BigM_s *  MHA_Block.s_cv[h,n,p]
                    # )

                    # # set convex aux var -- f(x_LB) <= 0.5 --> convex zone else f(x_LB) >= 0.5 --> concave zone
                    # MHA_Block.attention_constraints.add(
                    #     expr= MHA_Block.attention_weight[h, n, p].lb >= 0.5 - (BigM_s *  (MHA_Block.s_cc[h,n,p]))
                    # )
                    # MHA_Block.attention_constraints.add(
                    #     expr= (BigM_s * MHA_Block.s_cc[h,n,p]) <= 0.5 + BigM_s - MHA_Block.attention_weight[h, n, p].lb
                    # )
                    
                    # # # sct(x)
                    # A = ((MHA_Block.attention_weight[h, n, p].ub - MHA_Block.attention_weight[h, n, p].lb) / (MHA_Block.compatibility[h,n,p].ub - MHA_Block.compatibility[h,n,p].lb )) 
                    # b = ( (MHA_Block.compatibility[h,n,p].ub * MHA_Block.attention_weight[h, n, p].lb) - (MHA_Block.compatibility[h,n,p].lb * MHA_Block.attention_weight[h, n, p].ub)) /(MHA_Block.compatibility[h,n,p].ub - MHA_Block.compatibility[h,n,p].lb )
                    # MHA_Block.attention_constraints.add(
                    #     MHA_Block.sct[h, n, p]   == (A *  MHA_Block.compatibility[h,n,p]) + b
                    # )

                    
                    # # # # Add concave/convex evelope function constraints
                    # # # when f(UB) <= 0.5: convex
                    # MHA_Block.constr_convex.add( 
                    #     MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.attention_weight[h, n, p]
                    # )
                    # MHA_Block.constr_convex.add( 
                    #     MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.sct[h, n, p] 
                    # )
                    # # when f(LB) >= 0.5: concave 
                    # MHA_Block.constr_concave.add( 
                    #     MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.sct[h, n, p] 
                    # )
                    # MHA_Block.constr_concave.add( 
                    #     MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.attention_weight[h, n, p] 
                    # )
                    # # otherwise: use concave and convex tie points
                    # MHA_Block.constr_concave_tp.add( # when x >= x_cc
                    #     MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.attention_weight[h, n, p] 
                    # )
                    # MHA_Block.constr_concave_tp_sct.add( # when x <= x_cc --> cc_sct()
                    #     MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.tp_cc_sct[h, n, p]
                    # ) 
                    # MHA_Block.constr_convex_tp_sct.add( # when x >= x_cv --> cv_sct()
                    #     MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.tp_cv_sct[h, n, p]
                    # ) 
                    # MHA_Block.constr_convex_tp.add( # when x <= x_cv
                    #     MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.attention_weight[h, n, p]
                    # )
                    
                    # ## Add tp_cv_sct constraints
                    # #bounds
                    # MHA_Block.attention_constraints.add(# att(cv_prime)
                    #     expr=  MHA_Block.attention_weight_x_cv_prime[h, n, n2] <= 1 
                    # )
                    # MHA_Block.attention_constraints.add( # att(x_cv)
                    #     expr=  MHA_Block.attention_weight_x_cv[h, n, n2] <= 1
                    # )
                    # MHA_Block.attention_constraints.add( # att(x_cv)
                    #     expr=   MHA_Block.tp_cv_sct[h, n, p] <= 1
                    # )
                    # # tie_point_cv[h, n, p] = max(tie_point_cv_prime, compatibility.lb  )
                    # BigM_prime = max( MHA_Block.compatibility[h,n,p_prime].ub for p_prime in time_dim)
                    # MHA_Block.attention_constraints.add(
                    #     MHA_Block.tie_point_cv_prime[h, n, p] - MHA_Block.compatibility[h,n,p].lb <= BigM_prime * (1 - MHA_Block.tp_cv[h,n,p])
                    # )
                    # MHA_Block.attention_constraints.add(
                    #     MHA_Block.tie_point_cv_prime[h, n, p] - MHA_Block.compatibility[h,n,p].lb >= -BigM_prime * ( MHA_Block.tp_cv[h,n,p])
                    # )
                    # MHA_Block.attention_constraints.add( # define tie_point_cv
                    #     MHA_Block.tie_point_cv[h, n, p]  == MHA_Block.tie_point_cv_prime[h, n, p]*(1 - MHA_Block.tp_cv[h,n,p])  + (MHA_Block.compatibility[h,n,p].lb * MHA_Block.tp_cv[h,n,p])
                    # )
                    # MHA_Block.attention_constraints.add( # softmax(tie_point_cv)
                    #     MHA_Block.attention_weight_x_cv[h, n, p] == MHA_Block.attention_weight_x_cv_prime[h, n, p]*(1 - MHA_Block.tp_cv[h,n,p])  + MHA_Block.attention_weight[h,n,p].lb * MHA_Block.tp_cv[h,n,p]
                    # )
                    # # Is x <= x_cv? --> convex zone
                    # MHA_Block.attention_constraints.add(
                    #     expr=  MHA_Block.tie_point_cv[h, n, p] - MHA_Block.compatibility[h,n,p] <= BigM_prime * (1-MHA_Block.t_cv[h,n,p])
                    # )
                    # MHA_Block.attention_constraints.add(
                    #     expr=  MHA_Block.tie_point_cv[h, n, p] - MHA_Block.compatibility[h,n,p] >= - BigM_prime * (MHA_Block.t_cv[h,n,p])
                    # )
                    # # define tie_point_cv_prime[h, n, p]
                    # MHA_Block.attention_constraints.add( # 
                    #     expr=  MHA_Block.tp_cv_mult_1[h, n, p]  == MHA_Block.attention_weight[h,n,p].ub  - MHA_Block.attention_weight_x_cv_prime[h, n, p]
                    # )
                    # MHA_Block.attention_constraints.add( # 
                    #     expr=  MHA_Block.tp_cv_mult_2[h, n, p]  == MHA_Block.attention_weight_x_cv_prime[h, n, p] * ( 1 -  MHA_Block.attention_weight_x_cv_prime[h, n, p])
                    # )
                    # MHA_Block.attention_constraints.add( 
                    #     expr=  (MHA_Block.compatibility[h,n,p].ub - MHA_Block.tie_point_cv_prime[h, n, p]) * MHA_Block.tp_cv_mult_2[h, n, p]  == MHA_Block.tp_cv_mult_1[h, n, p]
                    # )
                    # # define tie point cv  secant
                    # MHA_Block.constr_convex_tp_sct.add( 
                    #     expr=  MHA_Block.tp_cv_sct[h, n, p] - MHA_Block.attention_weight[h,n,p].ub == 
                    #                                         + (MHA_Block.tp_cv_sct_mult_1_2[h, n, p] 
                    #                                            * (MHA_Block.compatibility[h,n,p]
                    #                                             - MHA_Block.compatibility[h,n,p].ub))
                    # )
                    # MHA_Block.constr_convex_tp_sct.add( 
                    #     expr=  MHA_Block.tp_cv_sct_mult_1_2[h, n, p] * MHA_Block.tp_cv_sct_mult_2[h, n, p] == MHA_Block.tp_cv_sct_mult_1[h, n, p] 
                    # )
                    # MHA_Block.constr_convex_tp_sct.add( 
                    #     expr=  MHA_Block.tp_cv_sct_mult_1[h, n, p] == MHA_Block.attention_weight[h,n,p].ub -  MHA_Block.attention_weight_x_cv[h, n, p]
                    # )
                    # MHA_Block.constr_convex_tp_sct.add( 
                    #     expr=  MHA_Block.tp_cv_sct_mult_2[h, n, p] == MHA_Block.compatibility[h,n,p].ub - MHA_Block.tie_point_cv[h, n, p]
                    # )
                    
                    # ## Add tp_cc_sct constraints
                    # #bounds
                    # MHA_Block.attention_constraints.add(# att(cc_prime)
                    #     expr=  MHA_Block.attention_weight_x_cc_prime[h, n, n2] <= 1 
                    # )
                    # MHA_Block.attention_constraints.add( # att(x_cc)
                    #     expr=  MHA_Block.attention_weight_x_cc[h, n, n2] <= 1
                    # )
                    # MHA_Block.attention_constraints.add( # att(x_cc)
                    #     expr=   MHA_Block.tp_cc_sct[h, n, p] <= 1
                    # )
                    # # tie_point_cc[h, n, p] = min(tie_point_cc_prime, compatibility.ub  )
                    # MHA_Block.attention_constraints.add(
                    #     MHA_Block.tie_point_cc_prime[h, n, p] - MHA_Block.compatibility[h,n,p].ub <= BigM_prime * (1 - MHA_Block.tp_cc[h,n,p])
                    # )
                    # MHA_Block.attention_constraints.add(
                    #     MHA_Block.tie_point_cc_prime[h, n, p] - MHA_Block.compatibility[h,n,p].ub >= -BigM_prime * ( MHA_Block.tp_cc[h,n,p])
                    # )
                    # MHA_Block.attention_constraints.add( # define tie_point_cc
                    #     MHA_Block.tie_point_cc[h, n, p]  == MHA_Block.tie_point_cc_prime[h, n, p]*(MHA_Block.tp_cc[h,n,p])  + (MHA_Block.compatibility[h,n,p].ub * (1 - MHA_Block.tp_cc[h,n,p]))
                    # )
                    # MHA_Block.attention_constraints.add( # softmax(tie_point_cc)
                    #     MHA_Block.attention_weight_x_cc[h, n, p] == MHA_Block.attention_weight_x_cc_prime[h, n, p]*(MHA_Block.tp_cc[h,n,p])  + (MHA_Block.attention_weight[h,n,p].ub * (1 - MHA_Block.tp_cc[h,n,p]))
                    # )
                    # # Is x <= x_cc? --> convex zone
                    # MHA_Block.attention_constraints.add(
                    #     expr=  MHA_Block.compatibility[h,n,p] - MHA_Block.tie_point_cc[h, n, p] <= BigM_prime * (1 - MHA_Block.t_cc[h,n,p])
                    # )
                    # MHA_Block.attention_constraints.add(
                    #     expr=  MHA_Block.compatibility[h,n,p] - MHA_Block.tie_point_cc[h, n, p]>= - BigM_prime * (MHA_Block.t_cc[h,n,p])
                    # )
                    # # define tie_point_cc_prime[h, n, p]
                    # MHA_Block.attention_constraints.add( # 
                    #     expr=  MHA_Block.tp_cc_mult_1[h, n, p]  == MHA_Block.attention_weight_x_cc_prime[h, n, p] - MHA_Block.attention_weight[h,n,p].lb
                    # )
                    # MHA_Block.attention_constraints.add( # 
                    #     expr=  MHA_Block.tp_cc_mult_2[h, n, p]  == MHA_Block.attention_weight_x_cc_prime[h, n, p] * ( 1 -  MHA_Block.attention_weight_x_cc_prime[h, n, p])
                    # )
                    # MHA_Block.attention_constraints.add( 
                    #     expr=  (MHA_Block.tie_point_cc_prime[h, n, p] - MHA_Block.compatibility[h,n,p].lb ) * MHA_Block.tp_cc_mult_2[h, n, p]  == MHA_Block.tp_cc_mult_1[h, n, p]
                    # )
                    # # define tie point cc  secant
                    # MHA_Block.constr_concave_tp_sct.add( 
                    #     expr=  MHA_Block.tp_cc_sct[h, n, p] - MHA_Block.attention_weight[h,n,p].lb == 
                    #                                         + (MHA_Block.tp_cc_sct_mult_1_2[h, n, p] 
                    #                                            * (MHA_Block.compatibility[h,n,p]
                    #                                             - MHA_Block.compatibility[h,n,p].lb))
                    # )
                    # MHA_Block.constr_concave_tp_sct.add( 
                    #     expr=  MHA_Block.tp_cc_sct_mult_1_2[h, n, p] * MHA_Block.tp_cc_sct_mult_2[h, n, p] == MHA_Block.tp_cc_sct_mult_1[h, n, p] 
                    # )
                    # MHA_Block.constr_concave_tp_sct.add( 
                    #     expr=  MHA_Block.tp_cc_sct_mult_1[h, n, p] == MHA_Block.attention_weight[h,n,p].lb -  MHA_Block.attention_weight_x_cc[h, n, p]
                    # )
                    # MHA_Block.constr_concave_tp_sct.add( 
                    #     expr=  MHA_Block.tp_cc_sct_mult_2[h, n, p] == MHA_Block.compatibility[h,n,p].lb - MHA_Block.tie_point_cc[h, n, p]
                    # )
                    
                    
   
        # multihead attention output constraint
        for n in time_dim:
            for d in model_dims :
                if not b_o is None:
                    MHA_Block.attention_constraints.add(
                        expr= attention_output[n, d]
                        == sum(
                            (sum(
                            MHA_Block.attention_score[h, n, k] * MHA_Block.W_o[d,h, k]
                            for k in MHA_Block.k_dims
                             ) )
                        for h in MHA_Block.heads
                        
                        ) + MHA_Block.b_o[d]
                    )
                    attention_output[n, d].ub  = sum(sum( max(MHA_Block.attention_score[h, n, k].ub * MHA_Block.W_o[d,h, k], MHA_Block.attention_score[h, n, k].lb * MHA_Block.W_o[d,h, k]) for k in MHA_Block.k_dims) for h in MHA_Block.heads) + MHA_Block.b_o[d]
                    attention_output[n, d].lb  = sum(sum( min(MHA_Block.attention_score[h, n, k].ub * MHA_Block.W_o[d,h, k], MHA_Block.attention_score[h, n, k].lb * MHA_Block.W_o[d,h, k]) for k in MHA_Block.k_dims) for h in MHA_Block.heads) + MHA_Block.b_o[d]
                    
                    
                else:
                    MHA_Block.attention_constraints.add(
                        expr= attention_output[n, d]
                        == sum(
                            (sum(
                            MHA_Block.attention_score[h, n, k] * MHA_Block.W_o[d,h, k]
                            for k in MHA_Block.k_dims
                             ) )
                        for h in MHA_Block.heads
                        )
                    )
                    attention_output[n, d].ub  = sum(sum( max(MHA_Block.attention_score[h, n, k].ub * MHA_Block.W_o[d,h, k], MHA_Block.attention_score[h, n, k].lb * MHA_Block.W_o[d,h, k]) for k in MHA_Block.k_dims) for h in MHA_Block.heads)
                    attention_output[n, d].lb  = sum(sum( min(MHA_Block.attention_score[h, n, k].ub * MHA_Block.W_o[d,h, k], MHA_Block.attention_score[h, n, k].lb * MHA_Block.W_o[d,h, k]) for k in MHA_Block.k_dims) for h in MHA_Block.heads)
        
        # # activate softmax envelope constraints              
        # MHA_Block.activate_constraints = pyo.BuildAction(rule=activate_envelope_att)            
                
    # def add_attention_approx(self, input_var_name, W_q, W_k, W_v, W_o, b_q = None, b_k = None, b_v = None, b_o = None):
    #     """
    #     Multihead attention between each element of embedded sequence
        
    #     Exp function created using power series approximation (11 elements of power series). 
    #     This formulation avoids the pyomo solving error when calculating pyo.exp(pyo.Var())
    #     """
    #     if not hasattr( self.M, "attention_constraints"):
    #         MHA_Block.attention_constraints = pyo.ConstraintList()
            
    #     input_var = getattr( self.M, input_var_name)

    #     # define sets, vars
    #     MHA_Block.heads = pyo.RangeSet(1, self.d_H)
    #     MHA_Block.k_dims = pyo.RangeSet(1, self.d_k)

    #     W_q_dict = {
    #         (D, H, K): W_q[d][h][k]
    #         for d,D in enumerate(model_dims )
    #         for h,H in enumerate(MHA_Block.heads)
    #         for k,K in enumerate(MHA_Block.k_dims)
    #     }
    #     W_k_dict = {
    #         (D, H, K): W_k[d][h][k]
    #         for d,D in enumerate(self.M.model_dims)
    #         for h,H in enumerate(MHA_Block.heads)
    #         for k,K in enumerate(MHA_Block.k_dims)
    #     }
    #     W_v_dict = {
    #         (D, H, K): W_v[d][h][k]
    #         for d,D in enumerate(self.M.model_dims)
    #         for h,H in enumerate(MHA_Block.heads)
    #         for k,K in enumerate(MHA_Block.k_dims)
    #     }
    #     W_o_dict = {
    #         (D, H, K): W_o[h][k][d]
    #         for d,D in enumerate(self.M.model_dims)
    #         for h,H in enumerate(MHA_Block.heads)
    #         for k,K in enumerate(MHA_Block.k_dims)
    #     }
 
    #     MHA_Block.W_q = pyo.Param(self.M.model_dims, MHA_Block.heads, MHA_Block.k_dims, initialize=W_q_dict, mutable=False)
    #     MHA_Block.W_k = pyo.Param(self.M.model_dims, MHA_Block.heads, MHA_Block.k_dims, initialize=W_k_dict, mutable=False)
    #     MHA_Block.W_v = pyo.Param(self.M.model_dims, MHA_Block.heads, MHA_Block.k_dims, initialize=W_v_dict, mutable=False)
    #     MHA_Block.W_o = pyo.Param(self.M.model_dims,MHA_Block.heads, MHA_Block.k_dims, initialize=W_o_dict, mutable=False)
       
    #     if b_q:
    #         b_q_dict = {
    #                     (h, k): b_q[h-1][k-1]
    #                     for h in MHA_Block.heads
    #                     for k in MHA_Block.k_dims
    #                    }
    #         MHA_Block.b_q = pyo.Param(MHA_Block.heads, MHA_Block.k_dims, initialize=b_q_dict, mutable=False)
            
    #     if b_k:
    #         b_k_dict = {
    #                     (h, k): b_k[h-1][k-1]
    #                     for h in MHA_Block.heads
    #                     for k in MHA_Block.k_dims
    #                    }
    #         MHA_Block.b_k = pyo.Param(MHA_Block.heads, MHA_Block.k_dims, initialize=b_k_dict, mutable=False)
            
    #     if b_v: 
    #         b_v_dict = {
    #                     (h, k): b_v[h-1][k-1]
    #                     for h in MHA_Block.heads
    #                     for k in MHA_Block.k_dims
    #                    }
    #         MHA_Block.b_v = pyo.Param(MHA_Block.heads, MHA_Block.k_dims, initialize=b_v_dict, mutable=False)
            
    #     if b_o:
    #         b_o_dict = {(d): val for d, val in zip(self.M.model_dims, b_o) }
    #         MHA_Block.b_o = pyo.Param(self.M.model_dims, initialize=b_o_dict, mutable=False)
            

    #     MHA_Block.Q = pyo.Var(MHA_Block.heads, time_dim, MHA_Block.k_dims, within=pyo.Reals) 
    
    #     MHA_Block.K = pyo.Var(MHA_Block.heads, time_dim, MHA_Block.k_dims, within=pyo.Reals)
        
    #     MHA_Block.V = pyo.Var(MHA_Block.heads, time_dim, MHA_Block.k_dims, within=pyo.Reals) 
        
        
    #     #init_compatibility = {
    #                 #     (H, T, P): 1
    #                 #     for h,H in enumerate(MHA_Block.heads)
    #                 #     for n,T in enumerate(time_input)
    #                 #     for p,P in enumerate(time_input)
    #                 #    }
    #     MHA_Block.compatibility = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals) #, initialize=init_compatibility, bounds=(-10,10))  # sqrt(Q * K)
    #     MHA_Block.compatibility_exp = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.NonNegativeReals, bounds=(0,None)) # range: 0-->inf, initialize=init_compatibility_exp)
    #     MHA_Block.compatibility_exp_sum = pyo.Var(MHA_Block.heads, time_input) #, initialize=init_compatibility_sum)
    #     MHA_Block.compatibility_squ = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
    #     MHA_Block.compatibility_3 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
    #     MHA_Block.compatibility_4 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
    #     MHA_Block.compatibility_5 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
    #     MHA_Block.compatibility_6 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
    #     MHA_Block.compatibility_7 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
    #     MHA_Block.compatibility_8 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
    #     MHA_Block.compatibility_9 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
    #     MHA_Block.compatibility_10 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
    #     MHA_Block.compatibility_11 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        
    #     MHA_Block.compatibility_pos = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.NonNegativeReals, bounds=(0,None)) 
    #     MHA_Block.compatibility_neg = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.NonNegativeReals, bounds=(0,None)) 
        
    #     # MHA_Block.tie_point_cc = pyo.Var(MHA_Block.heads, time_input, time_input)
    #     # MHA_Block.tie_point_cv = pyo.Var(MHA_Block.heads, time_input, time_input)
    #     BigM_s = 1
    #     # BigM_t = 1
    #     MHA_Block.s_cc= pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Binary)
    #     MHA_Block.s_cv= pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Binary)
    #     # MHA_Block.t_cc= pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Binary)
    #     # MHA_Block.t_cv= pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Binary)
    #     # MHA_Block.attention_weight_cc = pyo.Var(MHA_Block.heads, time_input, time_input, bounds=(0,1))
    #     # MHA_Block.attention_weight_cv = pyo.Var(MHA_Block.heads, time_input, time_input, bounds=(0,1))

    #     MHA_Block.attention_weight = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.NonNegativeReals, bounds=(0,1))  # softmax ( (Q * K)/sqrt(d_k) )
    #     MHA_Block.attention_score = pyo.Var(
    #         MHA_Block.heads, time_input, MHA_Block.k_dims, within=pyo.Reals
    #     )  # softmax ( (Q * K)/sqrt(d_k) ) * V
    #     MHA_Block.attention_output = pyo.Var(
    #         time_input, self.M.model_dims, within=pyo.Reals
    #     )  # concat heads and linear transform

    #     for h in MHA_Block.heads:
    #         for n in time_input:
    #                 for k in MHA_Block.k_dims:
                        
    #                      # constraints for Query
    #                     if b_q:
    #                         MHA_Block.attention_constraints.add(
    #                         expr=MHA_Block.Q[h, n, k]
    #                         == sum(input_var[n,d] * MHA_Block.W_q[d, h, k] for d in self.M.model_dims) + MHA_Block.b_q[h,k] 
    #                         )  
    #                         #Add bounds
    #                         q_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in self.M.model_dims) + MHA_Block.b_q[h,k]
    #                         q_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in self.M.model_dims) + MHA_Block.b_q[h,k]
    #                         if q_bound_1 < q_bound_2: 
    #                             MHA_Block.Q[h, n, k].ub = q_bound_2
    #                             MHA_Block.Q[h, n, k].lb = q_bound_1
    #                         else:
    #                             MHA_Block.Q[h, n, k].ub = q_bound_1
    #                             MHA_Block.Q[h, n, k].lb = q_bound_2
                                
    #                         # print("bounds")
    #                         # print("--", input_var[n,'0'].lb, input_var[n,'0'].ub, MHA_Block.W_q['0', h, k])
    #                         # print("--", input_var[n,'1'].lb, input_var[n,'1'].ub, MHA_Block.W_q['1', h, k])
    #                         # print(q_bound_1, q_bound_2)
    #                         # print(MHA_Block.Q_pos[h, n, k].ub)

    #                     else: 
    #                         MHA_Block.attention_constraints.add(
    #                             expr=MHA_Block.Q[h, n, k]
    #                             == sum(input_var[n, d] * MHA_Block.W_q[d, h, k] for d in self.M.model_dims)
    #                         )
    #                         #Add bounds
    #                         q_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in self.M.model_dims)
    #                         q_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in self.M.model_dims)
    #                         if q_bound_1 < q_bound_2: 
    #                             MHA_Block.Q[h, n, k].ub = q_bound_2
    #                             MHA_Block.Q[h, n, k].lb = q_bound_1
    #                         else:
    #                             MHA_Block.Q[h, n, k].ub = q_bound_1
    #                             MHA_Block.Q[h, n, k].lb = q_bound_2
                              
    #                     # constraints for Key
    #                     if b_k:
    #                         MHA_Block.attention_constraints.add(
    #                         expr=MHA_Block.K[h, n, k]
    #                         == sum(input_var[n, d] * MHA_Block.W_k[d, h, k] for d in self.M.model_dims) + MHA_Block.b_k[h,k]
    #                         )  
    #                         #Add bounds
    #                         k_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in self.M.model_dims) + MHA_Block.b_k[h,k]
    #                         k_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in self.M.model_dims) + MHA_Block.b_k[h,k]
    #                         if k_bound_1 < k_bound_2: 
    #                             MHA_Block.K[h, n, k].ub = k_bound_2
    #                             MHA_Block.K[h, n, k].lb = k_bound_1
    #                         else:
    #                             MHA_Block.K[h, n, k].ub = k_bound_1
    #                             MHA_Block.K[h, n, k].lb = k_bound_2
                            
    #                     else: 
    #                         MHA_Block.attention_constraints.add(
    #                             expr=MHA_Block.K[h, n, k]
    #                             == sum(input_var[n, d] * MHA_Block.W_k[d, h, k] for d in self.M.model_dims)
    #                         )
    #                         #Add bounds
    #                         k_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in self.M.model_dims) 
    #                         k_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in self.M.model_dims) 
    #                         if k_bound_1 < k_bound_2: 
    #                             MHA_Block.K[h, n, k].ub = k_bound_2
    #                             MHA_Block.K[h, n, k].lb = k_bound_1
    #                         else:
    #                             MHA_Block.K[h, n, k].ub = k_bound_1
    #                             MHA_Block.K[h, n, k].lb = k_bound_2
                            
    #                     # constraints for Value    
    #                     if b_v:
    #                         MHA_Block.attention_constraints.add(
    #                         expr=MHA_Block.V[h, n, k]
    #                         == sum(input_var[n, d] * MHA_Block.W_v[d, h, k] for d in self.M.model_dims) + MHA_Block.b_v[h,k]
    #                         )  
    #                         #Add bounds
                            
    #                         v_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in self.M.model_dims) + MHA_Block.b_v[h,k]
    #                         v_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in self.M.model_dims) + MHA_Block.b_v[h,k]
    #                         if v_bound_1 < v_bound_2: 
    #                             MHA_Block.V[h, n, k].ub = v_bound_2
    #                             MHA_Block.V[h, n, k].lb = v_bound_1
    #                         else:
    #                             MHA_Block.V[h, n, k].ub = v_bound_1
    #                             MHA_Block.V[h, n, k].lb = v_bound_2
                            
    #                     else: 
    #                         MHA_Block.attention_constraints.add(
    #                             expr=MHA_Block.V[h, n, k]
    #                             == sum(input_var[n, d] * MHA_Block.W_v[d, h, k] for d in self.M.model_dims) 
    #                         )
    #                         #Add bounds     
    #                         v_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in self.M.model_dims)
    #                         v_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in self.M.model_dims)
    #                         if v_bound_1 < v_bound_2: 
    #                             MHA_Block.V[h, n, k].ub = v_bound_2
    #                             MHA_Block.V[h, n, k].lb = v_bound_1
    #                         else:
    #                             MHA_Block.V[h, n, k].ub = v_bound_1
    #                             MHA_Block.V[h, n, k].lb = v_bound_2

    #                     # attention score = sum(attention_weight * V)
    #                     MHA_Block.attention_constraints.add(
    #                         expr=MHA_Block.attention_score[h, n, k]
    #                         == sum(
    #                             MHA_Block.attention_weight[h, n, n2] * MHA_Block.V[h, n2, k]
    #                             for n2 in time_input
    #                         )
    #                     )

                        
    #                 for p in time_input:
    #                     # compatibility sqrt(Q * K) across all pairs of elements
    #                     scale = np.sqrt(self.d_k) 

    #                     MHA_Block.attention_constraints.add(
    #                         expr=MHA_Block.compatibility[h, n, p] *scale
    #                         == sum(MHA_Block.Q[h, n, k] * (MHA_Block.K[ h, p, k] )for k in MHA_Block.k_dims)
    #                     ) 
                        
                        
    # # # #                 # power series approx for EXP
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]**2 == MHA_Block.compatibility_squ[h, n, p] )#problem for gurobi
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_squ[h, n, p] == MHA_Block.compatibility_3[h, n, p] )
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_3[h, n, p] == MHA_Block.compatibility_4[h, n, p] )
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_4[h, n, p] == MHA_Block.compatibility_5[h, n, p] )
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_5[h, n, p] == MHA_Block.compatibility_6[h, n, p] )
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_6[h, n, p] == MHA_Block.compatibility_7[h, n, p] )
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_7[h, n, p] == MHA_Block.compatibility_8[h, n, p] )
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_8[h, n, p] == MHA_Block.compatibility_9[h, n, p] )
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_9[h, n, p] == MHA_Block.compatibility_10[h, n, p] )
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility[h, n, p]*MHA_Block.compatibility_10[h, n, p] == MHA_Block.compatibility_11[h, n, p] )
                        
    #                     MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility_exp[h, n, p] == 1
    #                                                 + MHA_Block.compatibility[h, n, p]
    #                                                 + (0.5*MHA_Block.compatibility_squ[h, n, p] ) 
    #                                                 + (0.166666667*MHA_Block.compatibility_3[h, n, p]) 
    #                                                 + (0.0416666667*MHA_Block.compatibility_4[h, n, p]) 
    #                                                 + (0.00833333333*MHA_Block.compatibility_5[h, n, p]) 
    #                                                 + (0.00138888889*MHA_Block.compatibility_6[h, n, p]) 
    #                                                 + (0.000198412698*MHA_Block.compatibility_7[h, n, p]) 
    #                                                 + (0.0000248015873*MHA_Block.compatibility_8[h, n, p]) 
    #                                                 + (0.00000275573192*MHA_Block.compatibility_9[h, n, p]) 
    #                                                 + (0.000000275573192*MHA_Block.compatibility_10[h, n, p])
    #                                                 + (0.0000000250521084*MHA_Block.compatibility_11[h, n, p])
    #                                                 )# pyo.exp() only seems to work for constant args and pow operator must be <= 2
                        
    #                 MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility_exp_sum[h, n] == sum(MHA_Block.compatibility_exp[h, n, p] for p in time_input))
                    
    #                 for n2 in time_input:

    #                     # attention weights softmax(compatibility)
    #                     MHA_Block.attention_constraints.add(
    #                         expr=MHA_Block.attention_weight[h, n, n2] * MHA_Block.compatibility_exp_sum[h, n]
    #                         == MHA_Block.compatibility_exp[h, n, n2]) 
                        
                        
    #                 # sum over softmax = 1    
    #                 MHA_Block.attention_constraints.add(
    #                     expr=sum(MHA_Block.attention_weight[h, n, n_prime] for n_prime in time_input) == 1
    #                 )
                   
                    
                    
    #         #Add bounds            
    #         for n in time_input:
    #             for p in time_input:
    #                 MHA_Block.attention_constraints.add(
    #                             expr=MHA_Block.compatibility[h,n,p] == MHA_Block.compatibility_pos[h,n,p] - MHA_Block.compatibility_neg[h,n,p] 
    #                         )
    #                 MHA_Block.compatibility_pos[h,n,p].ub = (1/scale ) * (sum( (MHA_Block.Q[h, n, k].ub)**2 for k in MHA_Block.k_dims)**0.5) * (sum( (MHA_Block.K[h, n, k].ub)**2 for k in MHA_Block.k_dims)**0.5)
    #                 MHA_Block.compatibility_neg[h,n,p].ub = MHA_Block.compatibility_pos[h,n,p].ub
    #                 MHA_Block.compatibility[h,n,p].ub = MHA_Block.compatibility_pos[h,n,p].ub
    #                 MHA_Block.compatibility[h,n,p].lb = -MHA_Block.compatibility_pos[h,n,p].ub
                    
    #                 MHA_Block.compatibility_exp[h,n,p].ub = math.exp(MHA_Block.compatibility[h,n,p].ub)
    #                 MHA_Block.compatibility_exp[h,n,p].lb = math.exp(MHA_Block.compatibility[h,n,p].lb)
                    
    #             MHA_Block.compatibility_exp_sum[h, n].ub = sum( MHA_Block.compatibility_exp[h,n,p].ub for p in time_input) 
    #             MHA_Block.compatibility_exp_sum[h, n].lb = sum( MHA_Block.compatibility_exp[h,n,p].lb for p in time_input) 
                
                    
    #             ##############-----------------------------------############    
    #             for p in time_input:    
    #                 MHA_Block.attention_weight[h, n, p].ub = MHA_Block.compatibility_exp[h,n,p].ub / (MHA_Block.compatibility_exp_sum[h, n].lb  - MHA_Block.compatibility_exp[h,n,p].lb + MHA_Block.compatibility_exp[h,n,p].ub  + 0.00000001)
    #                 MHA_Block.attention_weight[h, n, p].lb = MHA_Block.compatibility_exp[h,n,p].lb / (MHA_Block.compatibility_exp_sum[h, n].ub - MHA_Block.compatibility_exp[h,n,p].ub + MHA_Block.compatibility_exp[h,n,p].lb + 0.00000001)
    #                 # print("compat", MHA_Block.compatibility[h,n,p].ub)
    #                 # print("1:", MHA_Block.compatibility_exp[h,n,p].ub , MHA_Block.compatibility_exp_sum[h, n].ub)
    #                 # print(MHA_Block.attention_weight[h, n, p].ub)
    #                 # print("compat l", MHA_Block.compatibility[h,n,p].lb)
    #                 # print("2:", MHA_Block.compatibility_exp[h,n,p].lb , MHA_Block.compatibility_exp_sum[h, n].lb)
    #                 # print(MHA_Block.attention_weight[h, n, p].lb)
    #                 # Concave/convex envelope

    #                 # #f(x_UB) <= 0.5
    #                 # MHA_Block.attention_constraints.add(
    #                 #     expr= MHA_Block.attention_weight[h, n, n2].ub <= 0.5  + (BigM_s * MHA_Block.s_cv[h,n,p])
    #                 # )
    #                 # # f(x_UB) >= 0.5
    #                 # MHA_Block.attention_constraints.add(
    #                 #      expr= MHA_Block.compatibility_exp[h,n,p].ub/sum( MHA_Block.compatibility_exp[h,n,n2].ub for n2 in time_input) >= 0.5  - (BigM_s * MHA_Block.s_cv[h,n,p])
    #                 # )
                
            
                    
    #     # multihead attention output constraint
    #     for n in time_input:
    #         for d in self.M.model_dims:
    #             if b_o:
    #                 MHA_Block.attention_constraints.add(
    #                     expr=MHA_Block.attention_output[n, d]
    #                     == sum(
    #                         (sum(
    #                         MHA_Block.attention_score[h, n, k] * MHA_Block.W_o[d,h, k]
    #                         for k in MHA_Block.k_dims
    #                          ) )
    #                     for h in MHA_Block.heads
                        
    #                     ) + MHA_Block.b_o[d]
    #                 )
                    
                    
    #             else:
    #                 MHA_Block.attention_constraints.add(
    #                     expr=MHA_Block.attention_output[n, d]
    #                     == sum(
    #                         (sum(
    #                         MHA_Block.attention_score[h, n, k] * MHA_Block.W_o[d,h, k]
    #                         for k in MHA_Block.k_dims
    #                          ) )
    #                     for h in MHA_Block.heads
    #                     )
    #                 )
    #                 # MHA_Block.attention_output[n, d].ub  = (self.d_H * sum(MHA_Block.attention_score[h, n, k].ub * MHA_Block.W_o[d,h, k] for k in MHA_Block.k_dims))
    #                 # MHA_Block.attention_output[n, d].lb  = (self.d_H * sum(MHA_Block.attention_score[h, n, k].lb * MHA_Block.W_o[d,h, k] for k in MHA_Block.k_dims))
                
     # def add_FFN_2D(self, input_var_name, output_var_name, input_shape, model_parameters):
    #     input_var = getattr( self.M, input_var_name)

    #     # add new variable
    #     if not hasattr( self.M, output_var_name + "_NN_Block"):
    #         NN_name = output_var_name + "_NN_Block"
    #         setattr( self.M, NN_name, OmltBlock())
    #         NN_block = getattr( self.M, NN_name)
            
    #         setattr( self.M, output_var_name, pyo.Var(input_var.index_set(), within=pyo.Reals))
    #         output_var = getattr( self.M, output_var_name)
            
    #         setattr( self.M, output_var_name+"_constraints", pyo.ConstraintList())
    #         ffn_constraints = getattr( self.M, output_var_name+"_constraints")
    #     else:
    #         raise ValueError('Attempting to overwrite variable')
        
    #     ###### GET BOUNDS
    #     input_bounds={0: (-4,4), 1: (-4,4), 2: (-4,4), 3:(-4,4), 4:(-4,4), 5: (-4,4), 6: (-4,4), 7: (-4,4), 8: (-4,4), 9: (-4,4)} ### fix input bounds
    #     net_relu = OMLT_helper.weights_to_NetDef(output_var_name, input_shape, model_parameters, input_bounds)
    #     NN_block.build_formulation(ReluBigMFormulation(net_relu))
        
    #     # Set input constraints
    #     input_indices_len, input_indices_attr = self.__get_indices( input_var)
    #     if input_indices_len == 1:
    #         for i, index in  enumerate(input_indices_attr[0]):
    #             ffn_constraints.add(expr= input_var[index] == NN_block.inputs[i])
    #     elif input_indices_len == 2:
    #         for i, i_index in  enumerate(input_indices_attr[0]):
    #             for j, j_index in  enumerate(input_indices_attr[1]):
    #                 ffn_constraints.add(expr= input_var[i_index, j_index] == NN_block.inputs[j])
                    
                    
    #     # Set output constraints
    #     output_indices_len, output_indices_attr = self.__get_indices( output_var)
    #     if output_indices_len == 1:
    #         for i, index in  enumerate(output_indices_attr[0]):
    #             ffn_constraints.add(expr= output_var[index] == NN_block.outputs[i])
    #     elif output_indices_len == 2:
    #         for i, i_index in  enumerate(output_indices_attr[0]):
    #             for j, j_index in  enumerate(output_indices_attr[1]):
    #                 ffn_constraints.add(expr= output_var[i_index, j_index] == NN_block.outputs[j])
                      

    def add_residual_connection(self, input_1_name, input_2_name, output_var_name):
        # determine indices of input
        input_1 = getattr( self.M, input_1_name)
        input_2 = getattr( self.M, input_2_name)
        
        if input_1.is_indexed():
            set_var = input_1.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( self.M, set) )
                
            time_dim = indices[0]
            model_dims = indices[1]
        else:
            raise ValueError('Input value must be indexed (time, model_dim)')
        
        # create constraint list
        if not hasattr( self.M, "residual_constraints"):
            self.M.residual_constraints = pyo.ConstraintList()
        
        # add new variable
        if not hasattr( self.M, output_var_name):
            setattr( self.M, output_var_name, pyo.Var(time_dim, model_dims , within=pyo.Reals))
            residual_var = getattr( self.M, output_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')
        
        # Add constraints and bounds
        for n in time_dim:
            for d in model_dims :
                self.M.residual_constraints.add(expr= residual_var[n,d] == input_1[n,d] + input_2[n,d])
                try:
                    residual_var[n,d].ub == input_1[n,d].ub + input_2[n,d].ub
                    residual_var[n,d].lb == input_1[n,d].lb + input_2[n,d].lb
                except:
                    continue
                
   
    def get_fnn(self, input_var_name, output_var_name, nn_name, input_shape, model_parameters):
        
        input_var = getattr( self.M, input_var_name)
        # determine indices of input
        if input_var.is_indexed():
            set_var = input_var.index_set()
        else:
            raise ValueError('Input value must be indexed (time, model_dim)')
        
        # add new variable
        if not hasattr( self.M, output_var_name + "_NN_Block"):
            
            setattr( self.M, output_var_name, pyo.Var(set_var, within=pyo.Reals))
            output_var = getattr( self.M, output_var_name)
            
            setattr( self.M, output_var_name+"_constraints", pyo.ConstraintList())
            ffn_constraints = getattr( self.M, output_var_name+"_constraints")
        else:
            raise ValueError('Attempting to overwrite variable')
        
        nn= GUROBI_ML_helper.weights_to_NetDef(output_var_name, nn_name, input_shape, model_parameters)
       
        return nn, input_var, output_var
            
        
    def add_avg_pool(self, input_var_name, output_var_name):
        # get input
        input_var = getattr( self.M, input_var_name)
        
        # determine indices of input
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( self.M, set) )
                
            time_dim = indices[0]
            model_dims = indices[1]
        else:
            raise ValueError('Input value must be indexed (time, model_dim)')

        # add new variable
        if not hasattr( self.M, output_var_name):
            setattr( self.M, "avg_pool_constr_"+output_var_name, pyo.ConstraintList())
            constraints = getattr( self.M, "avg_pool_constr_"+output_var_name) 
            
            setattr( self.M, output_var_name, pyo.Var(model_dims , within=pyo.Reals))
            output_var = getattr( self.M, output_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')


        for d in model_dims : 
            constraints.add(expr= output_var[d] * self.N == sum(input_var[t,d] for t in time_dim))
            
            try:
                output_var[d].ub  == sum(input_var[t,d].ub for t in time_dim) / self.N
                output_var[d].lb  == sum(input_var[t,d].lb for t in time_dim) / self.N
            except:
                continue
    def __McCormick_bb(self, w, x, y):
        """ Add McMcormick envelope for bilinear variable w = x * y"""
        
        if not hasattr( self.M, "mccormick_bb_constr_list"):
            setattr( self.M, "mccormick_bb_constr_list", pyo.ConstraintList())
            
        constraints = getattr( self.M, "mccormick_bb_constr_list")    
        constraints.add( expr= w >= (x.lb * y) + (x*y.lb) - (x.lb*y.lb))
        constraints.add( expr= w >= (x.ub * y) + (x*y.ub) - (x.ub*y.ub))
        constraints.add( expr= w <= (x.ub * y) + (x*y.lb) - (x.ub*y.lb))
        constraints.add( expr= w <= (x * y.ub) + (x.lb*y) - (x.lb*y.ub))

