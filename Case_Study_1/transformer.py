import pyomo.environ as pyo
import numpy as np
import math
from pyomo import dae
import json
import os
from omlt import OmltBlock
from omlt.neuralnet import NetworkDefinition, ReluBigMFormulation
from omlt.io.keras import keras_reader
import omlt
import helpers.OMLT_helper 
import helpers.GUROBI_ML_helper

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

def activate_envelope_att(model):
        model.constr_convex.deactivate()
        model.constr_concave.deactivate() 
        model.constr_convex_tp.deactivate()
        model.constr_convex_tp_sct.deactivate()
        model.constr_concave_tp.deactivate()
        model.constr_concave_tp_sct.deactivate()

        
        if model.s_cv == 0: # --> convex region onlyt
            model.constr_convex.activate()
        elif model.s_cc == 0: # --> concave region only
            model.constr_concave.activate() 
        else: # in both regions
            if model.t_cv == 0: # --> if x <= x_cv_tiepoint -->convex region
                model.constr_convex_tp.activate()
            elif model.t_cv == 1: # -->concave region
                model.constr_convex_tp_sct.activate()
                
            if model.t_cc == 0: # --> if x >= x_cc_tiepoint -->concave region
                model.constr_concave_tp.activate()
            elif model.t_cc == 1:# --> convex region
                model.constr_concave_tp_sct.activate()

class Transformer:
    def __init__(self, model, config_file, name):
        
        setattr(model, name, pyo.Block())
        self.Transformer_Block  = getattr(model, name)
        
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
        
    def add_input(self, input_var, name):
        setattr(self.Transformer_Block, name, pyo.Var(input_var.index_set(), within=pyo.Reals))
        transformer_input = getattr(self.Transformer_Block, name)
        
        self.Transformer_Block.transformer_input_constraints = pyo.ConstraintList()
        for index in input_var.index_set():
            self.Transformer_Block.transformer_input_constraints.add( input_var[index] == transformer_input[index])
            
        
    def embed_input(self, model, input_var_name, embed_var_name, W_emb=None, b_emb = None):
        """
        Embed the feature dimensions of input
        """
        if not hasattr(self.Transformer_Block, "embed_constraints"):
            self.Transformer_Block.embed_constraints = pyo.ConstraintList()
            
        input_var = getattr(self.Transformer_Block, input_var_name)
        
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( model, set) )
                
            # define embedding var
            if not hasattr(self.Transformer_Block, embed_var_name):
                setattr(self.Transformer_Block, embed_var_name, pyo.Var(indices[0], model.model_dims, within=pyo.Reals, initialize= 0))
                embed_var = getattr(self.Transformer_Block, embed_var_name)   
            else:
                raise ValueError('Attempting to overwrite variable')
            
            if W_emb is None:
                for index, index_input in zip( embed_var.index_set(), set_var):
                    self.Transformer_Block.embed_constraints.add(embed_var[index] == input_var[index_input])
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
                    (indices[1].at(s+1),model.model_dims.at(d+1)): W_emb[s][d]
                    for s in range(len(indices[1]))
                    for d in range(len(model.model_dims))
                }
                setattr(self.Transformer_Block, embed_var_name+"_W_emb", pyo.Param(indices[1], model.model_dims, initialize=W_emb_dict))
                W_emb= getattr(self.Transformer_Block, embed_var_name+"_W_emb")   
                
                if not b_emb is None:
                    # Create bias variable
                    b_emb_dict = {
                        (model.model_dims.at(d+1)): b_emb[d]
                        for d in range(len(model.model_dims))
                    }
                    
                    setattr(self.Transformer_Block, embed_var_name+"_b_emb", pyo.Param(model.model_dims, initialize=b_emb_dict))
                    b_emb= getattr(self.Transformer_Block, embed_var_name+"_b_emb")  
                
                    for d in model.model_dims:
                        for t in indices[0]:
                            self.Transformer_Block.embed_constraints.add(embed_var[t, d] 
                                                    == sum(input_var[t,s] * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                                                    )
                            if isinstance(input_var, pyo.Var):
                                try:
                                    embed_var[t, d].ub = sum(input_var[t,s].ub * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                                    embed_var[t, d].lb = sum(input_var[t,s].lb * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                                except:
                                    continue
                            elif isinstance(input_var, pyo.Param):
                                embed_var[t, d].ub = sum(input_var[t,s] * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                                embed_var[t, d].lb = sum(input_var[t,s] * W_emb[s,d] for s in indices[1]) +  b_emb[d]
                else:
                    for d in model.model_dims:
                        for t in indices[0]:
                            self.Transformer_Block.embed_constraints.add(embed_var[t, d] 
                                                    == sum(input_var[t,s] * W_emb[s,d] for s in indices[1])
                                                    )
                            if isinstance(input_var, pyo.Var):
                                try:
                                    embed_var[t, d].ub = sum(input_var[t,s].ub * W_emb[s,d] for s in indices[1])
                                    embed_var[t, d].lb = sum(input_var[t,s].lb * W_emb[s,d] for s in indices[1])
                                except:
                                    continue
                            elif isinstance(input_var, pyo.Param):
                                embed_var[t, d].ub = sum(input_var[t,s] * W_emb[s,d] for s in indices[1])
                                embed_var[t, d].lb = sum(input_var[t,s] * W_emb[s,d] for s in indices[1])
        else:
            raise ValueError('Input value must be indexed')
           

    def add_layer_norm(self, model, input_var_name, layer_norm_var_name, gamma= None, beta = None, std=None):  # non-linear
        """
        Normalization over the sequennce of input
        """
        if not hasattr(self.Transformer_Block, "layer_norm_constraints"):
            self.Transformer_Block.layer_norm_constraints = pyo.ConstraintList()
        
        input_var = getattr(self.Transformer_Block, input_var_name)
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( model, set) )
            
            time_input = indices[0]
        else:
            ValueError('Input value must be indexed')
        
        # Initialize variables
        if not hasattr(self.Transformer_Block, layer_norm_var_name):
            # define layer norm output var
            setattr(self.Transformer_Block, layer_norm_var_name, pyo.Var(time_input, model.model_dims, within=pyo.Reals))
            layer_norm_var = getattr(self.Transformer_Block, layer_norm_var_name)
            
            # define calculation variables
            sum_name = 'sum_'+ layer_norm_var_name
            setattr(self.Transformer_Block, sum_name, pyo.Var(time_input, within=pyo.Reals))
            sum_t = getattr(self.Transformer_Block, sum_name)
            
            variance_name = 'variance_'+ layer_norm_var_name
            setattr(self.Transformer_Block, variance_name, pyo.Var(time_input, within=pyo.Reals))
            variance = getattr(self.Transformer_Block, variance_name)
            
            div_name = 'div_'+ layer_norm_var_name
            setattr(self.Transformer_Block, div_name, pyo.Var(time_input, model.model_dims, within=pyo.Reals))
            div = getattr(self.Transformer_Block, div_name)
            
            denominator_name = 'denominator_'+ layer_norm_var_name
            setattr(self.Transformer_Block, denominator_name, pyo.Var(time_input, within=pyo.Reals))
            denominator = getattr(self.Transformer_Block, denominator_name)
            
            denominator_abs_name = 'denominator_abs_'+ layer_norm_var_name
            setattr(self.Transformer_Block, denominator_abs_name, pyo.Var(time_input, within=pyo.NonNegativeReals, bounds=(0,None)))
            denominator_abs = getattr(self.Transformer_Block, denominator_abs_name)
            
            numerator_name = 'numerator_'+ layer_norm_var_name
            setattr(self.Transformer_Block, numerator_name, pyo.Var(time_input, model.model_dims, within=pyo.Reals))
            numerator = getattr(self.Transformer_Block, numerator_name)

            numerator_scaled_name = 'numerator_scaled_'+ layer_norm_var_name
            setattr(self.Transformer_Block, numerator_scaled_name, pyo.Var(time_input, model.model_dims, within=pyo.Reals))
            numerator_scaled = getattr(self.Transformer_Block, numerator_scaled_name)
            
            numerator_squared_name = 'numerator_squared_'+ layer_norm_var_name
            setattr(self.Transformer_Block, numerator_squared_name, pyo.Var(time_input, model.model_dims, within=pyo.Reals, bounds=(0,None)))
            numerator_squared = getattr(self.Transformer_Block, numerator_squared_name)
              
            numerator_squared_sum_name = 'numerator_squared_sum_'+ layer_norm_var_name
            setattr(self.Transformer_Block, numerator_squared_sum_name, pyo.Var(time_input, within=pyo.Reals, bounds=(0,None)))
            numerator_squared_sum = getattr(self.Transformer_Block, numerator_squared_sum_name)
            
            
            
        else:
            raise ValueError('Attempting to overwrite variable')

        # Add constraints for layer norm
        if self.d_model == 1:
            return
            
        for t in time_input: 
            self.Transformer_Block.layer_norm_constraints.add(expr= sum_t[t] == sum(input_var[t, d] for d in model.model_dims) )
            
            # Constraints for each element in sequence
            for d in model.model_dims:  
                self.Transformer_Block.layer_norm_constraints.add(expr= numerator[t,d] == input_var[t, d] - ((1/ self.d_model) *sum_t[t]))
                self.Transformer_Block.layer_norm_constraints.add(expr= numerator_squared[t,d] == numerator[t,d]**2)
                
                self.Transformer_Block.layer_norm_constraints.add(expr= numerator_squared_sum[t] == sum(numerator_squared[t,d_prime] for d_prime in model.model_dims))
                self.Transformer_Block.layer_norm_constraints.add(expr= variance[t] * self.d_model == numerator_squared_sum[t])
                
                #self.Transformer_Block.layer_norm_constraints.add(expr= denominator[t] **2 == variance[t] )     ##IF SCIP SOLVER
                ## FOR SCIP or GUROBI: determine abs(denominator)
                self.Transformer_Block.layer_norm_constraints.add(expr= denominator[t] <= denominator_abs[t]) 
                self.Transformer_Block.layer_norm_constraints.add(expr= denominator[t]*denominator[t] == denominator_abs[t] * denominator_abs[t]) 
                
                self.Transformer_Block.layer_norm_constraints.add(expr= variance[t] == (denominator[t] * denominator_abs[t] ) )
                # if std:
                #     denominator[t].ub = std
                #     denominator[t].lb = -std
                #     denominator_abs[t].ub = std
                    
                
                self.Transformer_Block.layer_norm_constraints.add(expr= div[t,d] * denominator[t] == numerator[t,d] )
                div[t,d].ub = 4
                div[t,d].lb = -4
                
                if gamma and beta:
                    self.Transformer_Block.layer_norm_constraints.add(expr= numerator_scaled[t,d] == getattr(model, gamma)[d] * div[t,d])
                    self.Transformer_Block.layer_norm_constraints.add(expr=layer_norm_var[t, d] == numerator_scaled[t,d] + getattr(model, beta)[d])
                    layer_norm_var[t, d].ub = getattr(model, beta)[d] + 4*getattr(model, gamma)[d]
                    layer_norm_var[t, d].lb = getattr(model, beta)[d] - 4*getattr(model, gamma)[d]
                    
                else: 
                    self.Transformer_Block.layer_norm_constraints.add(expr= numerator_scaled[t,d] == div[t,d])
                    self.Transformer_Block.layer_norm_constraints.add(expr=layer_norm_var[t, d] == numerator_scaled[t,d])
                    layer_norm_var[t, d].ub = 4
                    layer_norm_var[t, d].lb = -4
                    
                #Add bounds
                try:
                
                    if input_var[t, d].ub and input_var[t, d].lb:
                        numerator[t,d].ub = input_var[t, d].ub - min(input_var[t,:].lb)
                        # numerator[t,d].lb = input_var[t, d].lb - max(input_var[t,:].ub) 
                        # numerator_squared[t,d].ub = max(numerator[t,d].ub**2, numerator[t,d].lb**2) 
                        numerator_squared[t,d].lb = 0
                        
                        if not std :
                            denominator[t].ub = abs( max(input_var[t,:].ub) - min(input_var[t,:].lb)) 
                            denominator[t].lb = - abs( max(input_var[t,:].ub) - min(input_var[t,:].lb))
                            denominator_abs[t].ub = abs( max(input_var[t,:].ub) - min(input_var[t,:].lb)) 
                except:
                    raise ValueError('Supply bounds to input variable')
                
                numerator_squared[t,d].lb = 0
            # if input_var[t, d].ub and input_var[t, d].lb:
            #     numerator_s quared_sum[t].ub = sum( (numerator_squared[t,d_prime].ub) for d_prime in model.model_dims) 
            numerator_squared_sum[t].lb = 0
            
    
    def add_attention(self, model, input_var_name, output_var_name, W_q, W_k, W_v, W_o, b_q = None, b_k = None, b_v = None, b_o = None):
        """
        Multihead attention between each element of embedded sequence
        
        Uses the pyo.exp() function to calculate softmax. 
        This is compatible with gurobi which allows for the outer approximation of the function to be calculated
        """
            
        input_var = getattr(self.Transformer_Block, input_var_name)
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( model, set) )
            
            time_input = indices[0]
        else:
            ValueError('Input value must be indexed')
        
        if not hasattr(self.Transformer_Block, output_var_name):
            setattr(self.Transformer_Block, output_var_name, pyo.Var(time_input, model.model_dims, within=pyo.Reals))
            attention_output = getattr(self.Transformer_Block, output_var_name)
            
            setattr(self.Transformer_Block, "Block_"+output_var_name, pyo.Block())
            MHA_Block  = getattr(self.Transformer_Block, "Block_"+output_var_name)
            
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
            for d,D in enumerate(model.model_dims)
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.k_dims)
        }
        W_k_dict = {
            (D, H, K): W_k[d][h][k]
            for d,D in enumerate(model.model_dims)
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.k_dims)
        }
        W_v_dict = {
            (D, H, K): W_v[d][h][k]
            for d,D in enumerate(model.model_dims)
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.k_dims)
        }
        W_o_dict = {
            (D, H, K): W_o[h][k][d]
            for d,D in enumerate(model.model_dims)
            for h,H in enumerate(MHA_Block.heads)
            for k,K in enumerate(MHA_Block.k_dims)
        }
 
        MHA_Block.W_q = pyo.Param(model.model_dims,MHA_Block.heads,MHA_Block.k_dims, initialize=W_q_dict, mutable=False)
        MHA_Block.W_k = pyo.Param(model.model_dims,MHA_Block.heads,MHA_Block.k_dims, initialize=W_k_dict, mutable=False)
        MHA_Block.W_v = pyo.Param(model.model_dims,MHA_Block.heads,MHA_Block.k_dims, initialize=W_v_dict, mutable=False)
        MHA_Block.W_o = pyo.Param(model.model_dims,MHA_Block.heads,MHA_Block.k_dims, initialize=W_o_dict, mutable=False)
        
        if b_q:
            b_q_dict = {
                        (h, k): b_q[h-1][k-1]
                        for h in MHA_Block.heads
                        for k in MHA_Block.k_dims
                       }
            MHA_Block.b_q = pyo.Param(MHA_Block.heads, MHA_Block.k_dims, initialize=b_q_dict, mutable=False)
            
        if b_k:
            b_k_dict = {
                        (h, k): b_k[h-1][k-1]
                        for h in MHA_Block.heads
                        for k in MHA_Block.k_dims
                       }
            MHA_Block.b_k = pyo.Param(MHA_Block.heads, MHA_Block.k_dims, initialize=b_k_dict, mutable=False)
            
        if b_v: 
            b_v_dict = {
                        (h, k): b_v[h-1][k-1]
                        for h in MHA_Block.heads
                        for k in MHA_Block.k_dims
                       }
            MHA_Block.b_v = pyo.Param(MHA_Block.heads, MHA_Block.k_dims, initialize=b_v_dict, mutable=False)
            
        if b_o:
            b_o_dict = {(d): val for d, val in zip(model.model_dims, b_o) }
            MHA_Block.b_o = pyo.Param(model.model_dims, initialize=b_o_dict, mutable=False)
            

        MHA_Block.Q = pyo.Var(MHA_Block.heads, time_input, MHA_Block.k_dims, within=pyo.Reals) 
        MHA_Block.K = pyo.Var(MHA_Block.heads, time_input, MHA_Block.k_dims, within=pyo.Reals)
        MHA_Block.V = pyo.Var(MHA_Block.heads, time_input, MHA_Block.k_dims, within=pyo.Reals) 

        MHA_Block.compatibility = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals) 
        MHA_Block.compatibility_pos = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.NonNegativeReals, bounds=(0,None)) 
        MHA_Block.compatibility_neg = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.NonNegativeReals, bounds=(0,None)) 
        
        MHA_Block.compatibility_exp = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.NonNegativeReals, bounds=(0,None)) # range: 0-->inf, initialize=init_compatibility_exp)
        MHA_Block.compatibility_exp_sum = pyo.Var(MHA_Block.heads, time_input, within=pyo.NonNegativeReals, bounds=(0,None)) #, initialize=init_compatibility_sum)
        MHA_Block.tie_point_cc = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tie_point_cv = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tie_point_cc_prime = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tie_point_cv_prime = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tp_cv_mult_1 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tp_cv_mult_2 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tp_cc_mult_1 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tp_cc_mult_2 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        
        BigM_s = 0.5
        MHA_Block.sct = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals, bounds=(0,1))
        
        MHA_Block.s_cv= pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Binary)
        MHA_Block.t_cv= pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Binary)
        
        MHA_Block.s_cc= pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Binary)
        MHA_Block.t_cc= pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Binary)
        
        MHA_Block.tp_cv =pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Binary)
        MHA_Block.tp_cc =pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Binary)

        MHA_Block.attention_weight_cc = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals, bounds=(0,1))
        MHA_Block.attention_weight_x_cc_prime = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals, bounds=(0,1))
        MHA_Block.attention_weight_x_cc= pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals, bounds=(0,1))
        
        MHA_Block.attention_weight_cv = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals, bounds=(0,1))
        MHA_Block.attention_weight_x_cv_prime = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals, bounds=(0,1))
        MHA_Block.attention_weight_x_cv = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals, bounds=(0,1))
        
        MHA_Block.attention_weight = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals, bounds=(0,1))  # softmax ( (Q * K)/sqrt(d_k) )
        MHA_Block.tp_cv_sct = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals, bounds=(0,1))
        MHA_Block.tp_cv_sct_mult_1 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tp_cv_sct_mult_2 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tp_cv_sct_mult_1_2 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        
        
        MHA_Block.tp_cc_sct = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals, bounds=(0,1))
        MHA_Block.tp_cc_sct_mult_1 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tp_cc_sct_mult_2 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        MHA_Block.tp_cc_sct_mult_1_2 = pyo.Var(MHA_Block.heads, time_input, time_input, within=pyo.Reals)
        
        
        MHA_Block.attention_score = pyo.Var(
            MHA_Block.heads, time_input, MHA_Block.k_dims, within=pyo.Reals
        )  # softmax ( (Q * K)/sqrt(d_k) ) * V
        
        for h in MHA_Block.heads:
            for n in time_input:
                    for k in MHA_Block.k_dims:
                        
                         # constraints for Query
                        if b_q:
                            MHA_Block.attention_constraints.add(
                            expr=MHA_Block.Q[h, n, k]
                            == sum(input_var[n,d] * MHA_Block.W_q[d, h, k] for d in model.model_dims) + MHA_Block.b_q[h,k] 
                            )  
                            #Add bounds
                            try:
                                q_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model.model_dims) + MHA_Block.b_q[h,k]
                            except:
                                raise ValueError('Supply bounds to input variable')
                            #q_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model.model_dims) + MHA_Block.b_q[h,k]
                            q_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model.model_dims) + MHA_Block.b_q[h,k]
                            if q_bound_1 < q_bound_2: 
                                MHA_Block.Q[h, n, k].ub = q_bound_2
                                MHA_Block.Q[h, n, k].lb = q_bound_1
                            else:
                                MHA_Block.Q[h, n, k].ub = q_bound_1
                                MHA_Block.Q[h, n, k].lb = q_bound_2


                        else: 
                            MHA_Block.attention_constraints.add(
                                expr=MHA_Block.Q[h, n, k]
                                == sum(input_var[n, d] * MHA_Block.W_q[d, h, k] for d in model.model_dims)
                            )
                            #Add bounds
                            q_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model.model_dims)
                            q_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model.model_dims)
                            if q_bound_1 < q_bound_2: 
                                MHA_Block.Q[h, n, k].ub = q_bound_2
                                MHA_Block.Q[h, n, k].lb = q_bound_1
                            else:
                                MHA_Block.Q[h, n, k].ub = q_bound_1
                                MHA_Block.Q[h, n, k].lb = q_bound_2
                              
                        # constraints for Key
                        if b_k:
                            MHA_Block.attention_constraints.add(
                            expr=MHA_Block.K[h, n, k]
                            == sum(input_var[n, d] * MHA_Block.W_k[d, h, k] for d in model.model_dims) + MHA_Block.b_k[h,k]
                            )  
                            #Add bounds
                            k_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in model.model_dims) + MHA_Block.b_k[h,k]
                            k_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in model.model_dims) + MHA_Block.b_k[h,k]
                            if k_bound_1 < k_bound_2: 
                                MHA_Block.K[h, n, k].ub = k_bound_2
                                MHA_Block.K[h, n, k].lb = k_bound_1
                            else:
                                MHA_Block.K[h, n, k].ub = k_bound_1
                                MHA_Block.K[h, n, k].lb = k_bound_2
                            
                        else: 
                            MHA_Block.attention_constraints.add(
                                expr=MHA_Block.K[h, n, k]
                                == sum(input_var[n, d] * MHA_Block.W_k[d, h, k] for d in model.model_dims)
                            )
                            #Add bounds
                            k_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in model.model_dims) 
                            k_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in model.model_dims) 
                            if k_bound_1 < k_bound_2: 
                                MHA_Block.K[h, n, k].ub = k_bound_2
                                MHA_Block.K[h, n, k].lb = k_bound_1
                            else:
                                MHA_Block.K[h, n, k].ub = k_bound_1
                                MHA_Block.K[h, n, k].lb = k_bound_2
                            
                        # constraints for Value    
                        if b_v:
                            MHA_Block.attention_constraints.add(
                            expr=MHA_Block.V[h, n, k]
                            == sum(input_var[n, d] * MHA_Block.W_v[d, h, k] for d in model.model_dims) + MHA_Block.b_v[h,k]
                            )  
                            #Add bounds
                            
                            v_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in model.model_dims) + MHA_Block.b_v[h,k]
                            v_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in model.model_dims) + MHA_Block.b_v[h,k]
                            if v_bound_1 < v_bound_2: 
                                MHA_Block.V[h, n, k].ub = v_bound_2
                                MHA_Block.V[h, n, k].lb = v_bound_1
                            else:
                                MHA_Block.V[h, n, k].ub = v_bound_1
                                MHA_Block.V[h, n, k].lb = v_bound_2
                            
                        else: 
                            MHA_Block.attention_constraints.add(
                                expr=MHA_Block.V[h, n, k]
                                == sum(input_var[n, d] * MHA_Block.W_v[d, h, k] for d in model.model_dims) 
                            )
                            #Add bounds     
                            v_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in model.model_dims)
                            v_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in model.model_dims)
                            if v_bound_1 < v_bound_2: 
                                MHA_Block.V[h, n, k].ub = v_bound_2
                                MHA_Block.V[h, n, k].lb = v_bound_1
                            else:
                                MHA_Block.V[h, n, k].ub = v_bound_1
                                MHA_Block.V[h, n, k].lb = v_bound_2

                        # attention score = sum(attention_weight * V)
                        MHA_Block.attention_constraints.add(
                            expr=MHA_Block.attention_score[h, n, k]
                            == sum(
                                MHA_Block.attention_weight[h, n, n2] * MHA_Block.V[h, n2, k]
                                for n2 in time_input
                            )
                        )
                        
                        
                        
                    for p in time_input:
                        # compatibility sqrt(Q * K) across all pairs of elements
                        scale = np.sqrt(self.d_k) 
                        MHA_Block.attention_constraints.add(
                            expr=MHA_Block.compatibility[h, n, p] *scale
                            == sum(MHA_Block.Q[h, n, k] * (MHA_Block.K[ h, p, k] )for k in MHA_Block.k_dims)
                        )  
                        
                        # exp(compatibility)
                        MHA_Block.attention_constraints.add(expr= pyo.exp(MHA_Block.compatibility[h,n,p]) == MHA_Block.compatibility_exp[h, n, p] )
                        
                        
                        
                    # sum over exp(compatbility)
                    MHA_Block.attention_constraints.add(expr= MHA_Block.compatibility_exp_sum[h, n] == sum(MHA_Block.compatibility_exp[h, n, p] for p in time_input))
                    
                    #sum over softmax = 1    
                    MHA_Block.attention_constraints.add(
                        expr=sum(MHA_Block.attention_weight[h, n, n_prime] for n_prime in time_input) == 1
                    )
                    
                    for n2 in time_input:

                        #attention weights softmax(compatibility)   
                        MHA_Block.attention_constraints.add(
                            expr=MHA_Block.attention_weight[h, n, n2] * MHA_Block.compatibility_exp_sum[h, n]
                            == MHA_Block.compatibility_exp[h, n, n2]) 
                        
                    
            #Add bounds            
            for n in time_input:
                for p in time_input:
                    MHA_Block.attention_constraints.add(
                                expr=MHA_Block.compatibility[h,n,p] == MHA_Block.compatibility_pos[h,n,p] - MHA_Block.compatibility_neg[h,n,p] 
                            )
                    MHA_Block.compatibility_pos[h,n,p].ub = (1/scale ) * (sum( max((MHA_Block.Q[h, n, k].ub)**2, (MHA_Block.Q[h, n, k].lb)**2) for k in MHA_Block.k_dims)**0.5) * (sum( max((MHA_Block.K[h, n, k].ub)**2, (MHA_Block.K[h, n, k].lb)**2) for k in MHA_Block.k_dims)**0.5)
                    MHA_Block.compatibility_neg[h,n,p].ub = MHA_Block.compatibility_pos[h,n,p].ub
                    MHA_Block.compatibility[h,n,p].ub = MHA_Block.compatibility_pos[h,n,p].ub
                    MHA_Block.compatibility[h,n,p].lb = -MHA_Block.compatibility_pos[h,n,p].ub
                    
                    MHA_Block.compatibility_exp[h,n,p].ub = math.exp(MHA_Block.compatibility[h,n,p].ub)
                    MHA_Block.compatibility_exp[h,n,p].lb = max(0, 1 + MHA_Block.compatibility[h,n,p].lb)
                    
                MHA_Block.compatibility_exp_sum[h, n].ub = sum( MHA_Block.compatibility_exp[h,n,p].ub for p in time_input) # exp --> pos
                MHA_Block.compatibility_exp_sum[h, n].lb = max(0, sum( MHA_Block.compatibility_exp[h,n,p].lb for p in time_input))
                
                for k in MHA_Block.k_dims:
                    MHA_Block.attention_score[h, n, k].ub = sum(MHA_Block.V[h, n2, k].ub for n2 in time_input)
                    MHA_Block.attention_score[h, n, k].lb = min(0, sum(MHA_Block.V[h, n2, k].lb for n2 in time_input))
                      
                # ##############-----------------------------------############    
                for p in time_input:    
                    MHA_Block.attention_weight[h, n, p].ub = MHA_Block.compatibility_exp[h,n,p].ub / (MHA_Block.compatibility_exp_sum[h, n].lb  - MHA_Block.compatibility_exp[h,n,p].lb + MHA_Block.compatibility_exp[h,n,p].ub  + 0.00000001)
                    MHA_Block.attention_weight[h, n, p].lb = max(0, MHA_Block.compatibility_exp[h,n,p].lb / (MHA_Block.compatibility_exp_sum[h, n].ub - MHA_Block.compatibility_exp[h,n,p].ub + MHA_Block.compatibility_exp[h,n,p].lb + 0.00000001))
                    
                    # f(x) >= f_cv(x): attention weight >= convex envelope
                    MHA_Block.attention_constraints.add(
                        MHA_Block.attention_weight[h, n, p]  >= MHA_Block.attention_weight_cv[h, n, p]
                    )
                    # f(x) <= f_cc(x): attention weight <= concave envelope
                    MHA_Block.attention_constraints.add(
                        MHA_Block.attention_weight[h, n, p]  <= MHA_Block.attention_weight_cc[h, n, p]
                    )
           
                    # Constraints for Concave/convex envelope
                    # set convex aux var -- s=0: f(x_UB) <= 0.5 --> convex zone, s=1: f(x_UB) >= 0.5 --> concave zone
                    MHA_Block.attention_constraints.add(
                        expr= MHA_Block.attention_weight[h, n, p].ub <= 0.5  + (BigM_s * MHA_Block.s_cv[h,n,p])
                    )
                    
                    # MHA_Block.attention_constraints.add(
                    #     expr= MHA_Block.attention_weight[h, n, p].ub >= 0.5  - (BigM_s * (1 - MHA_Block.s_cv[h,n,p]))
                    # )
                    MHA_Block.attention_constraints.add(
                        expr= MHA_Block.attention_weight[h, n, p].ub >= BigM_s *  MHA_Block.s_cv[h,n,p]
                    )

                    # set convex aux var -- f(x_LB) <= 0.5 --> convex zone else f(x_LB) >= 0.5 --> concave zone
                    MHA_Block.attention_constraints.add(
                        expr= MHA_Block.attention_weight[h, n, p].lb >= 0.5 - (BigM_s *  (MHA_Block.s_cc[h,n,p]))
                    )
                    MHA_Block.attention_constraints.add(
                        expr= (BigM_s * MHA_Block.s_cc[h,n,p]) <= 0.5 + BigM_s - MHA_Block.attention_weight[h, n, p].lb
                    )
                    
                    # # sct(x)
                    A = ((MHA_Block.attention_weight[h, n, p].ub - MHA_Block.attention_weight[h, n, p].lb) / (MHA_Block.compatibility[h,n,p].ub - MHA_Block.compatibility[h,n,p].lb )) 
                    b = ( (MHA_Block.compatibility[h,n,p].ub * MHA_Block.attention_weight[h, n, p].lb) - (MHA_Block.compatibility[h,n,p].lb * MHA_Block.attention_weight[h, n, p].ub)) /(MHA_Block.compatibility[h,n,p].ub - MHA_Block.compatibility[h,n,p].lb )
                    MHA_Block.attention_constraints.add(
                        MHA_Block.sct[h, n, p]   == (A *  MHA_Block.compatibility[h,n,p]) + b
                    )

                    
                    # # # Add concave/convex evelope function constraints
                    # # when f(UB) <= 0.5: convex
                    MHA_Block.constr_convex.add( 
                        MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.attention_weight[h, n, p]
                    )
                    MHA_Block.constr_convex.add( 
                        MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.sct[h, n, p] 
                    )
                    # when f(LB) >= 0.5: concave 
                    MHA_Block.constr_concave.add( 
                        MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.sct[h, n, p] 
                    )
                    MHA_Block.constr_concave.add( 
                        MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.attention_weight[h, n, p] 
                    )
                    # otherwise: use concave and convex tie points
                    MHA_Block.constr_concave_tp.add( # when x >= x_cc
                        MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.attention_weight[h, n, p] 
                    )
                    MHA_Block.constr_concave_tp_sct.add( # when x <= x_cc --> cc_sct()
                        MHA_Block.attention_weight_cc[h, n, p] == MHA_Block.tp_cc_sct[h, n, p]
                    ) 
                    MHA_Block.constr_convex_tp_sct.add( # when x >= x_cv --> cv_sct()
                        MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.tp_cv_sct[h, n, p]
                    ) 
                    MHA_Block.constr_convex_tp.add( # when x <= x_cv
                        MHA_Block.attention_weight_cv[h, n, p] == MHA_Block.attention_weight[h, n, p]
                    )
                    
                    ## Add tp_cv_sct constraints
                    #bounds
                    MHA_Block.attention_constraints.add(# att(cv_prime)
                        expr=  MHA_Block.attention_weight_x_cv_prime[h, n, n2] <= 1 
                    )
                    MHA_Block.attention_constraints.add( # att(x_cv)
                        expr=  MHA_Block.attention_weight_x_cv[h, n, n2] <= 1
                    )
                    
                    # tie_point_cv[h, n, p] = max(tie_point_cv_prime, compatibility.lb  
                    BigM_prime =  max( MHA_Block.compatibility[h,n,p_prime].ub for p_prime in time_input)
                    MHA_Block.attention_constraints.add(
                        MHA_Block.tie_point_cv_prime[h, n, p] - MHA_Block.compatibility[h,n,p].lb <= BigM_prime * (1 - MHA_Block.tp_cv[h,n,p])
                    )
                    MHA_Block.attention_constraints.add(
                        MHA_Block.tie_point_cv_prime[h, n, p] - MHA_Block.compatibility[h,n,p].lb >= -BigM_prime * ( MHA_Block.tp_cv[h,n,p])
                    )
                    MHA_Block.attention_constraints.add( # define tie_point_cv
                        MHA_Block.tie_point_cv[h, n, p]  == MHA_Block.tie_point_cv_prime[h, n, p]*(1 - MHA_Block.tp_cv[h,n,p])  + (MHA_Block.compatibility[h,n,p].lb * MHA_Block.tp_cv[h,n,p])
                    )
                    MHA_Block.attention_constraints.add( # softmax(tie_point_cv)
                        MHA_Block.attention_weight_x_cv[h, n, p] == MHA_Block.attention_weight_x_cv_prime[h, n, p]*(1 - MHA_Block.tp_cv[h,n,p])  + MHA_Block.attention_weight[h,n,p].lb * MHA_Block.tp_cv[h,n,p]
                    )
                    
                    MHA_Block.tie_point_cv[h, n, p].ub = math.log(0.5 * MHA_Block.compatibility_exp_sum[h, n].ub)
                    MHA_Block.tie_point_cv[h, n, p].lb =  math.log(1E-9)
                    MHA_Block.tie_point_cv_prime[h, n, p].ub = math.log(0.5 * MHA_Block.compatibility_exp_sum[h, n].ub)
                    MHA_Block.tie_point_cv_prime[h, n, p].lb = math.log(1E-9)
                    
                    # Is x <= x_cv? --> convex zone
                    MHA_Block.attention_constraints.add(
                        expr=  MHA_Block.tie_point_cv[h, n, p] - MHA_Block.compatibility[h,n,p] <= BigM_prime * (1-MHA_Block.t_cv[h,n,p])
                    )
                    MHA_Block.attention_constraints.add(
                        expr=  MHA_Block.tie_point_cv[h, n, p] - MHA_Block.compatibility[h,n,p] >= - BigM_prime * (MHA_Block.t_cv[h,n,p])
                    )
                    # define tie_point_cv_prime[h, n, p]
                    MHA_Block.attention_constraints.add( # 
                        expr=  MHA_Block.tp_cv_mult_1[h, n, p]  == MHA_Block.attention_weight[h,n,p].ub  - MHA_Block.attention_weight_x_cv_prime[h, n, p]
                    )
                    
                    MHA_Block.attention_constraints.add( # 
                        expr=  MHA_Block.tp_cv_mult_2[h, n, p]  == MHA_Block.attention_weight_x_cv_prime[h, n, p] * ( 1 -  MHA_Block.attention_weight_x_cv_prime[h, n, p])
                    )
                    MHA_Block.tp_cv_mult_1[h, n, p].ub = MHA_Block.attention_weight[h,n,p].ub  - MHA_Block.attention_weight_x_cv_prime[h, n, p].lb
                    MHA_Block.tp_cv_mult_1[h, n, p].lb = MHA_Block.attention_weight[h,n,p].ub  - MHA_Block.attention_weight_x_cv_prime[h, n, p].ub
                    
                    MHA_Block.tp_cv_mult_2[h, n, p].ub = 1
                    MHA_Block.tp_cv_mult_2[h, n, p].lb = 0
                    
                    
                    MHA_Block.attention_constraints.add( 
                        expr=  (MHA_Block.compatibility[h,n,p].ub - MHA_Block.tie_point_cv_prime[h, n, p]) * MHA_Block.tp_cv_mult_2[h, n, p]  == MHA_Block.tp_cv_mult_1[h, n, p]
                    )
                    # define tie point cv  secant
                    MHA_Block.constr_convex_tp_sct.add( 
                        expr=  MHA_Block.tp_cv_sct[h, n, p] - MHA_Block.attention_weight[h,n,p].ub == 
                                                            + (MHA_Block.tp_cv_sct_mult_1_2[h, n, p] 
                                                               * (MHA_Block.compatibility[h,n,p]
                                                                - MHA_Block.compatibility[h,n,p].ub))
                    )
                    MHA_Block.constr_convex_tp_sct.add( 
                        expr=  MHA_Block.tp_cv_sct_mult_1_2[h, n, p] * MHA_Block.tp_cv_sct_mult_2[h, n, p] == MHA_Block.tp_cv_sct_mult_1[h, n, p] 
                    )
                    MHA_Block.constr_convex_tp_sct.add( 
                        expr=  MHA_Block.tp_cv_sct_mult_1[h, n, p] == MHA_Block.attention_weight[h,n,p].ub -  MHA_Block.attention_weight_x_cv[h, n, p]
                    )
                    
                    MHA_Block.constr_convex_tp_sct.add( 
                        expr=  MHA_Block.tp_cv_sct_mult_2[h, n, p] == MHA_Block.compatibility[h,n,p].ub - MHA_Block.tie_point_cv[h, n, p]
                    )
                    MHA_Block.tp_cv_sct_mult_1[h, n, p].ub = MHA_Block.attention_weight[h,n,p].ub -  MHA_Block.attention_weight_x_cv[h, n, p].lb
                    MHA_Block.tp_cv_sct_mult_1[h, n, p].lb = MHA_Block.attention_weight[h,n,p].ub -  MHA_Block.attention_weight_x_cv[h, n, p].ub
                    
                    MHA_Block.tp_cv_sct_mult_2[h, n, p].ub = MHA_Block.compatibility[h,n,p].ub - MHA_Block.tie_point_cv[h, n, p].lb
                    MHA_Block.tp_cv_sct_mult_2[h, n, p].lb = MHA_Block.compatibility[h,n,p].ub - MHA_Block.tie_point_cv[h, n, p].ub
                    
                    MHA_Block.tp_cv_sct_mult_1_2[h, n, p].ub = max( MHA_Block.tp_cv_sct_mult_1[h, n, p].ub/ (MHA_Block.tp_cv_sct_mult_2[h, n, p].lb), MHA_Block.tp_cv_sct_mult_1[h, n, p].lb/ (MHA_Block.tp_cv_sct_mult_2[h, n, p].ub))
                    MHA_Block.tp_cv_sct_mult_1_2[h, n, p].lb = min( MHA_Block.tp_cv_sct_mult_1[h, n, p].ub/ (MHA_Block.tp_cv_sct_mult_2[h, n, p].lb), MHA_Block.tp_cv_sct_mult_1[h, n, p].lb/ (MHA_Block.tp_cv_sct_mult_2[h, n, p].ub))
                    
                    ## Add tp_cc_sct constraints
                    #bounds
                    MHA_Block.attention_constraints.add(# att(cc_prime)
                        expr=  MHA_Block.attention_weight_x_cc_prime[h, n, n2] <= 1 
                    )
                    MHA_Block.attention_constraints.add( # att(x_cc)
                        expr=  MHA_Block.attention_weight_x_cc[h, n, n2] <= 1
                    )
                    
                    # tie_point_cc[h, n, p] = min(tie_point_cc_prime, compatibility.ub  )
                    MHA_Block.attention_constraints.add(
                        MHA_Block.tie_point_cc_prime[h, n, p] - MHA_Block.compatibility[h,n,p].ub <= BigM_prime * (1 - MHA_Block.tp_cc[h,n,p])
                    )
                    MHA_Block.attention_constraints.add(
                        MHA_Block.tie_point_cc_prime[h, n, p] - MHA_Block.compatibility[h,n,p].ub >= -BigM_prime * ( MHA_Block.tp_cc[h,n,p])
                    )
                    MHA_Block.attention_constraints.add( # define tie_point_cc
                        MHA_Block.tie_point_cc[h, n, p]  == MHA_Block.tie_point_cc_prime[h, n, p]*(MHA_Block.tp_cc[h,n,p])  + (MHA_Block.compatibility[h,n,p].ub * (1 - MHA_Block.tp_cc[h,n,p]))
                    )
                    MHA_Block.attention_constraints.add( # softmax(tie_point_cc)
                        MHA_Block.attention_weight_x_cc[h, n, p] == MHA_Block.attention_weight_x_cc_prime[h, n, p]*(MHA_Block.tp_cc[h,n,p])  + (MHA_Block.attention_weight[h,n,p].ub * (1 - MHA_Block.tp_cc[h,n,p]))
                    )
                    
                    MHA_Block.tie_point_cc[h, n, p].ub = math.log( MHA_Block.compatibility_exp_sum[h, n].ub)
                    MHA_Block.tie_point_cc_prime[h, n, p].ub = math.log( MHA_Block.compatibility_exp_sum[h, n].ub)
                    val = math.log(0.5 * MHA_Block.compatibility_exp_sum[h, n].lb) if  MHA_Block.compatibility_exp_sum[h, n].lb > 0 else math.log(1E-9)
                    MHA_Block.tie_point_cc[h, n, p].lb = val
                    MHA_Block.tie_point_cc_prime[h, n, p].lb = val
                        
                    # Is x <= x_cc? --> convex zone
                    MHA_Block.attention_constraints.add(
                        expr=  MHA_Block.compatibility[h,n,p] - MHA_Block.tie_point_cc[h, n, p] <= BigM_prime * (1 - MHA_Block.t_cc[h,n,p])
                    )
                    MHA_Block.attention_constraints.add(
                        expr=  MHA_Block.compatibility[h,n,p] - MHA_Block.tie_point_cc[h, n, p]>= - BigM_prime * (MHA_Block.t_cc[h,n,p])
                    )
                    # define tie_point_cc_prime[h, n, p]
                    MHA_Block.attention_constraints.add( # 
                        expr=  MHA_Block.tp_cc_mult_1[h, n, p]  == MHA_Block.attention_weight_x_cc_prime[h, n, p] - MHA_Block.attention_weight[h,n,p].lb
                    )
                    
                    MHA_Block.attention_constraints.add( # 
                        expr=  MHA_Block.tp_cc_mult_2[h, n, p]  == MHA_Block.attention_weight_x_cc_prime[h, n, p] * ( 1 -  MHA_Block.attention_weight_x_cc_prime[h, n, p])
                    )
                    MHA_Block.tp_cc_mult_1[h, n, p].ub = MHA_Block.attention_weight_x_cc_prime[h, n, p].ub - MHA_Block.attention_weight[h,n,p].lb
                    MHA_Block.tp_cc_mult_1[h, n, p].lb = MHA_Block.attention_weight_x_cc_prime[h, n, p].lb - MHA_Block.attention_weight[h,n,p].lb
                    
                    MHA_Block.tp_cc_mult_2[h, n, p].ub = 1
                    MHA_Block.tp_cc_mult_2[h, n, p].lb = 0
                    
                    MHA_Block.attention_constraints.add( 
                        expr=  (MHA_Block.tie_point_cc_prime[h, n, p] - MHA_Block.compatibility[h,n,p].lb ) * MHA_Block.tp_cc_mult_2[h, n, p]  == MHA_Block.tp_cc_mult_1[h, n, p]
                    )
                    # define tie point cc  secant
                    MHA_Block.constr_concave_tp_sct.add( 
                        expr=  MHA_Block.tp_cc_sct[h, n, p] - MHA_Block.attention_weight[h,n,p].lb == 
                                                            + (MHA_Block.tp_cc_sct_mult_1_2[h, n, p] 
                                                               * (MHA_Block.compatibility[h,n,p]
                                                                - MHA_Block.compatibility[h,n,p].lb))
                    )
                    MHA_Block.constr_concave_tp_sct.add( 
                        expr=  MHA_Block.tp_cc_sct_mult_1_2[h, n, p] * MHA_Block.tp_cc_sct_mult_2[h, n, p] == MHA_Block.tp_cc_sct_mult_1[h, n, p] 
                    )
                    MHA_Block.constr_concave_tp_sct.add( 
                        expr=  MHA_Block.tp_cc_sct_mult_1[h, n, p] == MHA_Block.attention_weight[h,n,p].lb -  MHA_Block.attention_weight_x_cc[h, n, p]
                    )
                    MHA_Block.constr_concave_tp_sct.add( 
                        expr=  MHA_Block.tp_cc_sct_mult_2[h, n, p] == MHA_Block.compatibility[h,n,p].lb - MHA_Block.tie_point_cc[h, n, p]
                    )
                    
                    MHA_Block.tp_cc_sct_mult_1[h, n, p].ub = MHA_Block.attention_weight[h,n,p].lb - MHA_Block.attention_weight_x_cc[h, n, p].lb
                    MHA_Block.tp_cc_sct_mult_1[h, n, p].lb = MHA_Block.attention_weight[h,n,p].lb - MHA_Block.attention_weight_x_cc[h, n, p].ub
                    
                    MHA_Block.tp_cc_sct_mult_2[h, n, p].ub = MHA_Block.compatibility[h,n,p].lb - MHA_Block.tie_point_cc[h, n, p].lb
                    MHA_Block.tp_cc_sct_mult_2[h, n, p].lb = MHA_Block.compatibility[h,n,p].lb - MHA_Block.tie_point_cc[h, n, p].ub
                    
                    MHA_Block.tp_cc_sct_mult_1_2[h, n, p].ub = max( MHA_Block.tp_cc_sct_mult_1[h, n, p].ub/ (MHA_Block.tp_cc_sct_mult_2[h, n, p].lb), MHA_Block.tp_cc_sct_mult_1[h, n, p].lb/ (MHA_Block.tp_cc_sct_mult_2[h, n, p].ub))
                    MHA_Block.tp_cc_sct_mult_1_2[h, n, p].lb = min( MHA_Block.tp_cc_sct_mult_1[h, n, p].ub/ (MHA_Block.tp_cc_sct_mult_2[h, n, p].lb), MHA_Block.tp_cc_sct_mult_1[h, n, p].lb/ (MHA_Block.tp_cc_sct_mult_2[h, n, p].ub))
                    
   
        # multihead attention output constraint
        for n in time_input:
            for d in model.model_dims:
                out_bound_1 = sum( sum( max(MHA_Block.attention_score[h, n, k].ub * MHA_Block.W_v[d, h, k], MHA_Block.attention_score[h, n, k].lb * MHA_Block.W_o[d, h, k])  for k in MHA_Block.k_dims) for h in MHA_Block.heads) 
                out_bound_2 = sum( sum( min(MHA_Block.attention_score[h, n, k].ub * MHA_Block.W_v[d, h, k], MHA_Block.attention_score[h, n, k].lb * MHA_Block.W_o[d, h, k])  for k in MHA_Block.k_dims) for h in MHA_Block.heads) 
                    
                if b_o:
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

                    if out_bound_1 < out_bound_2: 
                        attention_output[n, d].ub = out_bound_2 + MHA_Block.b_o[d]
                        attention_output[n, d].lb = out_bound_1 + MHA_Block.b_o[d]
                    else:
                        attention_output[n, d].ub = out_bound_1 + MHA_Block.b_o[d]
                        attention_output[n, d].lb = out_bound_2 + MHA_Block.b_o[d]
                    
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
                    
                    if out_bound_1 < out_bound_2: 
                        attention_output[n, d].ub = out_bound_2 
                        attention_output[n, d].lb = out_bound_1 
                    else:
                        attention_output[n, d].ub = out_bound_1 
                        attention_output[n, d].lb = out_bound_2 
                    
        #activate softmax envelope constraints              
        MHA_Block.activate_constraints = pyo.BuildAction(rule=activate_envelope_att)            
                
    # def add_attention_approx(self, model,, input_var_name, W_q, W_k, W_v, W_o, b_q = None, b_k = None, b_v = None, b_o = None):
    #     """
    #     Multihead attention between each element of embedded sequence
        
    #     Exp function created using power series approximation (11 elements of power series). 
    #     This formulation avoids the pyomo solving error when calculating pyo.exp(pyo.Var())
    #     """
    #     if not hasattr(self.Transformer_Block, "attention_constraints"):
    #         MHA_Block.attention_constraints = pyo.ConstraintList()
            
    #     input_var = getattr(self.Transformer_Block, input_var_name)

    #     # define sets, vars
    #     MHA_Block.heads = pyo.RangeSet(1, self.d_H)
    #     MHA_Block.k_dims = pyo.RangeSet(1, self.d_k)

    #     W_q_dict = {
    #         (D, H, K): W_q[d][h][k]
    #         for d,D in enumerate(model.model_dims)
    #         for h,H in enumerate(MHA_Block.heads)
    #         for k,K in enumerate(MHA_Block.k_dims)
    #     }
    #     W_k_dict = {
    #         (D, H, K): W_k[d][h][k]
    #         for d,D in enumerate(model.model_dims)
    #         for h,H in enumerate(MHA_Block.heads)
    #         for k,K in enumerate(MHA_Block.k_dims)
    #     }
    #     W_v_dict = {
    #         (D, H, K): W_v[d][h][k]
    #         for d,D in enumerate(model.model_dims)
    #         for h,H in enumerate(MHA_Block.heads)
    #         for k,K in enumerate(MHA_Block.k_dims)
    #     }
    #     W_o_dict = {
    #         (D, H, K): W_o[h][k][d]
    #         for d,D in enumerate(model.model_dims)
    #         for h,H in enumerate(MHA_Block.heads)
    #         for k,K in enumerate(MHA_Block.k_dims)
    #     }
 
    #     MHA_Block.W_q = pyo.Param(model.model_dims, MHA_Block.heads, MHA_Block.k_dims, initialize=W_q_dict, mutable=False)
    #     MHA_Block.W_k = pyo.Param(model.model_dims, MHA_Block.heads, MHA_Block.k_dims, initialize=W_k_dict, mutable=False)
    #     MHA_Block.W_v = pyo.Param(model.model_dims, MHA_Block.heads, MHA_Block.k_dims, initialize=W_v_dict, mutable=False)
    #     MHA_Block.W_o = pyo.Param(model.model_dims,MHA_Block.heads, MHA_Block.k_dims, initialize=W_o_dict, mutable=False)
       
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
    #         b_o_dict = {(d): val for d, val in zip(model.model_dims, b_o) }
    #         MHA_Block.b_o = pyo.Param(model.model_dims, initialize=b_o_dict, mutable=False)
            

    #     MHA_Block.Q = pyo.Var(MHA_Block.heads, time_input, MHA_Block.k_dims, within=pyo.Reals) 
    
    #     MHA_Block.K = pyo.Var(MHA_Block.heads, time_input, MHA_Block.k_dims, within=pyo.Reals)
        
    #     MHA_Block.V = pyo.Var(MHA_Block.heads, time_input, MHA_Block.k_dims, within=pyo.Reals) 
        
        
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
    #         time_input, model.model_dims, within=pyo.Reals
    #     )  # concat heads and linear transform

    #     for h in MHA_Block.heads:
    #         for n in time_input:
    #                 for k in MHA_Block.k_dims:
                        
    #                      # constraints for Query
    #                     if b_q:
    #                         MHA_Block.attention_constraints.add(
    #                         expr=MHA_Block.Q[h, n, k]
    #                         == sum(input_var[n,d] * MHA_Block.W_q[d, h, k] for d in model.model_dims) + MHA_Block.b_q[h,k] 
    #                         )  
    #                         #Add bounds
    #                         q_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model.model_dims) + MHA_Block.b_q[h,k]
    #                         q_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model.model_dims) + MHA_Block.b_q[h,k]
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
    #                             == sum(input_var[n, d] * MHA_Block.W_q[d, h, k] for d in model.model_dims)
    #                         )
    #                         #Add bounds
    #                         q_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model.model_dims)
    #                         q_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_q[d, h, k], input_var[n,d].lb * MHA_Block.W_q[d, h, k])  for d in model.model_dims)
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
    #                         == sum(input_var[n, d] * MHA_Block.W_k[d, h, k] for d in model.model_dims) + MHA_Block.b_k[h,k]
    #                         )  
    #                         #Add bounds
    #                         k_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in model.model_dims) + MHA_Block.b_k[h,k]
    #                         k_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in model.model_dims) + MHA_Block.b_k[h,k]
    #                         if k_bound_1 < k_bound_2: 
    #                             MHA_Block.K[h, n, k].ub = k_bound_2
    #                             MHA_Block.K[h, n, k].lb = k_bound_1
    #                         else:
    #                             MHA_Block.K[h, n, k].ub = k_bound_1
    #                             MHA_Block.K[h, n, k].lb = k_bound_2
                            
    #                     else: 
    #                         MHA_Block.attention_constraints.add(
    #                             expr=MHA_Block.K[h, n, k]
    #                             == sum(input_var[n, d] * MHA_Block.W_k[d, h, k] for d in model.model_dims)
    #                         )
    #                         #Add bounds
    #                         k_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in model.model_dims) 
    #                         k_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_k[d, h, k], input_var[n,d].lb * MHA_Block.W_k[d, h, k])  for d in model.model_dims) 
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
    #                         == sum(input_var[n, d] * MHA_Block.W_v[d, h, k] for d in model.model_dims) + MHA_Block.b_v[h,k]
    #                         )  
    #                         #Add bounds
                            
    #                         v_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in model.model_dims) + MHA_Block.b_v[h,k]
    #                         v_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in model.model_dims) + MHA_Block.b_v[h,k]
    #                         if v_bound_1 < v_bound_2: 
    #                             MHA_Block.V[h, n, k].ub = v_bound_2
    #                             MHA_Block.V[h, n, k].lb = v_bound_1
    #                         else:
    #                             MHA_Block.V[h, n, k].ub = v_bound_1
    #                             MHA_Block.V[h, n, k].lb = v_bound_2
                            
    #                     else: 
    #                         MHA_Block.attention_constraints.add(
    #                             expr=MHA_Block.V[h, n, k]
    #                             == sum(input_var[n, d] * MHA_Block.W_v[d, h, k] for d in model.model_dims) 
    #                         )
    #                         #Add bounds     
    #                         v_bound_1 = sum( max(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in model.model_dims)
    #                         v_bound_2 = sum( min(input_var[n,d].ub * MHA_Block.W_v[d, h, k], input_var[n,d].lb * MHA_Block.W_v[d, h, k])  for d in model.model_dims)
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
    #         for d in model.model_dims:
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
                
                

    def add_residual_connection(self, model, input_1_name, input_2_name, output_var_name):
        input_1 = getattr(self.Transformer_Block, input_1_name)
        input_2 = getattr(self.Transformer_Block, input_2_name)
        if input_1.is_indexed():
            set_var = input_1.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( model, set) )
            
            time_input = indices[0]
        else:
            ValueError('Input value must be indexed')
            
        # create constraint list
        if not hasattr(self.Transformer_Block, "residual_constraints"):
            self.Transformer_Block.residual_constraints = pyo.ConstraintList()
        
        # add new variable
        if not hasattr(self.Transformer_Block, output_var_name):
            setattr(self.Transformer_Block, output_var_name, pyo.Var(time_input, model.model_dims, within=pyo.Reals))
            residual_var = getattr(self.Transformer_Block, output_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')
        
        
        for n in time_input:
            for d in model.model_dims:
                self.Transformer_Block.residual_constraints.add(expr= residual_var[n,d] == input_1[n,d] + input_2[n,d])
                try:
                    residual_var[n,d].ub == input_1[n,d].ub + input_2[n,d].ub
                    residual_var[n,d].lb == input_1[n,d].lb + input_2[n,d].lb
                except:
                    continue
                
    # def add_FFN_2D(self, modelself.Transformer_Block, input_var_name, output_var_name, input_shape, model_parameters):
    #     input_var = getattr(self.Transformer_Block, input_var_name)

    #     # add new variable
    #     if not hasattr(self.Transformer_Block, output_var_name + "_NN_Block"):
    #         NN_name = output_var_name + "_NN_Block"
    #         setattr(self.Transformer_Block, NN_name, OmltBlock())
    #         NN_block = getattr(self.Transformer_Block, NN_name)
            
    #         setattr(self.Transformer_Block, output_var_name, pyo.Var(input_var.index_set(), within=pyo.Reals))
    #         output_var = getattr(self.Transformer_Block, output_var_name)
            
    #         setattr(self.Transformer_Block, output_var_name+"_constraints", pyo.ConstraintList())
    #         ffn_constraints = getattr(self.Transformer_Block, output_var_name+"_constraints")
    #     else:
    #         raise ValueError('Attempting to overwrite variable')
        
    #     ###### GET BOUNDS
    #     input_bounds={0: (-4,4), 1: (-4,4), 2: (-4,4), 3:(-4,4), 4:(-4,4), 5: (-4,4), 6: (-4,4), 7: (-4,4), 8: (-4,4), 9: (-4,4)} ### fix input bounds
    #     net_relu = OMLT_helper.weights_to_NetDef(output_var_name, input_shape, model_parameters, input_bounds)
    #     NN_block.build_formulation(ReluBigMFormulation(net_relu))
        
    #     # Set input constraints
    #     input_indices_len, input_indices_attr = self.get_indices(self.Transformer_Block, input_var)
    #     if input_indices_len == 1:
    #         for i, index in  enumerate(input_indices_attr[0]):
    #             ffn_constraints.add(expr= input_var[index] == NN_block.inputs[i])
    #     elif input_indices_len == 2:
    #         for i, i_index in  enumerate(input_indices_attr[0]):
    #             for j, j_index in  enumerate(input_indices_attr[1]):
    #                 ffn_constraints.add(expr= input_var[i_index, j_index] == NN_block.inputs[j])
                    
                    
    #     # Set output constraints
    #     output_indices_len, output_indices_attr = self.get_indices(self.Transformer_Block, output_var)
    #     if output_indices_len == 1:
    #         for i, index in  enumerate(output_indices_attr[0]):
    #             ffn_constraints.add(expr= output_var[index] == NN_block.outputs[i])
    #     elif output_indices_len == 2:
    #         for i, i_index in  enumerate(output_indices_attr[0]):
    #             for j, j_index in  enumerate(output_indices_attr[1]):
    #                 ffn_constraints.add(expr= output_var[i_index, j_index] == NN_block.outputs[j])
            
    def get_fnn(self, model, input_var_name, output_var_name, nn_name, input_shape, model_parameters):
        input_var = getattr(self.Transformer_Block, input_var_name)
        
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( model, set) )
            
            time_input = indices[0]
        else:
            ValueError('Input value must be indexed')
        
        # add new variable
        if not hasattr(self.Transformer_Block, output_var_name):
            setattr(self.Transformer_Block, output_var_name, pyo.Var(input_var.index_set(), within=pyo.Reals))
            output_var = getattr(self.Transformer_Block, output_var_name)

            #set bounds
            for i in input_var.index_set():
                if input_var[i].lb:
                    output_var[i].lb = input_var[i].lb
                if input_var[i].ub:
                    output_var[i].ub = input_var[i].ub
                
            
            setattr(self.Transformer_Block, output_var_name+"_constraints", pyo.ConstraintList())
            ffn_constraints = getattr(self.Transformer_Block, output_var_name+"_constraints")
        else:
            raise ValueError('Attempting to overwrite variable')
        
        nn= GUROBI_ML_helper.weights_to_NetDef(output_var_name, nn_name, input_shape, model_parameters)
       
        return nn, input_var, output_var
            
        
    def get_indices(self, model, input_var):
        # Get indices of var
        indices = str(input_var.index_set()).split('*')
        indices_len = len(indices)
        indices_attr = []
        for i in indices:
            try: 
                indices_attr += [getattr(self.Transformer_Block, i)]
            except:
                raise ValueError('Input variable not indexed by a pyomo Set')
        
        return indices_len, indices_attr
        
    def add_avg_pool(self, model, input_var_name, output_var_name):
        input_var = getattr(self.Transformer_Block, input_var_name)
        
        if input_var.is_indexed():
            set_var = input_var.index_set()
            indices = []
            for set in str(set_var).split("*"):
                indices.append( getattr( model, set) )
            
            time_input = indices[0]
        else:
            ValueError('Input value must be indexed')

        
        # add new variable
        if not hasattr(self.Transformer_Block, output_var_name):
            setattr(self.Transformer_Block, "avg_pool_constr_"+output_var_name, pyo.ConstraintList())
            constraints = getattr(self.Transformer_Block, "avg_pool_constr_"+output_var_name) 
            
            setattr(self.Transformer_Block, output_var_name, pyo.Var(model.model_dims, within=pyo.Reals))
            output_var = getattr(self.Transformer_Block, output_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')


        for d in model.model_dims: 
            constraints.add(expr= output_var[d] * self.N == sum(input_var[t,d] for t in time_input))
            
            try:
                output_var[d].ub  == sum(input_var[t,d].ub for t in time_input) / self.N
                output_var[d].lb  == sum(input_var[t,d].lb for t in time_input) / self.N
            except:
                continue
            
            
            
            
    #def add_output_constraints(self, model,, input_var):
        # if not hasattr(self.Transformer_Block, "output_constraints"):
        #     self.Transformer_Block.output_constraints = pyo.ConstraintList()

        # # predict x, u
        # output = np.ones((len(self.Transformer_Block.time), self.d_model))
        # dict_output = {(t, str(d)): output[i, d] for i, t in enumerate(self.Transformer_Block.time) for d in range(len(model.model_dims))}
        # print(dict_output)
        # self.Transformer_Block.transformer_output = pyo.Param(self.Transformer_Block.time, model.model_dims, initialize=dict_output)
        
        # for t in self.Transformer_Block.time:
        #     if t > 0.9:
        #         # add constraints for next value
        #         for d in model.model_dims:
        #             self.Transformer_Block.output_constraints.add(expr=input_var[t,d] == self.Transformer_Block.transformer_output[t, d])
        #             self.Transformer_Block.output_constraints.add(expr=input_var[t,d] == self.Transformer_Block.transformer_output[t, d])






