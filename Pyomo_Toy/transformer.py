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
import OMLT_helper 
import GUROBI_ML_helper

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

# def activate_envelope_att(model):
#         model.constr_convex.deactivate()
#         model.constr_concave.deactivate() 
#         model.constr_convex_tp.deactivate()
#         model.constr_convex_tp_sct.deactivate()
#         model.constr_concave_tp.deactivate()
#         model.constr_concave_tp_sct.deactivate()

        # if model.s_cv == 0: # --> convex region onlyt
        #     model.constr_convex.activate()
        # elif model.s_cc == 0: # --> concave region only
        #     model.constr_concave.activate() 
        # else: # in both regions
        #     if model.t_cv == 0: # --> if x <= x_cv_tiepoint -->convex region
        #         model.constr_convex_tp.activate()
        #     else: # -->concave region
        #         model.constr_convex_tp_sct.activate()
                
        #     if model.t_cc == 0: # --> if x >= x_cc_tiepoint -->concave region
        #         model.constr_concave_tp.activate()
        #     else:# --> convex region
        #         model.constr_concave_tp_sct.activate()

class Transformer:
    def __init__(self, M, config_file):
        
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
        
        # initialise set of model dims
        if self.d_model > 1:
            str_array = ["{}".format(x) for x in range(0, self.d_model)]
            M.model_dims = pyo.Set(initialize=str_array)
        else:
            M.model_dims = pyo.Set(initialize=[str(0)])

    def embed_input(self, M, input_var_name, embed_var_name, set_input_var_name, W_emb=None):
        """
        Embed the feature dimensions of input
        """
        if not hasattr(M, "embed_constraints"):
            M.embed_constraints = pyo.ConstraintList()
            
        input_var = getattr(M, input_var_name)
        set_var = getattr(M, set_input_var_name)
        
        # define embedding var
        if not hasattr(M, embed_var_name):
            init_array = 0.5 * np.ones((self.N, self.d_model)) #randomly initialize embed array to create pyo.Var
            dict_embed = {}
            for t in range(len(M.time_input)):
                for s in range(len(set_var)):
                    dict_embed[(M.time_input.at(t+1), set_var.at(s+1))] = init_array[t,s]
                
            setattr(M, embed_var_name, pyo.Var(M.time_input, M.model_dims, initialize=dict_embed))

            embed_var = getattr(M, embed_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')
  
        
        #if model dims = number of variables in input var
        if W_emb is None:
            for s in set_var:
                for t in M.time_input:
                    M.embed_constraints.add(embed_var[t, s] == input_var[t,s])
                    if isinstance(input_var, pyo.Var):
                        if input_var[t,s].ub:
                            embed_var[t, s].ub = input_var[t,s].ub
                        if input_var[t,s].lb:
                            embed_var[t, s].lb = input_var[t,s].lb
                    elif isinstance(input_var, pyo.Param):
                        embed_var[t, s].ub = input_var[t,s]
                        embed_var[t, s].lb = input_var[t,s]
                    
        else: # create embedded var
            W_emb_dict = {
                (set_var.at(s+1),M.model_dims.at(d+1)): W_emb[s][d]
                for s in range(len(set_var))
                for d in range(len(M.model_dims))
            }
            M.W_emb = pyo.Param(set_var, M.model_dims, initialize=W_emb_dict)
            
            for d in M.model_dims:
                for t in M.time_input:
                    M.embed_constraints.add(embed_var[t, d] 
                                            == sum(input_var[t,s] * M.W_emb[s,d] for s in set_var)
                                            )
                    if isinstance(input_var, pyo.Var):
                        try:
                            embed_var[t, d].ub = sum(input_var[t,s].ub * M.W_emb[s,d] for s in set_var)
                            embed_var[t, d].lb = sum(input_var[t,s].lb * M.W_emb[s,d] for s in set_var)
                        except:
                            continue
                    elif isinstance(input_var, pyo.Param):
                        embed_var[t, d].ub = sum(input_var[t,s] * M.W_emb[s,d] for s in set_var)
                        embed_var[t, d].lb = sum(input_var[t,s] * M.W_emb[s,d] for s in set_var)

    def add_layer_norm(self, M, input_var_name, layer_norm_var_name, gamma= None, beta = None, std=None):  # non-linear
        """
        Normalization over the sequennce of input
        """
        if not hasattr(M, "layer_norm_constraints"):
            M.layer_norm_constraints = pyo.ConstraintList()
        
        input_var = getattr(M, input_var_name)
        
        # Initialize variables
        if not hasattr(M, layer_norm_var_name):
            # define layer norm output var
            setattr(M, layer_norm_var_name, pyo.Var(M.time_input, M.model_dims, within=pyo.Reals))
            layer_norm_var = getattr(M, layer_norm_var_name)
            
            # define calculation variables
            sum_name = 'sum_'+ layer_norm_var_name
            setattr(M, sum_name, pyo.Var(M.time_input, within=pyo.Reals))
            sum_t = getattr(M, sum_name)
            
            variance_name = 'variance_'+ layer_norm_var_name
            setattr(M, variance_name, pyo.Var(M.time_input, within=pyo.Reals))
            variance = getattr(M, variance_name)
            
            div_name = 'div_'+ layer_norm_var_name
            setattr(M, div_name, pyo.Var(M.time_input, M.model_dims, within=pyo.Reals))
            div = getattr(M, div_name)
            
            denominator_name = 'denominator_'+ layer_norm_var_name
            setattr(M, denominator_name, pyo.Var(M.time_input, within=pyo.Reals))
            denominator = getattr(M, denominator_name)
            
            denominator_abs_name = 'denominator_abs_'+ layer_norm_var_name
            setattr(M, denominator_abs_name, pyo.Var(M.time_input, within=pyo.NonNegativeReals, bounds=(0,None)))
            denominator_abs = getattr(M, denominator_abs_name)
            
            numerator_name = 'numerator_'+ layer_norm_var_name
            setattr(M, numerator_name, pyo.Var(M.time_input, M.model_dims, within=pyo.Reals))
            numerator = getattr(M, numerator_name)

            numerator_scaled_name = 'numerator_scaled_'+ layer_norm_var_name
            setattr(M, numerator_scaled_name, pyo.Var(M.time_input, M.model_dims, within=pyo.Reals))
            numerator_scaled = getattr(M, numerator_scaled_name)
            
            numerator_squared_name = 'numerator_squared_'+ layer_norm_var_name
            setattr(M, numerator_squared_name, pyo.Var(M.time_input, M.model_dims, within=pyo.Reals, bounds=(0,None)))
            numerator_squared = getattr(M, numerator_squared_name)
              
            numerator_squared_sum_name = 'numerator_squared_sum_'+ layer_norm_var_name
            setattr(M, numerator_squared_sum_name, pyo.Var(M.time_input, within=pyo.Reals, bounds=(0,None)))
            numerator_squared_sum = getattr(M, numerator_squared_sum_name)
            
        else:
            raise ValueError('Attempting to overwrite variable')

        # Add constraints for layer norm
        if self.d_model == 1:
            return
            
        for t in M.time_input: 
            M.layer_norm_constraints.add(expr= sum_t[t] == sum(input_var[t, d] for d in M.model_dims) )
            
            # Constraints for each element in sequence
            for d in M.model_dims:  
                M.layer_norm_constraints.add(expr= numerator[t,d] == input_var[t, d] - ((1/ self.d_model) *sum_t[t]))
                M.layer_norm_constraints.add(expr= numerator_squared[t,d] == numerator[t,d]**2)
                
                M.layer_norm_constraints.add(expr= numerator_squared_sum[t] == sum(numerator_squared[t,d_prime] for d_prime in M.model_dims))
                M.layer_norm_constraints.add(expr= variance[t] * self.d_model == numerator_squared_sum[t])
                
                #M.layer_norm_constraints.add(expr= denominator[t] **2 == variance[t] )     ##IF SCIP SOLVER
                ## FOR SCIP or GUROBI: determine abs(denominator)
                M.layer_norm_constraints.add(expr= denominator[t] <= denominator_abs[t]) 
                M.layer_norm_constraints.add(expr= denominator[t]*denominator[t] == denominator_abs[t] * denominator_abs[t]) 
                
                M.layer_norm_constraints.add(expr= variance[t] == denominator[t] * denominator_abs[t]) 
                if std:
                    denominator[t].ub = std
                    denominator[t].lb = -std
                    
                
                M.layer_norm_constraints.add(expr= div[t,d] * denominator[t] == numerator[t,d] )
                div[t,d].ub = 4
                div[t,d].lb = -4
                
                if gamma and beta:
                    M.layer_norm_constraints.add(expr= numerator_scaled[t,d] == getattr(M, gamma)[d] * div[t,d])
                    M.layer_norm_constraints.add(expr=layer_norm_var[t, d] == numerator_scaled[t,d] + getattr(M, beta)[d])
                    layer_norm_var[t, d].ub = getattr(M, beta)[d] + 4*getattr(M, gamma)[d]
                    layer_norm_var[t, d].lb = getattr(M, beta)[d] - 4*getattr(M, gamma)[d]
                else: 
                    M.layer_norm_constraints.add(expr= numerator_scaled[t,d] == div[t,d])
                    M.layer_norm_constraints.add(expr=layer_norm_var[t, d] == numerator_scaled[t,d])
                    layer_norm_var[t, d].ub = 4
                    layer_norm_var[t, d].lb = -4
                    
                #Add bounds
                if input_var[t, d].ub and input_var[t, d].lb:
                    mean_u = (sum(input_var[t, d_prime].ub for d_prime in M.model_dims)/ self.d_model )
                    mean_l = (sum(input_var[t, d_prime].lb for d_prime in M.model_dims)/ self.d_model )
                    numerator[t,d].ub = input_var[t, d].ub - mean_l
                    numerator[t,d].lb = input_var[t, d].lb - mean_u
                    numerator_squared[t,d].ub = max(numerator[t,d].ub**2, numerator[t,d].lb**2) 
                    
                    if not std :
                        denominator[t].ub = abs( max(input_var[t,:].ub) - min(input_var[t,:].lb))
                        denominator[t].lb = - denominator[t].ub 
                numerator_squared[t,d].lb = 0
            if input_var[t, d].ub and input_var[t, d].lb:
                numerator_squared_sum[t].ub = 2 * (sum( (numerator_squared[t,d_prime].ub)**2  for d_prime in M.model_dims)**0.5) 
            numerator_squared_sum[t].lb = 0
            
    
    def add_attention(self, M, input_var_name, W_q, W_k, W_v, W_o, b_q = None, b_k = None, b_v = None, b_o = None):
        """
        Multihead attention between each element of embedded sequence
        
        Uses the pyo.exp() function to calculate softmax. 
        This is compatible with gurobi which allows for the outer approximation of the function to be calculated
        """
        if not hasattr(M, "attention_constraints"):
            M.attention_constraints = pyo.ConstraintList()
            M.constr_convex = pyo.ConstraintList()
            M.constr_concave = pyo.ConstraintList()
            M.constr_convex_tp = pyo.ConstraintList()
            M.constr_convex_tp_sct = pyo.ConstraintList()
            M.constr_concave_tp = pyo.ConstraintList()
            M.constr_concave_tp_sct = pyo.ConstraintList()
            
        input_var = getattr(M, input_var_name)

        # define sets, vars
        M.heads = pyo.RangeSet(1, self.d_H)
        M.k_dims = pyo.RangeSet(1, self.d_k)

        W_q_dict = {
            (D, H, K): W_q[d][h][k]
            for d,D in enumerate(M.model_dims)
            for h,H in enumerate(M.heads)
            for k,K in enumerate(M.k_dims)
        }
        W_k_dict = {
            (D, H, K): W_k[d][h][k]
            for d,D in enumerate(M.model_dims)
            for h,H in enumerate(M.heads)
            for k,K in enumerate(M.k_dims)
        }
        W_v_dict = {
            (D, H, K): W_v[d][h][k]
            for d,D in enumerate(M.model_dims)
            for h,H in enumerate(M.heads)
            for k,K in enumerate(M.k_dims)
        }
        W_o_dict = {
            (D, H, K): W_o[h][k][d]
            for d,D in enumerate(M.model_dims)
            for h,H in enumerate(M.heads)
            for k,K in enumerate(M.k_dims)
        }
 
        M.W_q = pyo.Param(M.model_dims, M.heads, M.k_dims, initialize=W_q_dict, mutable=False)
        M.W_k = pyo.Param(M.model_dims, M.heads, M.k_dims, initialize=W_k_dict, mutable=False)
        M.W_v = pyo.Param(M.model_dims, M.heads, M.k_dims, initialize=W_v_dict, mutable=False)
        M.W_o = pyo.Param(M.model_dims,M.heads, M.k_dims, initialize=W_o_dict, mutable=False)
       
        if b_q:
            b_q_dict = {
                        (h, k): b_q[h-1][k-1]
                        for h in M.heads
                        for k in M.k_dims
                       }
            M.b_q = pyo.Param(M.heads, M.k_dims, initialize=b_q_dict, mutable=False)
            
        if b_k:
            b_k_dict = {
                        (h, k): b_k[h-1][k-1]
                        for h in M.heads
                        for k in M.k_dims
                       }
            M.b_k = pyo.Param(M.heads, M.k_dims, initialize=b_k_dict, mutable=False)
            
        if b_v: 
            b_v_dict = {
                        (h, k): b_v[h-1][k-1]
                        for h in M.heads
                        for k in M.k_dims
                       }
            M.b_v = pyo.Param(M.heads, M.k_dims, initialize=b_v_dict, mutable=False)
            
        if b_o:
            b_o_dict = {(d): val for d, val in zip(M.model_dims, b_o) }
            M.b_o = pyo.Param(M.model_dims, initialize=b_o_dict, mutable=False)
            

        M.Q = pyo.Var(M.heads, M.time_input, M.k_dims, within=pyo.Reals) 
        M.K = pyo.Var(M.heads, M.time_input, M.k_dims, within=pyo.Reals)
        M.V = pyo.Var(M.heads, M.time_input, M.k_dims, within=pyo.Reals) 

        M.compatibility = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals) 
        M.compatibility_pos = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,None)) 
        M.compatibility_neg = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,None)) 
        
        M.compatibility_exp = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,None)) # range: 0-->inf, initialize=init_compatibility_exp)
        M.compatibility_exp_sum = pyo.Var(M.heads, M.time_input, bounds=(0,None)) #, initialize=init_compatibility_sum)
        # M.tie_point_cc = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.tie_point_cv = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.tie_point_cc_prime = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.tie_point_cv_prime = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.cv_prime_exp = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,None))
        # M.cv_exp = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,None))
        
        # BigM_s = 0.5
        # BigM_t = 1
        # M.tp_cv =pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Binary)
        
        # M.s_cc= pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Binary)
        # M.s_cv= pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Binary)
        # M.t_cc= pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Binary)
        # M.t_cv= pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Binary)
        
        # M.sct = pyo.Var(M.heads, M.time_input, M.time_input)
        
        # M.attention_weight_cc = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,1))
        # M.attention_weight_x_cc_prime = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,1))
        # M.attention_weight_x_cc= pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,1))
        
        # M.attention_weight_cv = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,1))
        # M.attention_weight_x_cv_prime = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,1))
        # M.attention_weight_x_cv = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,1))
        
        M.attention_weight = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,1))  # softmax ( (Q * K)/sqrt(d_k) )
        
        # M.tp_grad_cv = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.tp_numerator = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.tp_denominator = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.tp_sct_numerator = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.tp_sct_denominator = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.tp_sct_frac = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.tp_sct_frac_scaled = pyo.Var(M.heads, M.time_input, M.time_input)
        # M.tp_sct_diff = pyo.Var(M.heads, M.time_input, M.time_input)
        
        M.attention_score = pyo.Var(
            M.heads, M.time_input, M.k_dims, within=pyo.Reals
        )  # softmax ( (Q * K)/sqrt(d_k) ) * V
        M.attention_output = pyo.Var(
            M.time_input, M.model_dims, within=pyo.Reals #, bounds=(-1,1)
        )  # concat heads and linear transform

        for h in M.heads:
            for n in M.time_input:
                    for k in M.k_dims:
                        
                         # constraints for Query
                        if b_q:
                            M.attention_constraints.add(
                            expr=M.Q[h, n, k]
                            == sum(input_var[n,d] * M.W_q[d, h, k] for d in M.model_dims) + M.b_q[h,k] 
                            )  
                            #Add bounds
                            q_bound_1 = sum( max(input_var[n,d].ub * M.W_q[d, h, k], input_var[n,d].lb * M.W_q[d, h, k])  for d in M.model_dims) + M.b_q[h,k]
                            q_bound_2 = sum( min(input_var[n,d].ub * M.W_q[d, h, k], input_var[n,d].lb * M.W_q[d, h, k])  for d in M.model_dims) + M.b_q[h,k]
                            if q_bound_1 < q_bound_2: 
                                M.Q[h, n, k].ub = q_bound_2
                                M.Q[h, n, k].lb = q_bound_1
                            else:
                                M.Q[h, n, k].ub = q_bound_1
                                M.Q[h, n, k].lb = q_bound_2
                                
                            # print("bounds")
                            # print("--", input_var[n,'0'].lb, input_var[n,'0'].ub, M.W_q['0', h, k])
                            # print("--", input_var[n,'1'].lb, input_var[n,'1'].ub, M.W_q['1', h, k])
                            # print(q_bound_1, q_bound_2)
                            # print(M.Q_pos[h, n, k].ub)

                        else: 
                            M.attention_constraints.add(
                                expr=M.Q[h, n, k]
                                == sum(input_var[n, d] * M.W_q[d, h, k] for d in M.model_dims)
                            )
                            #Add bounds
                            q_bound_1 = sum( max(input_var[n,d].ub * M.W_q[d, h, k], input_var[n,d].lb * M.W_q[d, h, k])  for d in M.model_dims)
                            q_bound_2 = sum( min(input_var[n,d].ub * M.W_q[d, h, k], input_var[n,d].lb * M.W_q[d, h, k])  for d in M.model_dims)
                            if q_bound_1 < q_bound_2: 
                                M.Q[h, n, k].ub = q_bound_2
                                M.Q[h, n, k].lb = q_bound_1
                            else:
                                M.Q[h, n, k].ub = q_bound_1
                                M.Q[h, n, k].lb = q_bound_2
                              
                        # constraints for Key
                        if b_k:
                            M.attention_constraints.add(
                            expr=M.K[h, n, k]
                            == sum(input_var[n, d] * M.W_k[d, h, k] for d in M.model_dims) + M.b_k[h,k]
                            )  
                            #Add bounds
                            k_bound_1 = sum( max(input_var[n,d].ub * M.W_k[d, h, k], input_var[n,d].lb * M.W_k[d, h, k])  for d in M.model_dims) + M.b_k[h,k]
                            k_bound_2 = sum( min(input_var[n,d].ub * M.W_k[d, h, k], input_var[n,d].lb * M.W_k[d, h, k])  for d in M.model_dims) + M.b_k[h,k]
                            if k_bound_1 < k_bound_2: 
                                M.K[h, n, k].ub = k_bound_2
                                M.K[h, n, k].lb = k_bound_1
                            else:
                                M.K[h, n, k].ub = k_bound_1
                                M.K[h, n, k].lb = k_bound_2
                            
                        else: 
                            M.attention_constraints.add(
                                expr=M.K[h, n, k]
                                == sum(input_var[n, d] * M.W_k[d, h, k] for d in M.model_dims)
                            )
                            #Add bounds
                            k_bound_1 = sum( max(input_var[n,d].ub * M.W_k[d, h, k], input_var[n,d].lb * M.W_k[d, h, k])  for d in M.model_dims) 
                            k_bound_2 = sum( min(input_var[n,d].ub * M.W_k[d, h, k], input_var[n,d].lb * M.W_k[d, h, k])  for d in M.model_dims) 
                            if k_bound_1 < k_bound_2: 
                                M.K[h, n, k].ub = k_bound_2
                                M.K[h, n, k].lb = k_bound_1
                            else:
                                M.K[h, n, k].ub = k_bound_1
                                M.K[h, n, k].lb = k_bound_2
                            
                        # constraints for Value    
                        if b_v:
                            M.attention_constraints.add(
                            expr=M.V[h, n, k]
                            == sum(input_var[n, d] * M.W_v[d, h, k] for d in M.model_dims) + M.b_v[h,k]
                            )  
                            #Add bounds
                            
                            v_bound_1 = sum( max(input_var[n,d].ub * M.W_v[d, h, k], input_var[n,d].lb * M.W_v[d, h, k])  for d in M.model_dims) + M.b_v[h,k]
                            v_bound_2 = sum( min(input_var[n,d].ub * M.W_v[d, h, k], input_var[n,d].lb * M.W_v[d, h, k])  for d in M.model_dims) + M.b_v[h,k]
                            if v_bound_1 < v_bound_2: 
                                M.V[h, n, k].ub = v_bound_2
                                M.V[h, n, k].lb = v_bound_1
                            else:
                                M.V[h, n, k].ub = v_bound_1
                                M.V[h, n, k].lb = v_bound_2
                            
                        else: 
                            M.attention_constraints.add(
                                expr=M.V[h, n, k]
                                == sum(input_var[n, d] * M.W_v[d, h, k] for d in M.model_dims) 
                            )
                            #Add bounds     
                            v_bound_1 = sum( max(input_var[n,d].ub * M.W_v[d, h, k], input_var[n,d].lb * M.W_v[d, h, k])  for d in M.model_dims)
                            v_bound_2 = sum( min(input_var[n,d].ub * M.W_v[d, h, k], input_var[n,d].lb * M.W_v[d, h, k])  for d in M.model_dims)
                            if v_bound_1 < v_bound_2: 
                                M.V[h, n, k].ub = v_bound_2
                                M.V[h, n, k].lb = v_bound_1
                            else:
                                M.V[h, n, k].ub = v_bound_1
                                M.V[h, n, k].lb = v_bound_2

                        # attention score = sum(attention_weight * V)
                        M.attention_constraints.add(
                            expr=M.attention_score[h, n, k]
                            == sum(
                                M.attention_weight[h, n, n2] * M.V[h, n2, k]
                                for n2 in M.time_input
                            )
                        )
                        
                        
                    for p in M.time_input:
                        # compatibility sqrt(Q * K) across all pairs of elements
                        scale = np.sqrt(self.d_k) 
                        # M.attention_constraints.add(
                        #     expr= M.tp_grad_cv[h, n, p] * M.tp_denominator[h, n, p] == M.tp_numerator[h, n, p]
                        # )

                        M.attention_constraints.add(
                            expr=M.compatibility[h, n, p] *scale
                            == sum(M.Q[h, n, k] * (M.K[ h, p, k] )for k in M.k_dims)
                        )  
                        
                        M.attention_constraints.add(expr= pyo.exp(M.compatibility[h,n,p]) == M.compatibility_exp[h, n, p] )
                        # M.constr_convex_tp_sct.add( # otherwise when x > x_cv
                        #     M.tp_sct_frac[h, n, p] * M.tp_sct_denominator[h, n, p] == M.tp_sct_numerator[h, n, p] *  M.tp_sct_diff [h, n, p] 
                        # )
                        # M.constr_convex_tp_sct.add( # otherwise when x > x_cv
                        #     M.tp_sct_frac_scaled[h, n, p]  == M.tp_sct_frac[h, n, p] 
                        # )
                        # M.attention_constraints.add(
                        #     expr=   pyo.exp(M.tie_point_cv_prime[h, n, p]) == M.cv_prime_exp[h, n, p] 
                        # )
                        # # M.attention_constraints.add(
                        # #     expr=   pyo.exp(M.tie_point_cv[h, n, p]) == M.cv_exp[h, n, p] 
                        # # )
                        # M.attention_constraints.add(
                        #     expr= M.attention_weight_x_cv_prime[h, n, p] * ( 1 - M.attention_weight_x_cv_prime[h, n, p]) == M.tp_grad_cv[h, n, p]
                        # )
                        
                        

                        
                    M.attention_constraints.add(expr= M.compatibility_exp_sum[h, n] == sum(M.compatibility_exp[h, n, p] for p in M.time_input))
                    
                    for n2 in M.time_input:

                        # attention weights softmax(compatibility)   
                        M.attention_constraints.add(
                            expr=M.attention_weight[h, n, n2] * M.compatibility_exp_sum[h, n]
                            == M.compatibility_exp[h, n, n2]) 
                        M.attention_constraints.add(
                            expr= M.attention_weight[h, n, n2] <= 1
                        )
                        
                        # # att(cv_prime)
                        # M.attention_constraints.add(
                        #     expr=  M.attention_weight_x_cv_prime[h, n, n2] * M.compatibility_exp_sum[h, n] == M.cv_prime_exp[h, n, n2]
                        # )
                        
                        # # att(x_cv)
                        # M.attention_constraints.add(
                        #     expr=  M.attention_weight_x_cv[h, n, n2] * M.compatibility_exp_sum[h, n]
                        #         == M.cv_exp[h, n, n2] 
                        # )


                    # sum over softmax = 1    
                    M.attention_constraints.add(
                        expr=sum(M.attention_weight[h, n, n_prime] for n_prime in M.time_input) == 1
                    )
                    # M.attention_constraints.add(
                    #     expr=sum(M.attention_weight_cc[h, n, n_prime] for n_prime in M.time_input) == 1
                    # )
                   
                    
                    
            #Add bounds            
            for n in M.time_input:
                for p in M.time_input:
                    M.attention_constraints.add(
                                expr=M.compatibility[h,n,p] == M.compatibility_pos[h,n,p] - M.compatibility_neg[h,n,p] 
                            )
                    M.compatibility_pos[h,n,p].ub = (1/scale ) * (sum( (M.Q[h, n, k].ub)**2 for k in M.k_dims)**0.5) * (sum( (M.K[h, n, k].ub)**2 for k in M.k_dims)**0.5)
                    M.compatibility_neg[h,n,p].ub = M.compatibility_pos[h,n,p].ub
                    M.compatibility[h,n,p].ub = M.compatibility_pos[h,n,p].ub
                    M.compatibility[h,n,p].lb = -M.compatibility_pos[h,n,p].ub
                    
                    M.compatibility_exp[h,n,p].ub = math.exp(M.compatibility[h,n,p].ub)
                    M.compatibility_exp[h,n,p].lb = math.exp(M.compatibility[h,n,p].lb)
                    
                    # # define M.tie_point_cv_prime[h, n, p]
                    # M.attention_constraints.add(
                    #     expr=  M.attention_weight[h, n, p].ub - M.attention_weight_x_cv_prime[h, n, p] == M.tp_numerator[h, n, p]
                    # )
                    # M.attention_constraints.add(
                    #     expr= M.compatibility[h,n,p].ub - M.tie_point_cv_prime[h, n, p] == M.tp_denominator[h, n, p]
                    # )
                    
                    

                M.compatibility_exp_sum[h, n].ub = sum( M.compatibility_exp[h,n,p].ub for p in M.time_input) 
                M.compatibility_exp_sum[h, n].lb = sum( M.compatibility_exp[h,n,p].lb for p in M.time_input) 
                
                    
                ##############-----------------------------------############    
                for p in M.time_input:    
                    M.attention_weight[h, n, p].ub = M.compatibility_exp[h,n,p].ub / (M.compatibility_exp_sum[h, n].lb  - M.compatibility_exp[h,n,p].lb + M.compatibility_exp[h,n,p].ub  + 0.00000001)
                    M.attention_weight[h, n, p].lb = M.compatibility_exp[h,n,p].lb / (M.compatibility_exp_sum[h, n].ub - M.compatibility_exp[h,n,p].ub + M.compatibility_exp[h,n,p].lb + 0.00000001)
                    # print("compat", M.compatibility[h,n,p].ub)
                    # print("1:", M.compatibility_exp[h,n,p].ub , M.compatibility_exp_sum[h, n].ub)
                    # print(M.attention_weight[h, n, p].ub)
                    # print("compat l", M.compatibility[h,n,p].lb)
                    # print("2:", M.compatibility_exp[h,n,p].lb , M.compatibility_exp_sum[h, n].lb)
                    # print(M.attention_weight[h, n, p].ub, M.attention_weight[h, n, p].lb)
                    
           
                    # # Concave/convex envelope
                    # # f(x_UB) <= 0.5 --> convex zone else f(x_UB) >= 0.5 --> concave zone
                    # M.attention_constraints.add(
                    #     expr= M.attention_weight[h, n, p].ub <= 0.5  + (BigM_s * M.s_cv[h,n,p])
                    # )

                    # # f(x_LB) <= 0.5 --> convex zone else f(x_LB) >= 0.5 --> concave zone
                    # M.attention_constraints.add(
                    #     expr= M.attention_weight[h, n, p].lb >= 0.5  - (BigM_s *  M.s_cc[h,n,p])
                    # )
                    
                    
                    # ## x <= x_cv_m -> convex zone
                    # # M.tie_point_cv[h, n, p] = max(M.tie_point_cv_prime[h, n, p], M.compatibility[h,n,p].lb  )
                    # BigM_cv_prime = sum( M.compatibility[h,n,p_prime].ub for p_prime in M.time_input)*100
                    # M.attention_constraints.add(
                    #     M.tie_point_cv_prime[h, n, p] - M.compatibility[h,n,p].lb <= BigM_cv_prime * (1 - M.tp_cv[h,n,p])
                    # )
                    # M.attention_constraints.add(
                    #     M.tie_point_cv[h, n, p]  <= M.tie_point_cv_prime[h, n, p]*(1 - M.tp_cv[h,n,p])  + (M.compatibility[h,n,p].lb * M.tp_cv[h,n,p])
                    # )
                    # # M.attention_constraints.add(
                    # #     M.tie_point_cv[h, n, p]  <= M.compatibility[h,n,p].lb + (BigM_cv_prime * M.tp_cv[h,n,p])
                    # # )
                    # M.attention_constraints.add(
                    #     M.tie_point_cv[h, n, p]  >= M.tie_point_cv_prime[h, n, p]*(1 - M.tp_cv[h,n,p])  + (M.compatibility[h,n,p].lb * M.tp_cv[h,n,p])
                    # )
                    # # M.attention_constraints.add(
                    # #     M.tie_point_cv[h, n, p]  >= M.tie_point_cv_prime[h, n, p]
                    # # )
                    # # M.tie_point_cv[h, n, p].lb  == M.compatibility[h,n,p].lb
                    
                    # M.attention_constraints.add(
                    #     M.attention_weight_x_cv[h, n, p] <= M.attention_weight_x_cv_prime[h, n, p]*(1 - M.tp_cv[h,n,p])  + M.attention_weight[h,n,p].lb * M.tp_cv[h,n,p]
                    # )
                    # M.attention_constraints.add(
                    #     M.attention_weight_x_cv[h, n, p] >= M.attention_weight_x_cv_prime[h, n, p]*(1 - M.tp_cv[h,n,p])  + M.attention_weight[h,n,p].lb * M.tp_cv[h,n,p]
                    # )
                    # # x <= x_tie_point_cv ? true --> convex zone
                    # M.attention_constraints.add(
                    #     expr=  M.tie_point_cv[h, n, p] - M.compatibility[h,n,p] <= BigM_cv_prime * (1-M.t_cv[h,n,p])
                    # )
                    # # M.attention_constraints.add(
                    # #     expr=  M.tie_point_cv[h, n, p] - M.compatibility[h,n,p] >= - BigM_cv_prime * (M.t_cv[h,n,p])
                    # # )
                    
                    
                    
                    # # f(x) >= f_cv(x)
                    # # M.attention_constraints.add(
                    # #     M.attention_weight[h, n, p]  >= M.attention_weight_cv[h, n, p]
                    # # )
                    # # f(x) <= f_cc(x)
                    # M.attention_constraints.add(
                    #     M.attention_weight[h, n, p]  <= M.attention_weight_cc[h, n, p]
                    # )
                    
                    # # sct(x)
                    # A = ((M.attention_weight[h, n, p].ub - M.attention_weight[h, n, p].lb) / (M.compatibility[h,n,p].ub - M.compatibility[h,n,p].lb )) 
                    # b = ( (M.compatibility[h,n,p].ub * M.attention_weight[h, n, p].lb) - (M.compatibility[h,n,p].lb * M.attention_weight[h, n, p].ub)) /(M.compatibility[h,n,p].ub - M.compatibility[h,n,p].lb )
                    # M.attention_constraints.add(
                    #     M.sct[h, n, p]   == (A *  M.compatibility[h,n,p]) - b
                    # )

                    
                    # # # Add convex function constraints
                    # M.constr_convex.add( # when f(UB) <= 0.5 
                    #     M.attention_weight_cv[h, n, p] == M.attention_weight[h, n, p]
                    # )
                    # M.constr_concave.add( # when f(LB) >= 0
                    #     M.attention_weight_cv[h, n, p] == M.sct[h, n, p] 
                    # )
                    # M.constr_convex_tp.add( # otherwise when x <= x_cv
                    #     M.attention_weight_cv[h, n, p] == M.attention_weight[h, n, p]
                    # )
                    
                    
                    # M.constr_convex_tp_sct.add( # otherwise
                    #     M.tp_sct_numerator[h, n, p] == (M.attention_weight[h, n, p].ub - M.attention_weight_x_cv[h, n, p])
                    # )
                    # M.constr_convex_tp_sct.add( # otherwise when x > x_cv
                    #     M.attention_weight_cv[h, n, p]  == M.attention_weight[h, n, p].ub + M.tp_sct_frac_scaled[h, n, p]
                    # )
                    # M.constr_convex_tp_sct.add( # otherwise
                    #     M.tp_sct_denominator[h, n, p] == M.compatibility[h,n,p].ub - M.tie_point_cv[h, n, p]
                    # )
                    # M.constr_convex_tp_sct.add( # otherwise when x > x_cv
                    #     M.tp_sct_diff [h, n, p] == (M.compatibility[h,n,p] - M.compatibility[h,n,p].ub)
                    # )
                   
                    
                    # # Add concave function constraints
                    # M.constr_convex.add( # when f(UB) <= 0.5 
                    #     M.attention_weight_cc[h, n, p] == M.sct[h, n, p] 
                    # )
                    # M.constr_concave.add( # when f(LB) >= 0
                    #     M.attention_weight_cc[h, n, p] == M.attention_weight[h, n, p] 
                    # )
                    # M.constr_concave_tp.add( # otherwise
                    #     M.attention_weight_cc[h, n, p] == M.attention_weight[h, n, p] 
                    # )
                    
                    # M.constr_convex.deactivate() #deactivate all F_convex constraints
                    # M.constr_concave.deactivate() 
                    # M.constr_convex_other.deactivate()
   
        # multihead attention output constraint
        for n in M.time_input:
            for d in M.model_dims:
                if b_o:
                    M.attention_constraints.add(
                        expr=M.attention_output[n, d]
                        == sum(
                            (sum(
                            M.attention_score[h, n, k] * M.W_o[d,h, k]
                            for k in M.k_dims
                             ) )
                        for h in M.heads
                        
                        ) + M.b_o[d]
                    )
                    
                    
                else:
                    M.attention_constraints.add(
                        expr=M.attention_output[n, d]
                        == sum(
                            (sum(
                            M.attention_score[h, n, k] * M.W_o[d,h, k]
                            for k in M.k_dims
                             ) )
                        for h in M.heads
                        )
                    )
                    # M.attention_output[n, d].ub  = (self.d_H * sum(M.attention_score[h, n, k].ub * M.W_o[d,h, k] for k in M.k_dims))
                    # M.attention_output[n, d].lb  = (self.d_H * sum(M.attention_score[h, n, k].lb * M.W_o[d,h, k] for k in M.k_dims))
        
        # # activate softmax envelope constraints              
        # M.activate_constraints = pyo.BuildAction(rule=activate_envelope_att)            
                
    # def add_attention_approx(self, M, input_var_name, W_q, W_k, W_v, W_o, b_q = None, b_k = None, b_v = None, b_o = None):
    #     """
    #     Multihead attention between each element of embedded sequence
        
    #     Exp function created using power series approximation (11 elements of power series). 
    #     This formulation avoids the pyomo solving error when calculating pyo.exp(pyo.Var())
    #     """
    #     if not hasattr(M, "attention_constraints"):
    #         M.attention_constraints = pyo.ConstraintList()
            
    #     input_var = getattr(M, input_var_name)

    #     # define sets, vars
    #     M.heads = pyo.RangeSet(1, self.d_H)
    #     M.k_dims = pyo.RangeSet(1, self.d_k)

    #     W_q_dict = {
    #         (D, H, K): W_q[d][h][k]
    #         for d,D in enumerate(M.model_dims)
    #         for h,H in enumerate(M.heads)
    #         for k,K in enumerate(M.k_dims)
    #     }
    #     W_k_dict = {
    #         (D, H, K): W_k[d][h][k]
    #         for d,D in enumerate(M.model_dims)
    #         for h,H in enumerate(M.heads)
    #         for k,K in enumerate(M.k_dims)
    #     }
    #     W_v_dict = {
    #         (D, H, K): W_v[d][h][k]
    #         for d,D in enumerate(M.model_dims)
    #         for h,H in enumerate(M.heads)
    #         for k,K in enumerate(M.k_dims)
    #     }
    #     W_o_dict = {
    #         (D, H, K): W_o[h][k][d]
    #         for d,D in enumerate(M.model_dims)
    #         for h,H in enumerate(M.heads)
    #         for k,K in enumerate(M.k_dims)
    #     }
 
    #     M.W_q = pyo.Param(M.model_dims, M.heads, M.k_dims, initialize=W_q_dict, mutable=False)
    #     M.W_k = pyo.Param(M.model_dims, M.heads, M.k_dims, initialize=W_k_dict, mutable=False)
    #     M.W_v = pyo.Param(M.model_dims, M.heads, M.k_dims, initialize=W_v_dict, mutable=False)
    #     M.W_o = pyo.Param(M.model_dims,M.heads, M.k_dims, initialize=W_o_dict, mutable=False)
       
    #     if b_q:
    #         b_q_dict = {
    #                     (h, k): b_q[h-1][k-1]
    #                     for h in M.heads
    #                     for k in M.k_dims
    #                    }
    #         M.b_q = pyo.Param(M.heads, M.k_dims, initialize=b_q_dict, mutable=False)
            
    #     if b_k:
    #         b_k_dict = {
    #                     (h, k): b_k[h-1][k-1]
    #                     for h in M.heads
    #                     for k in M.k_dims
    #                    }
    #         M.b_k = pyo.Param(M.heads, M.k_dims, initialize=b_k_dict, mutable=False)
            
    #     if b_v: 
    #         b_v_dict = {
    #                     (h, k): b_v[h-1][k-1]
    #                     for h in M.heads
    #                     for k in M.k_dims
    #                    }
    #         M.b_v = pyo.Param(M.heads, M.k_dims, initialize=b_v_dict, mutable=False)
            
    #     if b_o:
    #         b_o_dict = {(d): val for d, val in zip(M.model_dims, b_o) }
    #         M.b_o = pyo.Param(M.model_dims, initialize=b_o_dict, mutable=False)
            

    #     M.Q = pyo.Var(M.heads, M.time_input, M.k_dims, within=pyo.Reals) 
    
    #     M.K = pyo.Var(M.heads, M.time_input, M.k_dims, within=pyo.Reals)
        
    #     M.V = pyo.Var(M.heads, M.time_input, M.k_dims, within=pyo.Reals) 
        
        
    #     #init_compatibility = {
    #                 #     (H, T, P): 1
    #                 #     for h,H in enumerate(M.heads)
    #                 #     for n,T in enumerate(M.time_input)
    #                 #     for p,P in enumerate(M.time_input)
    #                 #    }
    #     M.compatibility = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals) #, initialize=init_compatibility, bounds=(-10,10))  # sqrt(Q * K)
    #     M.compatibility_exp = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.NonNegativeReals, bounds=(0,None)) # range: 0-->inf, initialize=init_compatibility_exp)
    #     M.compatibility_exp_sum = pyo.Var(M.heads, M.time_input) #, initialize=init_compatibility_sum)
    #     M.compatibility_squ = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals)
    #     M.compatibility_3 = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals)
    #     M.compatibility_4 = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals)
    #     M.compatibility_5 = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals)
    #     M.compatibility_6 = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals)
    #     M.compatibility_7 = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals)
    #     M.compatibility_8 = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals)
    #     M.compatibility_9 = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals)
    #     M.compatibility_10 = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals)
    #     M.compatibility_11 = pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Reals)
        
    #     M.compatibility_pos = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,None)) 
    #     M.compatibility_neg = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,None)) 
        
    #     # M.tie_point_cc = pyo.Var(M.heads, M.time_input, M.time_input)
    #     # M.tie_point_cv = pyo.Var(M.heads, M.time_input, M.time_input)
    #     BigM_s = 1
    #     # BigM_t = 1
    #     # M.s_cc= pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Binary)
    #     M.s_cv= pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Binary)
    #     # M.t_cc= pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Binary)
    #     # M.t_cv= pyo.Var(M.heads, M.time_input, M.time_input, within=pyo.Binary)
    #     # M.attention_weight_cc = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,1))
    #     # M.attention_weight_cv = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,1))

    #     M.attention_weight = pyo.Var(M.heads, M.time_input, M.time_input, bounds=(0,1))  # softmax ( (Q * K)/sqrt(d_k) )
    #     M.attention_score = pyo.Var(
    #         M.heads, M.time_input, M.k_dims, within=pyo.Reals
    #     )  # softmax ( (Q * K)/sqrt(d_k) ) * V
    #     M.attention_output = pyo.Var(
    #         M.time_input, M.model_dims, within=pyo.Reals
    #     )  # concat heads and linear transform

    #     for h in M.heads:
    #         for n in M.time_input:
    #                 for k in M.k_dims:
                        
    #                      # constraints for Query
    #                     if b_q:
    #                         M.attention_constraints.add(
    #                         expr=M.Q[h, n, k]
    #                         == sum(input_var[n,d] * M.W_q[d, h, k] for d in M.model_dims) + M.b_q[h,k] 
    #                         )  
    #                         #Add bounds
    #                         q_bound_1 = sum( max(input_var[n,d].ub * M.W_q[d, h, k], input_var[n,d].lb * M.W_q[d, h, k])  for d in M.model_dims) + M.b_q[h,k]
    #                         q_bound_2 = sum( min(input_var[n,d].ub * M.W_q[d, h, k], input_var[n,d].lb * M.W_q[d, h, k])  for d in M.model_dims) + M.b_q[h,k]
    #                         if q_bound_1 < q_bound_2: 
    #                             M.Q[h, n, k].ub = q_bound_2
    #                             M.Q[h, n, k].lb = q_bound_1
    #                         else:
    #                             M.Q[h, n, k].ub = q_bound_1
    #                             M.Q[h, n, k].lb = q_bound_2
                                
    #                         # print("bounds")
    #                         # print("--", input_var[n,'0'].lb, input_var[n,'0'].ub, M.W_q['0', h, k])
    #                         # print("--", input_var[n,'1'].lb, input_var[n,'1'].ub, M.W_q['1', h, k])
    #                         # print(q_bound_1, q_bound_2)
    #                         # print(M.Q_pos[h, n, k].ub)

    #                     else: 
    #                         M.attention_constraints.add(
    #                             expr=M.Q[h, n, k]
    #                             == sum(input_var[n, d] * M.W_q[d, h, k] for d in M.model_dims)
    #                         )
    #                         #Add bounds
    #                         q_bound_1 = sum( max(input_var[n,d].ub * M.W_q[d, h, k], input_var[n,d].lb * M.W_q[d, h, k])  for d in M.model_dims)
    #                         q_bound_2 = sum( min(input_var[n,d].ub * M.W_q[d, h, k], input_var[n,d].lb * M.W_q[d, h, k])  for d in M.model_dims)
    #                         if q_bound_1 < q_bound_2: 
    #                             M.Q[h, n, k].ub = q_bound_2
    #                             M.Q[h, n, k].lb = q_bound_1
    #                         else:
    #                             M.Q[h, n, k].ub = q_bound_1
    #                             M.Q[h, n, k].lb = q_bound_2
                              
    #                     # constraints for Key
    #                     if b_k:
    #                         M.attention_constraints.add(
    #                         expr=M.K[h, n, k]
    #                         == sum(input_var[n, d] * M.W_k[d, h, k] for d in M.model_dims) + M.b_k[h,k]
    #                         )  
    #                         #Add bounds
    #                         k_bound_1 = sum( max(input_var[n,d].ub * M.W_k[d, h, k], input_var[n,d].lb * M.W_k[d, h, k])  for d in M.model_dims) + M.b_k[h,k]
    #                         k_bound_2 = sum( min(input_var[n,d].ub * M.W_k[d, h, k], input_var[n,d].lb * M.W_k[d, h, k])  for d in M.model_dims) + M.b_k[h,k]
    #                         if k_bound_1 < k_bound_2: 
    #                             M.K[h, n, k].ub = k_bound_2
    #                             M.K[h, n, k].lb = k_bound_1
    #                         else:
    #                             M.K[h, n, k].ub = k_bound_1
    #                             M.K[h, n, k].lb = k_bound_2
                            
    #                     else: 
    #                         M.attention_constraints.add(
    #                             expr=M.K[h, n, k]
    #                             == sum(input_var[n, d] * M.W_k[d, h, k] for d in M.model_dims)
    #                         )
    #                         #Add bounds
    #                         k_bound_1 = sum( max(input_var[n,d].ub * M.W_k[d, h, k], input_var[n,d].lb * M.W_k[d, h, k])  for d in M.model_dims) 
    #                         k_bound_2 = sum( min(input_var[n,d].ub * M.W_k[d, h, k], input_var[n,d].lb * M.W_k[d, h, k])  for d in M.model_dims) 
    #                         if k_bound_1 < k_bound_2: 
    #                             M.K[h, n, k].ub = k_bound_2
    #                             M.K[h, n, k].lb = k_bound_1
    #                         else:
    #                             M.K[h, n, k].ub = k_bound_1
    #                             M.K[h, n, k].lb = k_bound_2
                            
    #                     # constraints for Value    
    #                     if b_v:
    #                         M.attention_constraints.add(
    #                         expr=M.V[h, n, k]
    #                         == sum(input_var[n, d] * M.W_v[d, h, k] for d in M.model_dims) + M.b_v[h,k]
    #                         )  
    #                         #Add bounds
                            
    #                         v_bound_1 = sum( max(input_var[n,d].ub * M.W_v[d, h, k], input_var[n,d].lb * M.W_v[d, h, k])  for d in M.model_dims) + M.b_v[h,k]
    #                         v_bound_2 = sum( min(input_var[n,d].ub * M.W_v[d, h, k], input_var[n,d].lb * M.W_v[d, h, k])  for d in M.model_dims) + M.b_v[h,k]
    #                         if v_bound_1 < v_bound_2: 
    #                             M.V[h, n, k].ub = v_bound_2
    #                             M.V[h, n, k].lb = v_bound_1
    #                         else:
    #                             M.V[h, n, k].ub = v_bound_1
    #                             M.V[h, n, k].lb = v_bound_2
                            
    #                     else: 
    #                         M.attention_constraints.add(
    #                             expr=M.V[h, n, k]
    #                             == sum(input_var[n, d] * M.W_v[d, h, k] for d in M.model_dims) 
    #                         )
    #                         #Add bounds     
    #                         v_bound_1 = sum( max(input_var[n,d].ub * M.W_v[d, h, k], input_var[n,d].lb * M.W_v[d, h, k])  for d in M.model_dims)
    #                         v_bound_2 = sum( min(input_var[n,d].ub * M.W_v[d, h, k], input_var[n,d].lb * M.W_v[d, h, k])  for d in M.model_dims)
    #                         if v_bound_1 < v_bound_2: 
    #                             M.V[h, n, k].ub = v_bound_2
    #                             M.V[h, n, k].lb = v_bound_1
    #                         else:
    #                             M.V[h, n, k].ub = v_bound_1
    #                             M.V[h, n, k].lb = v_bound_2

    #                     # attention score = sum(attention_weight * V)
    #                     M.attention_constraints.add(
    #                         expr=M.attention_score[h, n, k]
    #                         == sum(
    #                             M.attention_weight[h, n, n2] * M.V[h, n2, k]
    #                             for n2 in M.time_input
    #                         )
    #                     )

                        
    #                 for p in M.time_input:
    #                     # compatibility sqrt(Q * K) across all pairs of elements
    #                     scale = np.sqrt(self.d_k) 

    #                     M.attention_constraints.add(
    #                         expr=M.compatibility[h, n, p] *scale
    #                         == sum(M.Q[h, n, k] * (M.K[ h, p, k] )for k in M.k_dims)
    #                     ) 
                        
                        
    # # # #                 # power series approx for EXP
    #                     M.attention_constraints.add(expr= M.compatibility[h, n, p]**2 == M.compatibility_squ[h, n, p] )#problem for gurobi
    #                     M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_squ[h, n, p] == M.compatibility_3[h, n, p] )
    #                     M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_3[h, n, p] == M.compatibility_4[h, n, p] )
    #                     M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_4[h, n, p] == M.compatibility_5[h, n, p] )
    #                     M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_5[h, n, p] == M.compatibility_6[h, n, p] )
    #                     M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_6[h, n, p] == M.compatibility_7[h, n, p] )
    #                     M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_7[h, n, p] == M.compatibility_8[h, n, p] )
    #                     M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_8[h, n, p] == M.compatibility_9[h, n, p] )
    #                     M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_9[h, n, p] == M.compatibility_10[h, n, p] )
    #                     M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_10[h, n, p] == M.compatibility_11[h, n, p] )
                        
    #                     M.attention_constraints.add(expr= M.compatibility_exp[h, n, p] == 1
    #                                                 + M.compatibility[h, n, p]
    #                                                 + (0.5*M.compatibility_squ[h, n, p] ) 
    #                                                 + (0.166666667*M.compatibility_3[h, n, p]) 
    #                                                 + (0.0416666667*M.compatibility_4[h, n, p]) 
    #                                                 + (0.00833333333*M.compatibility_5[h, n, p]) 
    #                                                 + (0.00138888889*M.compatibility_6[h, n, p]) 
    #                                                 + (0.000198412698*M.compatibility_7[h, n, p]) 
    #                                                 + (0.0000248015873*M.compatibility_8[h, n, p]) 
    #                                                 + (0.00000275573192*M.compatibility_9[h, n, p]) 
    #                                                 + (0.000000275573192*M.compatibility_10[h, n, p])
    #                                                 + (0.0000000250521084*M.compatibility_11[h, n, p])
    #                                                 )# pyo.exp() only seems to work for constant args and pow operator must be <= 2
                        
    #                 M.attention_constraints.add(expr= M.compatibility_exp_sum[h, n] == sum(M.compatibility_exp[h, n, p] for p in M.time_input))
                    
    #                 for n2 in M.time_input:

    #                     # attention weights softmax(compatibility)
    #                     M.attention_constraints.add(
    #                         expr=M.attention_weight[h, n, n2] * M.compatibility_exp_sum[h, n]
    #                         == M.compatibility_exp[h, n, n2]) 
                        
                        
    #                 # sum over softmax = 1    
    #                 M.attention_constraints.add(
    #                     expr=sum(M.attention_weight[h, n, n_prime] for n_prime in M.time_input) == 1
    #                 )
                   
                    
                    
    #         #Add bounds            
    #         for n in M.time_input:
    #             for p in M.time_input:
    #                 M.attention_constraints.add(
    #                             expr=M.compatibility[h,n,p] == M.compatibility_pos[h,n,p] - M.compatibility_neg[h,n,p] 
    #                         )
    #                 M.compatibility_pos[h,n,p].ub = (1/scale ) * (sum( (M.Q[h, n, k].ub)**2 for k in M.k_dims)**0.5) * (sum( (M.K[h, n, k].ub)**2 for k in M.k_dims)**0.5)
    #                 M.compatibility_neg[h,n,p].ub = M.compatibility_pos[h,n,p].ub
    #                 M.compatibility[h,n,p].ub = M.compatibility_pos[h,n,p].ub
    #                 M.compatibility[h,n,p].lb = -M.compatibility_pos[h,n,p].ub
                    
    #                 M.compatibility_exp[h,n,p].ub = math.exp(M.compatibility[h,n,p].ub)
    #                 M.compatibility_exp[h,n,p].lb = math.exp(M.compatibility[h,n,p].lb)
                    
    #             M.compatibility_exp_sum[h, n].ub = sum( M.compatibility_exp[h,n,p].ub for p in M.time_input) 
    #             M.compatibility_exp_sum[h, n].lb = sum( M.compatibility_exp[h,n,p].lb for p in M.time_input) 
                
                    
    #             ##############-----------------------------------############    
    #             for p in M.time_input:    
    #                 M.attention_weight[h, n, p].ub = M.compatibility_exp[h,n,p].ub / (M.compatibility_exp_sum[h, n].lb  - M.compatibility_exp[h,n,p].lb + M.compatibility_exp[h,n,p].ub  + 0.00000001)
    #                 M.attention_weight[h, n, p].lb = M.compatibility_exp[h,n,p].lb / (M.compatibility_exp_sum[h, n].ub - M.compatibility_exp[h,n,p].ub + M.compatibility_exp[h,n,p].lb + 0.00000001)
    #                 # print("compat", M.compatibility[h,n,p].ub)
    #                 # print("1:", M.compatibility_exp[h,n,p].ub , M.compatibility_exp_sum[h, n].ub)
    #                 # print(M.attention_weight[h, n, p].ub)
    #                 # print("compat l", M.compatibility[h,n,p].lb)
    #                 # print("2:", M.compatibility_exp[h,n,p].lb , M.compatibility_exp_sum[h, n].lb)
    #                 # print(M.attention_weight[h, n, p].lb)
    #                 # Concave/convex envelope

    #                 #f(x_UB) <= 0.5
    #                 M.attention_constraints.add(
    #                     expr= M.attention_weight[h, n, n2].ub <= 0.5  + (BigM_s * M.s_cv[h,n,p])
    #                 )
    #                 # # f(x_UB) >= 0.5
    #                 # M.attention_constraints.add(
    #                 #      expr= M.compatibility_exp[h,n,p].ub/sum( M.compatibility_exp[h,n,n2].ub for n2 in M.time_input) >= 0.5  - (BigM_s * M.s_cv[h,n,p])
    #                 # )
                
            
                    
    #     # multihead attention output constraint
    #     for n in M.time_input:
    #         for d in M.model_dims:
    #             if b_o:
    #                 M.attention_constraints.add(
    #                     expr=M.attention_output[n, d]
    #                     == sum(
    #                         (sum(
    #                         M.attention_score[h, n, k] * M.W_o[d,h, k]
    #                         for k in M.k_dims
    #                          ) )
    #                     for h in M.heads
                        
    #                     ) + M.b_o[d]
    #                 )
                    
                    
    #             else:
    #                 M.attention_constraints.add(
    #                     expr=M.attention_output[n, d]
    #                     == sum(
    #                         (sum(
    #                         M.attention_score[h, n, k] * M.W_o[d,h, k]
    #                         for k in M.k_dims
    #                          ) )
    #                     for h in M.heads
    #                     )
    #                 )
    #                 # M.attention_output[n, d].ub  = (self.d_H * sum(M.attention_score[h, n, k].ub * M.W_o[d,h, k] for k in M.k_dims))
    #                 # M.attention_output[n, d].lb  = (self.d_H * sum(M.attention_score[h, n, k].lb * M.W_o[d,h, k] for k in M.k_dims))
                
                

    def add_residual_connection(self,M, input_1_name, input_2_name, output_var_name):
        # create constraint list
        if not hasattr(M, "residual_constraints"):
            M.residual_constraints = pyo.ConstraintList()
        
        # add new variable
        if not hasattr(M, output_var_name):
            setattr(M, output_var_name, pyo.Var(M.time_input, M.model_dims, within=pyo.Reals))
            residual_var = getattr(M, output_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')
        
        input_1 = getattr(M, input_1_name)
        input_2 = getattr(M, input_2_name)
        
        for n in M.time_input:
            for d in M.model_dims:
                M.residual_constraints.add(expr= residual_var[n,d] == input_1[n,d] + input_2[n,d])
                try:
                    residual_var[n,d].ub == input_1[n,d].ub + input_2[n,d].ub
                    residual_var[n,d].lb == input_1[n,d].lb + input_2[n,d].lb
                except:
                    continue
                
    def add_FFN_2D(self,M, input_var_name, output_var_name, input_shape, model_parameters):
        input_var = getattr(M, input_var_name)

        # add new variable
        if not hasattr(M, output_var_name + "_NN_Block"):
            NN_name = output_var_name + "_NN_Block"
            setattr(M, NN_name, OmltBlock())
            NN_block = getattr(M, NN_name)
            
            setattr(M, output_var_name, pyo.Var(input_var.index_set(), within=pyo.Reals))
            output_var = getattr(M, output_var_name)
            
            setattr(M, output_var_name+"_constraints", pyo.ConstraintList())
            ffn_constraints = getattr(M, output_var_name+"_constraints")
        else:
            raise ValueError('Attempting to overwrite variable')
        
        ###### GET BOUNDS
        input_bounds={0: (-4,4), 1: (-4,4), 2: (-4,4), 3:(-4,4), 4:(-4,4), 5: (-4,4), 6: (-4,4), 7: (-4,4), 8: (-4,4), 9: (-4,4)} ### fix input bounds
        net_relu = OMLT_helper.weights_to_NetDef(output_var_name, input_shape, model_parameters, input_bounds)
        NN_block.build_formulation(ReluBigMFormulation(net_relu))
        
        # Set input constraints
        input_indices_len, input_indices_attr = self.get_indices(M, input_var)
        if input_indices_len == 1:
            for i, index in  enumerate(input_indices_attr[0]):
                ffn_constraints.add(expr= input_var[index] == NN_block.inputs[i])
        elif input_indices_len == 2:
            for i, i_index in  enumerate(input_indices_attr[0]):
                for j, j_index in  enumerate(input_indices_attr[1]):
                    ffn_constraints.add(expr= input_var[i_index, j_index] == NN_block.inputs[j])
                    
                    
        # Set output constraints
        output_indices_len, output_indices_attr = self.get_indices(M, output_var)
        if output_indices_len == 1:
            for i, index in  enumerate(output_indices_attr[0]):
                ffn_constraints.add(expr= output_var[index] == NN_block.outputs[i])
        elif output_indices_len == 2:
            for i, i_index in  enumerate(output_indices_attr[0]):
                for j, j_index in  enumerate(output_indices_attr[1]):
                    ffn_constraints.add(expr= output_var[i_index, j_index] == NN_block.outputs[j])
            
    def get_fnn(self,M, input_var_name, output_var_name, input_shape, model_parameters):
        input_var = getattr(M, input_var_name)
        
        # add new variable
        if not hasattr(M, output_var_name + "_NN_Block"):
            
            setattr(M, output_var_name, pyo.Var(input_var.index_set(), within=pyo.Reals))
            output_var = getattr(M, output_var_name)
            
            setattr(M, output_var_name+"_constraints", pyo.ConstraintList())
            ffn_constraints = getattr(M, output_var_name+"_constraints")
        else:
            raise ValueError('Attempting to overwrite variable')
        
        nn= GUROBI_ML_helper.weights_to_NetDef(output_var_name, input_shape, model_parameters)
       
        return nn, input_var, output_var
            
        
    def get_indices(self, M, input_var):
        # Get indices of var
        indices = str(input_var.index_set()).split('*')
        indices_len = len(indices)
        indices_attr = []
        for i in indices:
            try: 
                indices_attr += [getattr(M, i)]
            except:
                raise ValueError('Input variable not indexed by a pyomo Set')
        
        return indices_len, indices_attr
        
    def add_avg_pool(self,M, input_var_name, output_var_name):
        input_var = getattr(M, input_var_name)
        M.avg_pool_constraints = pyo.ConstraintList()

        # add new variable
        if not hasattr(M, output_var_name):
            setattr(M, output_var_name, pyo.Var(M.model_dims, within=pyo.Reals))
            output_var = getattr(M, output_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')


        for d in M.model_dims: 
            M.avg_pool_constraints.add(expr= output_var[d] * self.N == sum(input_var[t,d] for t in M.time_input))
            
            try:
                output_var[d].ub  == sum(input_var[t,d].ub for t in M.time_input) / self.N
                output_var[d].lb  == sum(input_var[t,d].lb for t in M.time_input) / self.N
            except:
                continue
            
    #def add_output_constraints(self, M, input_var):
        # if not hasattr(M, "output_constraints"):
        #     M.output_constraints = pyo.ConstraintList()

        # # predict x, u
        # output = np.ones((len(M.time), self.d_model))
        # dict_output = {(t, str(d)): output[i, d] for i, t in enumerate(M.time) for d in range(len(M.model_dims))}
        # print(dict_output)
        # M.transformer_output = pyo.Param(M.time, M.model_dims, initialize=dict_output)
        
        # for t in M.time:
        #     if t > 0.9:
        #         # add constraints for next value
        #         for d in M.model_dims:
        #             M.output_constraints.add(expr=input_var[t,d] == M.transformer_output[t, d])
        #             M.output_constraints.add(expr=input_var[t,d] == M.transformer_output[t, d])






