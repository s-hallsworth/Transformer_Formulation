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
from OMLT_helper import weights_to_NetworkDefinition, weights_to_NetDef

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

class Transformer:
    def __init__(self, M, config_file):
        
         # get hyper params
        with open(config_file, "r") as file:
            config = json.load(file)

        self.N = config['hyper_params']['N']
        self.d_model = config['hyper_params']['d_model'] # embedding dimensions of model
        self.d_k = config['hyper_params']['d_k']
        self.d_H = config['hyper_params']['d_H']
        self.input_dim = config['hyper_params']['input_dim']
        
        file.close()
        
        #self.W_emb = np.ones((self.input_dim, self.d_model))
        # self.W_k = np.ones((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        # self.W_q = np.ones((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        # self.W_v = np.ones((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        # self.W_o = np.ones((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        
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

    def add_layer_norm(self, M, input_var_name, layer_norm_var_name, gamma, beta):  # non-linear
        """
        Normalization over the sequennce of input
        """
        if not hasattr(M, "layer_norm_constraints"):
            M.layer_norm_constraints = pyo.ConstraintList()
        
        input_var = getattr(M, input_var_name)
        
        # Initialize variables
        if not hasattr(M, layer_norm_var_name):
            # define layer norm output var
            setattr(M, layer_norm_var_name, pyo.Var(M.time_input, M.model_dims))
            layer_norm_var = getattr(M, layer_norm_var_name)
            
            # define calculation variables
            variance_name = 'variance_'+ layer_norm_var_name
            setattr(M, variance_name, pyo.Var(M.time_input))
            variance = getattr(M, variance_name)
            
            div_name = 'div_'+ layer_norm_var_name
            setattr(M, div_name, pyo.Var(M.time_input, M.model_dims))
            div = getattr(M, div_name)
            
            denominator_name = 'denominator_'+ layer_norm_var_name
            setattr(M, denominator_name, pyo.Var(M.time_input))
            denominator = getattr(M, denominator_name)
            
            numerator_name = 'numerator_'+ layer_norm_var_name
            setattr(M, numerator_name, pyo.Var(M.time_input, M.model_dims))
            numerator = getattr(M, numerator_name)

            numerator_scaled_name = 'numerator_scaled_'+ layer_norm_var_name
            setattr(M, numerator_scaled_name, pyo.Var(M.time_input, M.model_dims))
            numerator_scaled = getattr(M, numerator_scaled_name)
            
            numerator_squared_name = 'numerator_squared_'+ layer_norm_var_name
            setattr(M, numerator_squared_name, pyo.Var(M.time_input, M.model_dims))
            numerator_squared = getattr(M, numerator_squared_name)
              
            numerator_squared_sum_name = 'numerator_squared_sum_'+ layer_norm_var_name
            setattr(M, numerator_squared_sum_name, pyo.Var(M.time_input))
            numerator_squared_sum = getattr(M, numerator_squared_sum_name)
            
        else:
            raise ValueError('Attempting to overwrite variable')

        # Add constraints for layer norm
        if self.d_model == 1:
            return
            
        for t in M.time_input: 
            sum_t = sum(input_var[t, d] for d in M.model_dims) 
            mean_t = sum_t/ self.d_model
            
            # Constraints for each element in sequence
            
            for d in M.model_dims:   
                M.layer_norm_constraints.add(expr= numerator[t,d] == input_var[t, d] - mean_t)
                M.layer_norm_constraints.add(expr= numerator_squared[t,d] == numerator[t,d]**2)
                M.layer_norm_constraints.add(expr= numerator_squared_sum[t] == sum(numerator_squared[t,d_prime] for d_prime in M.model_dims))
                M.layer_norm_constraints.add(expr= variance[t] * self.d_model == numerator_squared_sum[t])
                M.layer_norm_constraints.add(expr= denominator[t] **2 == variance[t] )
                
                M.layer_norm_constraints.add(expr= div[t,d] * denominator[t] == numerator[t,d] )
                M.layer_norm_constraints.add(expr= numerator_scaled[t,d] == getattr(M, gamma)[d] * div[t,d])
                
                M.layer_norm_constraints.add(expr=layer_norm_var[t, d] == numerator_scaled[t,d] + getattr(M, beta)[d])
                

    def add_attention(self, M, input_var_name, W_q, W_k, W_v, W_o, b_q = None, b_k = None, b_v = None, b_o = None):
        """
        Multihead attention between each element of embedded sequence
        """
        if not hasattr(M, "attention_constraints"):
            M.attention_constraints = pyo.ConstraintList()
            
        input_var = getattr(M, input_var_name)

        # define sets, vars
        M.heads = pyo.RangeSet(1, self.d_H)
        M.k_dims = pyo.RangeSet(1, self.d_k)

        W_q_dict = {
            (M.model_dims.at(d+1), h, k): W_q[d][h - 1][k - 1]
            for d in range(len(M.model_dims))
            for h in M.heads
            for k in M.k_dims
        }
        W_k_dict = {
            (M.model_dims.at(d+1), h, k): W_k[d][h - 1][k - 1]
            for d in range(len(M.model_dims))
            for h in M.heads
            for k in M.k_dims
        }
        W_v_dict = {
            (M.model_dims.at(d+1), h, k): W_v[d][h - 1][k - 1]
            for d in range(len(M.model_dims))
            for h in M.heads
            for k in M.k_dims
        }
        W_o_dict = {
            (M.model_dims.at(d+1), h, k): W_o[h-1][k - 1][d]
            for d in range(len(M.model_dims))
            for h in M.heads
            for k in M.k_dims
        }
        

        M.W_q = pyo.Param(M.model_dims, M.heads, M.k_dims, initialize=W_q_dict)
        M.W_k = pyo.Param(M.model_dims, M.heads, M.k_dims, initialize=W_k_dict)
        M.W_v = pyo.Param(M.model_dims, M.heads, M.k_dims, initialize=W_v_dict)
        M.W_o = pyo.Param(M.model_dims,M.heads, M.k_dims, initialize=W_o_dict)
        
        if b_q:
            b_q_dict = {
                        (h, k): b_q[h-1][k-1]
                        for h in M.heads
                        for k in M.k_dims
                       }
            M.b_q = pyo.Param(M.heads, M.k_dims, initialize=b_q_dict)
            
        if b_k:
            b_k_dict = {
                        (h, k): b_k[h-1][k-1]
                        for h in M.heads
                        for k in M.k_dims
                       }
            M.b_k = pyo.Param(M.heads, M.k_dims, initialize=b_k_dict)
            
        if b_v: 
            b_v_dict = {
                        (h, k): b_v[h-1][k-1]
                        for h in M.heads
                        for k in M.k_dims
                       }
            M.b_v = pyo.Param(M.heads, M.k_dims, initialize=b_v_dict)
            
        if b_o:
            b_o_dict = {(d): val for d, val in zip(M.model_dims, b_o) }
            M.b_o = pyo.Param(M.model_dims, initialize=b_o_dict)

        M.Q = pyo.Var(M.heads, M.time_input, M.k_dims) 
        M.K = pyo.Var(M.heads, M.time_input, M.k_dims) 
        M.V = pyo.Var(M.heads, M.time_input, M.k_dims) 

        M.compatibility = pyo.Var(M.heads, M.time_input, M.time_input) #, initialize=init_compatibility)  # sqrt(Q * K)
        M.compatibility_exp = pyo.Var(M.heads, M.time_input, M.time_input) #, initialize=init_compatibility_exp)
        M.compatibility_exp_sum = pyo.Var(M.heads, M.time_input) #, initialize=init_compatibility_sum)
        M.compatibility_squ = pyo.Var(M.heads, M.time_input, M.time_input)
        M.compatibility_3 = pyo.Var(M.heads, M.time_input, M.time_input)
        M.compatibility_4 = pyo.Var(M.heads, M.time_input, M.time_input)
        M.compatibility_5 = pyo.Var(M.heads, M.time_input, M.time_input)
        M.compatibility_6 = pyo.Var(M.heads, M.time_input, M.time_input)
        M.compatibility_7 = pyo.Var(M.heads, M.time_input, M.time_input)
        M.compatibility_8 = pyo.Var(M.heads, M.time_input, M.time_input)
        M.compatibility_9 = pyo.Var(M.heads, M.time_input, M.time_input)
        M.compatibility_10 = pyo.Var(M.heads, M.time_input, M.time_input)
        M.compatibility_11 = pyo.Var(M.heads, M.time_input, M.time_input)
          
        M.attention_weight = pyo.Var(M.heads, M.time_input, M.time_input)  # softmax ( (Q * K)/sqrt(d_k) )
        M.attention_score = pyo.Var(
            M.heads, M.time_input, M.k_dims
        )  # softmax ( (Q * K)/sqrt(d_k) ) * V
        M.attention_output = pyo.Var(
            M.time_input, M.model_dims
        )  # concat heads and linear transform

        for h in M.heads:
            for n in M.time_input:
                    for k in M.k_dims:
                        
                        # constraints for Query
                        if b_q:
                            M.attention_constraints.add(
                            expr=M.Q[h, n, k]
                            == sum(input_var[n, d] * M.W_q[d, h, k] for d in M.model_dims) + M.b_q[h,k]
                            )  
                        else: 
                            M.attention_constraints.add(
                                expr=M.Q[h, n, k]
                                == sum(input_var[n, d] * M.W_q[d, h, k] for d in M.model_dims)
                            )
                        
                        # constraints for Key
                        if b_k:
                            M.attention_constraints.add(
                            expr=M.K[h, n, k]
                            == sum(input_var[n, d] * M.W_k[d, h, k] for d in M.model_dims) + M.b_k[h,k]
                        )  
                        else: 
                            M.attention_constraints.add(
                                expr=M.K[h, n, k]
                                == sum(input_var[n, d] * M.W_k[d, h, k] for d in M.model_dims)
                            )
                            
                        # constraints for Value    
                        if b_v:
                            M.attention_constraints.add(
                            expr=M.V[h, n, k]
                            == sum(input_var[n, d] * M.W_v[d, h, k] for d in M.model_dims) + M.b_v[h,k]
                        )  
                        else: 
                            M.attention_constraints.add(
                                expr=M.V[h, n, k]
                                == sum(input_var[n, d] * M.W_v[d, h, k] for d in M.model_dims) 
                            )

                        # attention score = sum(attention_weight * V)
                        M.attention_constraints.add(
                            expr=M.attention_score[h, n, k]
                            == sum(
                                M.attention_weight[h, n, n2] * M.V[h, n2, k]
                                for n2 in M.time_input
                            )
                        )
                        
                    for p in M.time_input:
                        # series approx for EXP
                        M.attention_constraints.add(expr= M.compatibility[h, n, p]**2 == M.compatibility_squ[h, n, p] )#problem for gurobi
                        M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_squ[h, n, p] == M.compatibility_3[h, n, p] )
                        M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_3[h, n, p] == M.compatibility_4[h, n, p] )
                        M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_4[h, n, p] == M.compatibility_5[h, n, p] )
                        M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_5[h, n, p] == M.compatibility_6[h, n, p] )
                        M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_6[h, n, p] == M.compatibility_7[h, n, p] )
                        M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_7[h, n, p] == M.compatibility_8[h, n, p] )
                        M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_8[h, n, p] == M.compatibility_9[h, n, p] )
                        M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_9[h, n, p] == M.compatibility_10[h, n, p] )
                        M.attention_constraints.add(expr= M.compatibility[h, n, p]*M.compatibility_10[h, n, p] == M.compatibility_11[h, n, p] )
                        
                        M.attention_constraints.add(expr= M.compatibility_exp[h, n, p] == 1
                                                    + M.compatibility[h, n, p]
                                                    + (0.5*M.compatibility_squ[h, n, p] ) 
                                                    + (0.166666667*M.compatibility_3[h, n, p]) 
                                                    + (0.0416666667*M.compatibility_4[h, n, p]) 
                                                    + (0.00833333333*M.compatibility_5[h, n, p]) 
                                                    + (0.00138888889*M.compatibility_6[h, n, p]) 
                                                    + (0.000198412698*M.compatibility_7[h, n, p]) 
                                                    + (0.0000248015873*M.compatibility_8[h, n, p]) 
                                                    + (0.00000275573192*M.compatibility_9[h, n, p]) 
                                                    + (0.000000275573192*M.compatibility_10[h, n, p])
                                                    + (0.0000000250521084*M.compatibility_11[h, n, p])
                                                    )# pyo.exp only seems to work for constant args and pow operator must be <= 2

                    for n2 in M.time_input:
                        # compatibility sqrt(Q * K) across all pairs of elements
                        scale = np.sqrt(self.d_k) 
                        M.attention_constraints.add(
                            expr=M.compatibility[h, n, n2] * scale
                            == sum(M.Q[h, n, k] * (M.K[ h, n2, k] )for k in M.k_dims) 
                        )  # non-linear

                        # attention weights softmax(compatibility)
                        M.attention_constraints.add(expr= M.compatibility_exp_sum[h, n] == sum(M.compatibility_exp[h, n, p] for p in M.time_input))
                        M.attention_constraints.add(
                            expr=M.attention_weight[h, n, n2] * M.compatibility_exp_sum[h, n]
                            == M.compatibility_exp[h, n, n2]) 

        # multihead attention output constraint
        for n in M.time_input:
            for d in M.model_dims:
                M.attention_constraints.add(
                    expr=M.attention_output[n, d]
                    == sum(
                        (sum(
                            M.attention_score[h, n, k] * M.W_o[d,h, k]
                            for k in M.k_dims
                        ) )
                        for h in M.heads
                        
                    )+ M.b_o[d]
                )
                
    def add_residual_connection(self,M, input_1, input_2, output_var_name):
        # create constraint list
        if not hasattr(M, "residual_constraints"):
            M.residual_constraints = pyo.ConstraintList()
        
        # add new variable
        if not hasattr(M, output_var_name):
            setattr(M, output_var_name, pyo.Var(M.time_input, M.model_dims))
            residual_var = getattr(M, output_var_name)
        else:
            raise ValueError('Attempting to overwrite variable')
        
        for d in M.model_dims:
            for t in M.time_input:
                M.residual_constraints.add(expr= residual_var[t,d] == input_1[t,d] + input_2[t,d])

    
    def add_FFN_2D(self,M, input_var_name, output_var_name, input_value, model_parameters):
        input_var = getattr(M, input_var_name)
        
        # create constraint list
        constraint_name = "constraints_"+input_var_name
        if not hasattr(M, constraint_name):
            setattr(M, constraint_name, pyo.ConstraintList())
            constraint_list = getattr(M, constraint_name)
            #M.residual_constraints = pyo.ConstraintList()
        
        # add new variable
        if not hasattr(M, output_var_name):
            setattr(M, output_var_name, pyo.Var(M.time_input, M.model_dims))
            output_var = getattr(M, output_var_name)
            
            NN_name = output_var_name
            setattr(M, NN_name, OmltBlock())
            NN_block = getattr(M, NN_name)
        
        else:
            raise ValueError('Attempting to overwrite variable')
        
        input_bounds={0: (-10,10), 1: (-10,10)}
        net_relu = weights_to_NetDef(output_var_name, input_value, model_parameters, input_bounds)
        #net_relu = weights_to_NetworkDefinition(output_var_name, model_parameters)
  
        formulation_bigm = ReluBigMFormulation(net_relu)
        NN_block.build_formulation(formulation_bigm)
        
        for t in M.time_input: 
            for j,d in enumerate(M.model_dims): 
                constraint_list.add(expr= input_var[t,d] == NN_block.inputs[j])
                constraint_list.add(expr= output_var[t,d] == NN_block.outputs[j])
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






# transformer_pred = [0,0]
# def _x_transformer(M, t):
#     if t == M.time.first() :
#         return pyo.Constraint.Skip
#     if t <= 0.9:
#         return M.x[t] == M.x_in[t]

#     return M.x[t] == transformer_pred[0]

# def _u_transformer(M, t):
#     if t == M.time.first():
#         return pyo.Constraint.Skip
#     if t > 0.9
#         return M.u[t] == M.u_in[t]
#     return M.u[t] == transformer_pred[1]