import pyomo.environ as pyo
import numpy as np
from pyomo import dae
import json
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off

class Transformer:
    def __init__(self, M, config_file):
        
         # get hyper params
        with open(config_file, "r") as file:
            config = json.load(file)

        self.N = config['hyper_params']['N']
        self.d_model = config['hyper_params']['d_model']
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
            init_array = 0.5 * np.ones((self.N, self.d_model)) #initialize embed array to 0.5
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
        else:
            W_emb_dict = {
                (set_var.at(s+1),M.model_dims.at(m+1)): W_emb[s][m]
                for s in range(len(set_var))
                for m in range(len(M.model_dims))
            }
            M.W_emb = pyo.Param(set_var, M.model_dims, initialize=W_emb_dict)
            
            for m in M.model_dims:
                for t in M.time_input:
                    M.embed_constraints.add(embed_var[t, m] 
                                            == sum(input_var[t,s] * M.W_emb[s,m] for s in set_var)
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
            setattr(M, layer_norm_var_name, pyo.Var(M.time_input, M.model_dims))
            layer_norm_var = getattr(M, layer_norm_var_name)
            
        else:
            raise ValueError('Attempting to overwrite variable')

        # Add constraints for layer norm
        if self.d_model == 1:
            return

        M.variance = pyo.Var(M.model_dims, within=pyo.Reals)
        M.div = pyo.Var(M.time_input, M.model_dims)
        M.denominator = pyo.Var( M.model_dims)
        M.numerator = pyo.Var(M.time_input, M.model_dims)  
        M.numerator_scaled = pyo.Var(M.time_input, M.model_dims)  
        M.numerator_squared = pyo.Var(M.time_input, M.model_dims) 
        M.numerator_squared_sum = pyo.Var(M.model_dims) 
            
        for d in M.model_dims:
            sum_t = sum(input_var[t, d] for t in M.time_input) 
            mean_t = sum_t/ self.N
            
            # Constraints for each element in sequence
            for t in M.time_input: 
                
                M.layer_norm_constraints.add(expr= M.numerator[t,d] == input_var[t, d] - mean_t)
                M.layer_norm_constraints.add(expr= M.numerator_squared[t,d] == M.numerator[t,d]**2)
                M.layer_norm_constraints.add(expr= M.numerator_squared_sum[d] == sum(M.numerator_squared[t_prime,d] for t_prime in M.time_input))
                M.layer_norm_constraints.add(expr= M.variance[d] == M.numerator_squared_sum[d] / self.N)
                M.layer_norm_constraints.add(expr= M.denominator[d] **2 == M.variance[d] )
                M.layer_norm_constraints.add(expr= M.numerator_scaled[t,d] == getattr(M, gamma)[t] * M.numerator[t,d])
                
                M.layer_norm_constraints.add(expr= M.div[t,d] * M.denominator[d] == M.numerator_scaled[t,d] )
                
                M.layer_norm_constraints.add(expr=layer_norm_var[t, d] == M.div[t,d] + getattr(M, beta)[t])
                
                # var_minus = input_var
                # var_squ = var_minus
                # for t_prime in M.time_input:
                #     var_minus[t_prime, d] -= mean_t 
                #     var_squ[t_prime, d] = var_minus[t_prime, d] ** 2
                    
                # var_sum_squ = sum(var_squ[t_prime, d] for t_prime in M.time_input)
                # variance = var_sum_squ / self.N 
                #variance_ep = variance + self.epsilon # epsilon to avoid div 0
                #std_dev =  variance_ep ** 0.5 
                
                # LP format
                
                # M.layer_norm_constraints.add(expr= M.variance[d] == variance + self.epsilon )
                # M.layer_norm_constraints.add(expr= M.variance[d] == M.std_dev[d] **2)
                # M.layer_norm_constraints.add(expr= M.denominator[t,d] == layer_norm_var[t, d] - getattr(M, beta)[t])
                # M.layer_norm_constraints.add(expr= M.numerator[t,d] == input_var[t, d] - mean_t)
                # M.layer_norm_constraints.add(expr= M.scaled_numerator[t,d] == getattr(M, gamma)[t] * M.numerator[t,d])
                # M.layer_norm_constraints.add(expr= M.std_dev[d] == M.scaled_numerator[t,d] / M.denominator[t,d])
                
                
                # # Add constraint for layer normalized output
                # numerator = input_var[t, d] - mean_t
                # frac = numerator / M.std_dev[d]
                # scaled_frac = getattr(M, gamma)[t] * frac
                # layer_norm = scaled_frac - getattr(M, beta)[t]
                # M.layer_norm_constraints.add(expr=layer_norm_var[t, d] == layer_norm)

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
            (M.time_input.at(t+1), h, k): W_q[t][h - 1][k - 1]
            for t in range(len(M.time_input))
            for h in M.heads
            for k in M.k_dims
        }
        W_k_dict = {
            (M.time_input.at(t+1), h, k): W_k[t][h - 1][k - 1]
            for t in range(len(M.time_input))
            for h in M.heads
            for k in M.k_dims
        }
        W_v_dict = {
            (M.time_input.at(t+1), h, k): W_v[t][h - 1][k - 1]
            for t in range(len(M.time_input))
            for h in M.heads
            for k in M.k_dims
        }
        W_o_dict = {
            (h, M.time_input.at(t+1), k): W_o[h-1][t][k - 1]
            for t in range(len(M.time_input))
            for h in M.heads
            for k in M.k_dims
        }
        

        M.W_q = pyo.Param(M.time_input, M.heads, M.k_dims, initialize=W_q_dict)
        M.W_k = pyo.Param(M.time_input, M.heads, M.k_dims, initialize=W_k_dict)
        M.W_v = pyo.Param(M.time_input, M.heads, M.k_dims, initialize=W_v_dict)
        M.W_o = pyo.Param(M.heads, M.time_input, M.k_dims, initialize=W_o_dict)
        
        
        b_q_dict = {
                        (h, k): b_q[h-1][k-1]
                        for h in M.heads
                        for k in M.k_dims
                    }
        b_k_dict = {
                        (h, k): b_k[h-1][k-1]
                        for h in M.heads
                        for k in M.k_dims
                    }
        b_v_dict = {
                        (h, k): b_v[h-1][k-1]
                        for h in M.heads
                        for k in M.k_dims
                    }

        b_o_dict = {(k): val for k, val in zip(M.k_dims, b_o) }

        M.b_q = pyo.Param(M.heads, M.k_dims, initialize=b_q_dict)
        M.b_k = pyo.Param(M.heads, M.k_dims, initialize=b_k_dict)
        M.b_v = pyo.Param(M.heads, M.k_dims, initialize=b_v_dict)
        M.b_o = pyo.Param(M.k_dims, initialize=b_o_dict)

        M.Q = pyo.Var(M.heads, M.time_input, M.k_dims, M.model_dims)
        M.K = pyo.Var(M.heads, M.time_input, M.k_dims, M.model_dims)
        M.V = pyo.Var(M.heads, M.time_input, M.k_dims, M.model_dims)

        M.compatability = pyo.Var(M.heads, M.time_input, M.time_input, M.model_dims)  # sqrt(Q * K)
        M.attention_weight = pyo.Var(M.heads, M.time_input, M.time_input, M.model_dims, bounds=(0,1))  # softmax ( sqrt(Q * K) )
        M.attention_score = pyo.Var(
            M.heads, M.time_input, M.k_dims , M.model_dims
        )  # softmax ( sqrt(Q * K) ) * V
        M.attention_output = pyo.Var(
            M.time_input, M.model_dims
        )  # concat heads and linear transform

        for h in M.heads:
            for n in M.time_input:
                for m in M.model_dims:
                    for k in M.k_dims:
                        
                        # M.embed_constraints.add(embed_var[t, m] 
                        #                     == sum(input_var[t,s] * M.W_emb[s,m] for s in set_var)
                        #                     )
                        
                        # constraints for Query, Key and Value
                        if b_q:
                            M.attention_constraints.add(
                            expr=M.Q[h, n, k, m]
                            == (input_var[n, m] * M.W_q[n, h, k]) + M.b_q[h,k]
                        )  
                        else: 
                            M.attention_constraints.add(
                                expr=M.Q[h, n, k, m]
                                == input_var[n, m] * M.W_q[n, h, k]
                            )
                            
                        if b_k:
                            M.attention_constraints.add(
                            expr=M.K[h, n, k, m]
                            == (input_var[n, m] * M.W_k[n, h, k]) + M.b_k[h,k]
                        )  
                        else: 
                            M.attention_constraints.add(
                                expr=M.K[h, n, k, m]
                                == input_var[n, m] * M.W_k[n, h, k]
                            )
                            
                            
                        if b_v:
                            M.attention_constraints.add(
                            expr=M.V[h, n, k, m]
                            == (input_var[n, m] * M.W_v[n, h, k]) + M.b_v[h,k]
                        )  
                        else: 
                            M.attention_constraints.add(
                                expr=M.V[h, n, k, m]
                                == input_var[n, m] * M.W_v[n, h, k]
                            )

                        # attention score = sum(attention_weight * V)
                        M.attention_constraints.add(
                            expr=M.attention_score[h, n, k, m]
                            == sum(
                                M.attention_weight[h, n, n2, m] * M.V[h, n2, k, m]
                                for n2 in M.time_input
                            )
                        )

                    for n2 in M.time_input:
                        # compatibility sqrt(Q * K) across all pairs of elements
                        K_scaled = M.K[ h, n2, k, m]  / (self.d_k ** 0.5) 
                        M.attention_constraints.add(
                            expr=M.compatability[h, n, n2, m]
                            == sum(M.Q[h, n, k, m] * K_scaled for k in M.k_dims)
                        )  # non-linear

                        # attention weights softmax(compatibility)
                        M.attention_constraints.add(
                            expr=M.attention_weight[h, n, n2, m]
                            == pyo.exp(M.compatability[h, n, n2, m])
                            / sum(pyo.exp(M.compatability[h, n, p, m]) for p in M.time_input)
                        )

        # multihead attention output constraint
        for n in M.time_input:
            for m in M.model_dims:
                M.attention_constraints.add(
                    expr=M.attention_output[n, m]
                    == sum(
                        sum(
                            M.attention_score[h, n, k, m] * M.W_o[h, n, k]
                            for k in M.k_dims
                        )
                        for h in M.heads
                    )
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
        
        for m in M.model_dims:
            for t in M.time_input:
                M.residual_constraints.add(expr= residual_var[t,m] == input_1[t,m] + input_2[t,m])


    #def add_output_constraints(self, M, input_var):
        # if not hasattr(M, "output_constraints"):
        #     M.output_constraints = pyo.ConstraintList()

        # # predict x, u
        # output = np.ones((len(M.time), self.d_model))
        # dict_output = {(t, str(m)): output[i, m] for i, t in enumerate(M.time) for m in range(len(M.model_dims))}
        # print(dict_output)
        # M.transformer_output = pyo.Param(M.time, M.model_dims, initialize=dict_output)
        
        # for t in M.time:
        #     if t > 0.9:
        #         # add constraints for next value
        #         for m in M.model_dims:
        #             M.output_constraints.add(expr=input_var[t,m] == M.transformer_output[t, m])
        #             M.output_constraints.add(expr=input_var[t,m] == M.transformer_output[t, m])






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