import pyomo.environ as pyo
import numpy as np


class transformer:
    def __init__(self, M):
        self.transformer_pred = [0, 0]
        self.input_array = []
        self.epsilon = 1e-10

        # get hyper params
        self.N = 4
        self.d_model = 3
        self.d_k = 1
        self.d_H = 1
        self.input_dim = 2

        # get learned params
        self.lambd = 1
        self.beta = 0
        self.W_emb = np.ones((self.input_dim, self.d_model))
        self.W_k = np.ones((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        self.W_q = np.zeros((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        self.W_v = np.ones((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        self.W_o = np.ones((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        
        # initialise set of model dims
        if self.d_model > 1:
            str_array = ["{}".format(x) for x in range(0, self.d_model)]
            M.model_dims = pyo.Set(initialize=str_array)
        else:
            M.model_dims = pyo.Set(initialize=[str(0)])

    def embed_input(self, M, embed_var_name, input_var, set_var):
        """
        Embed the feature dimensions of input
        """
        if not hasattr(M, "embed_constraints"):
            M.embed_constraints = pyo.ConstraintList()
        
        # define embedding var
        if not hasattr(M, embed_var_name):
            setattr(M, embed_var_name, pyo.Var(M.time, M.model_dims))
            embed_var = getattr(M, embed_var_name)
        else:
            raise ValueError('Attempting to overwrite variable: ', embed_var_name)
  
        
        #if model dims = number of variables in input var
        if self.d_model == len(set_var):
            for s,d in [set_var, M.model_dims]:
                for t in M.time:
                    M.embed_constraints.add(embed_var[t, s] == input_var[t,s])
        else:
            W_emb_dict = {
                (set_var.at(s+1),M.model_dims.at(m+1)): self.W_emb[s][m]
                for s in range(len(set_var))
                for m in range(len(M.model_dims))
            }
            M.W_emb = pyo.Param(set_var, M.model_dims, initialize=W_emb_dict)
            
            for m in M.model_dims:
                for t in M.time:
                    M.embed_constraints.add(embed_var[t, m] 
                                            == sum(input_var[t,s] * M.W_emb[s,m] for s in set_var)
                                            )

    def add_layer_norm(self, M, input_var, layer_norm_var_name):  # non-linear
        """
        Normalization over the feature dimensions of input
        """
        if not hasattr(M, "layer_norm_constraints"):
            M.layer_norm_constraints = pyo.ConstraintList()

        # Initialize variables
        M.x_sum = pyo.Var(M.time, initialize=0)
        
        if not hasattr(M, layer_norm_var_name):
            setattr(M, layer_norm_var_name, pyo.Var(M.time, M.model_dims, initialize=0))
            layer_norm_var = getattr(M, layer_norm_var_name)
        else:
            raise ValueError('Attempting to overwrite variable: ', layer_norm_var)


        # Add constraints for layer norm
        if self.d_model == 1:
            return

        for t in M.time:
            # Constraint for summing input_var over model_dims
            M.layer_norm_constraints.add(
                expr=M.x_sum[t] == sum(input_var[t, d] for d in M.model_dims)
            )

            # Constraints for each dimension
            for d in M.model_dims:
                mean_d = M.x_sum[t] / self.d_model  # Mean
                variance = (
                    sum((input_var[t, d_prime] - mean_d) ** 2 for d_prime in M.model_dims)
                    / self.d_model
                )
                std_dev = (variance + self.epsilon) ** 0.5  # epsilon to avoid div 0

                # Calculate layer normalization
                x_mean = input_var[t, d] - mean_d
                layer_norm = (self.lambd * (x_mean / std_dev)) - self.beta

                # Add constraint for layer normalized output
                M.layer_norm_constraints.add(expr=layer_norm_var[t, d] == layer_norm)

    def add_attention(self, M, input_var):
        """
        Multihead attention between each element of embedded sequence
        """
        if not hasattr(M, "attention_constraints"):
            M.attention_constraints = pyo.ConstraintList()


        # define sets, vars
        M.heads = pyo.RangeSet(1, self.d_H)
        M.k_dims = pyo.RangeSet(1, self.d_k)

        W_q_dict = {
            (h, M.model_dims.at(m+1), k): self.W_q[h - 1][m][k - 1]
            for h in M.heads
            for m in range(len(M.model_dims))
            for k in M.k_dims
        }
        W_k_dict = {
            (h, M.model_dims.at(m+1), k): self.W_k[h - 1][m][k - 1]
            for h in M.heads
            for m in range(len(M.model_dims))
            for k in M.k_dims
        }
        W_v_dict = {
            (h, M.model_dims.at(m+1), k): self.W_v[h - 1][m][k - 1]
            for h in M.heads
            for m in range(len(M.model_dims))
            for k in M.k_dims
        }
        W_o_dict = {
            (h, M.model_dims.at(m+1), k): self.W_o[h - 1][m][k - 1]
            for h in M.heads
            for m in range(len(M.model_dims))
            for k in M.k_dims
        }

        M.W_q = pyo.Param(M.heads, M.model_dims, M.k_dims, initialize=W_q_dict)
        M.W_k = pyo.Param(M.heads, M.model_dims, M.k_dims, initialize=W_k_dict)
        M.W_v = pyo.Param(M.heads, M.model_dims, M.k_dims, initialize=W_v_dict)
        M.W_o = pyo.Param(M.heads, M.model_dims, M.k_dims, initialize=W_o_dict)

        M.Q = pyo.Var(M.heads, M.time, M.k_dims)
        M.K = pyo.Var(M.heads, M.time, M.k_dims)
        M.V = pyo.Var(M.heads, M.time, M.k_dims)

        M.compatability = pyo.Var(M.heads, M.time, M.time)  # sqrt(Q * K)
        M.attention_weight = pyo.Var(M.heads, M.time, M.time)  # softmax ( sqrt(Q * K) )
        M.attention_score = pyo.Var(
            M.heads, M.time, M.k_dims
        )  # softmax ( sqrt(Q * K) ) * V
        M.attention_output = pyo.Var(
            M.time, M.model_dims
        )  # concat heads and linear transform

        for h in M.heads:
            for n in M.time:
                for k in M.k_dims:
                    # constraints for Query, Key and Value
                    M.attention_constraints.add(
                        expr=M.Q[h, n, k]
                        == sum(input_var[n, m] * M.W_q[h, m, k] for m in M.model_dims)
                    )
                    M.attention_constraints.add(
                        expr=M.K[h, n, k]
                        == sum(input_var[n, m] * M.W_k[h, m, k] for m in M.model_dims)
                    )
                    M.attention_constraints.add(
                        expr=M.V[h, n, k]
                        == sum(input_var[n, m] * M.W_v[h, m, k] for m in M.model_dims)
                    )

                    # attention score = sum(attention_weight * V)
                    M.attention_constraints.add(
                        expr=M.attention_score[h, n, k]
                        == sum(
                            M.attention_weight[h, n, n2] * M.V[h, n2, k]
                            for n2 in M.time
                        )
                    )

                for n2 in M.time:
                    # compatibility sqrt(Q * K) across all pairs of elements
                    M.attention_constraints.add(
                        expr=M.compatability[h, n, n2]
                        == sum(M.Q[h, n, k] * M.K[h, n, k] for k in M.k_dims)
                        ** (1 / self.d_k)
                    )  # non-linear

                    # attention weights softmax(compatibility)
                    M.attention_constraints.add(
                        expr=M.attention_weight[h, n, n2]
                        == pyo.exp(M.compatability[h, n, n2])
                        / sum(pyo.exp(M.compatability[h, n, p]) for p in M.time)
                    )

        # multihead attention output constraint
        for n in M.time:
            for m in M.model_dims:
                M.attention_constraints.add(
                    expr=M.attention_output[n, m]
                    == sum(
                        sum(
                            M.attention_score[h, n, k] * M.W_o[h, m, k]
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
            setattr(M, output_var_name, pyo.Var(M.time, M.model_dims))
            residual_var = getattr(M, output_var_name)
        else:
            raise ValueError('Attempting to overwrite variable ', output_var_name)
        
        for d in M.model_dims:
            for t in M.time:
                M.residual_constraints.add(expr= residual_var[t,d] == input_1[t,d] + input_2[t,d])


    def add_output_constraints(self, M, input_var):
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

#         return M.u[t] == M.u_in[t]
#     return M.u[t] == transformer_pred[1]
