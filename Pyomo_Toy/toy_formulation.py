import pyomo.environ as pyo
import numpy as np


class transformer:
    def __init__(self, M):
        self.transformer_pred = [0, 0]
        self.input_array = []
        self.epsilon = 1e-10

        # get hyper params
        self.N = 4
        self.d_model = 1
        self.d_k = 1
        self.d_H = 1

        # get learned params
        self.lambd = 1
        self.beta = 0
        self.W_k = np.ones((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        self.W_q = np.zeros((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        self.W_v = np.ones((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k
        self.W_o = np.ones((self.d_H, self.d_model, self.d_k))  # H x d_model x d_k

        # initialise set of model dims
        if self.d_model > 1:
            M.model_dims = pyo.Set(initialize=range(self.d_model))
        else:
            M.model_dims = pyo.Set(initialize=[1])

    def embed_input(self, M):
        """
        Embed the feature dimensions of input
        """
        if not hasattr(M, "embed_constraints"):
            M.embed_constraints = pyo.ConstraintList()

        M.x_embed = pyo.Var(M.time, M.model_dims)
        if self.d_model == 1:
            for t in M.time:
                M.embed_constraints.add(M.x_embed[t, 1] == M.x[t])
        # else:

    def add_layer_norm(self, M, var, layer_norm_var_name):  # non-linear
        """
        Normalization over the feature dimensions of input
        """
        if not hasattr(M, "layer_norm_constraints"):
            M.layer_norm_constraints = pyo.ConstraintList()

        # Initialize variables
        M.x_sum = pyo.Var(M.time, initialize=0)

        setattr(M, layer_norm_var_name, pyo.Var(M.time, M.model_dims, initialize=0))
        layer_norm_var = getattr(M, layer_norm_var_name)

        # Add constraints for layer norm
        if self.d_model == 1:
            return

        for t in M.time:
            # Constraint for summing var over model_dims
            M.layer_norm_constraints.add(
                expr=M.x_sum[t] == sum(M.var[t, d] for d in M.model_dims)
            )

            # Constraints for each dimension
            for d in M.model_dims:
                mean_d = M.x_sum[t] / self.d_model  # Mean
                variance = (
                    sum((M.var[t, d_prime] - mean_d) ** 2 for d_prime in M.model_dims)
                    / self.d_model
                )
                std_dev = (variance + self.epsilon) ** 0.5  # epsilon to avoid div 0

                # Calculate layer normalization
                x_mean = M.var[t, d] - mean_d
                layer_norm = (self.lambd * (x_mean / std_dev)) - self.beta

                # Add constraint for layer normalized output
                M.layer_norm_constraints.add(expr=layer_norm_var[t, d] == layer_norm)

    def add_attention(self, M):
        """
        Multihead attention between each element of embedded sequence
        """
        if not hasattr(M, "attention_constraints"):
            M.attention_constraints = pyo.ConstraintList()

        # define sets, vars
        M.heads = pyo.RangeSet(1, self.d_H)
        M.k_dims = pyo.RangeSet(1, self.d_k)

        W_q_dict = {
            (h, m, k): self.W_q[h - 1][m - 1][k - 1]
            for h in M.heads
            for m in M.model_dims
            for k in M.k_dims
        }
        W_k_dict = {
            (h, m, k): self.W_k[h - 1][m - 1][k - 1]
            for h in M.heads
            for m in M.model_dims
            for k in M.k_dims
        }
        W_v_dict = {
            (h, m, k): self.W_v[h - 1][m - 1][k - 1]
            for h in M.heads
            for m in M.model_dims
            for k in M.k_dims
        }
        W_o_dict = {
            (h, m, k): self.W_o[h - 1][m - 1][k - 1]
            for h in M.heads
            for m in M.model_dims
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
                        == sum(M.x_embed[n, m] * M.W_q[h, m, k] for m in M.model_dims)
                    )
                    M.attention_constraints.add(
                        expr=M.K[h, n, k]
                        == sum(M.x_embed[n, m] * M.W_k[h, m, k] for m in M.model_dims)
                    )
                    M.attention_constraints.add(
                        expr=M.V[h, n, k]
                        == sum(M.x_embed[n, m] * M.W_v[h, m, k] for m in M.model_dims)
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

        # output constraint
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

    def add_input_constraints(self, M):
        if not hasattr(M, "input_constraints"):
            M.input_constraints = pyo.ConstraintList()

        for t in M.time:
            if t == M.time.first():
                M.input_constraints.add(expr=M.u[t] == M.u_in[t])
                continue
            # create arrays for input x and u values
            if t <= 0.9:
                # add constraints that x,u = x,u input values
                M.input_constraints.add(expr=M.x[t] == M.x_in[t])
                M.input_constraints.add(expr=M.u[t] == M.u_in[t])

                # create arrays with input values
                self.input_array.append(
                    M.x_in[t]
                )  # use x values to predict next values of x,u

    def add_output_constraints(self, M):
        if not hasattr(M, "output_constraints"):
            M.output_constraints = pyo.ConstraintList()

        for t in M.time:
            if t > 0.9:
                # predict x, u
                transformer_output = [0, 0]

                # update input with predicted value
                self.input_array.append(transformer_output[0])

                # add constraints for next value
                M.output_constraints.add(expr=M.x[t] == transformer_output[0])
                M.output_constraints.add(expr=M.u[t] == transformer_output[1])


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
