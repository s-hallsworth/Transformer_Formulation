import pyomo.environ as pyo
import numpy as np

class transformer:
    def __init__(self):
        self.transformer_pred = [0,0]
        self.input_array = []
        
    def LAYER_NORM(self):
        """
        Maps an input representation x (dim = N x d) to an input representation x' (dim = N x d)
        """
        # read these params from model
        lambd = 1
        beta = 0
        
        x = np.array(self.input_array)
        if len(x) == 0:
            return np.array([])
    
        epsilon = 1e-10
        x_layer_norm = (lambd * (x - np.mean(x))/(np.std(x) + epsilon) ) + beta

        return x_layer_norm.tolist()
    
    def predict(self):
        x = self.LAYER_NORM()
        print('@@@ ', x)
        
        predictions = [0, 0]
        return predictions
        
    def add_constraints(self, M): 
        M.transformer_constraints = pyo.ConstraintList()
        
        for t in M.time:
            if t == M.time.first() :
                continue
            # create arrays for input x and u values
            if t <= 0.9:
                
                # add constraints that x,u = x,u input values 
                M.transformer_constraints.add(expr = M.x[t] == M.x_in[t])
                M.transformer_constraints.add(expr = M.u[t] == M.u_in[t])
                
                # create arrays with input values
                self.input_array.append(M.x_in[t]) # use x values to predict next values of x,u
                
            if t > 0.9:   
                # predict x, u 
                transformer_output = self.predict()
                
                # update input with predicted value
                self.input_array.append(transformer_output[0])
                
                # add constraints for next value 
                M.transformer_constraints.add(expr = M.x[t] == transformer_output[0])
                M.transformer_constraints.add(expr = M.u[t] == transformer_output[1])
        return
    
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


