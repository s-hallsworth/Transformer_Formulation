import pyomo.environ as pyo

transformer_pred = [0,0]

def LAYER_NORM(x):
    return 

##### GET RESULT ######
def _x_transformer(M, t):
    if t == M.time.first() :
        return pyo.Constraint.Skip
    if t <= 0.9:
        return M.x[t] == M.x_in[t]
    return M.x[t] == transformer_pred[0]

def _u_transformer(M, t):
    if t == M.time.first():
        return pyo.Constraint.Skip
    if t <=0.9:
        return M.u[t] == M.u_in[t]
    return M.u[t] == transformer_pred[1]


