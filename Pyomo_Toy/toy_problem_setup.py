import pyomo.environ as pyo
from pyomo import dae
import numpy as np
import extract_from_pretrained as extract_from_pretrained
import data_gen

"""
Define toy problem parametrs and var then run from another script like toy_problem.py or transformer_test.py
"""

NOT_WARM = False
## read model weights
#model_path = "..\\Transformer_Toy\\transformer_small_relu_2_seqlen_2.keras"
model_path = "..\\Transformer_Toy\\transformer_small_relu_2_TOY.keras" 
layer_names, parameters, TNN_model = extract_from_pretrained.get_learned_parameters(model_path)

## create model
model = pyo.ConcreteModel(name="(TOY_TEST)")

## define problem sets, vars, params
T = 11 #11
time = np.linspace(0, 1, num=T) # entire time t=0:1 including prediction times
model.time_input = dae.ContinuousSet(initialize=time[:-1]) # t < prediction times
model.time = dae.ContinuousSet(initialize=time)
set_variables = ['0','1'] ##--- NB: same order as trained input ---##
model.variables = pyo.Set(initialize=set_variables)

# Set bounds based on training dataset
UB_input = 3.5
LB_input = 0 

# Define inputs
if NOT_WARM:
    ## USE last 10 of 9000 data points
    # x_input = data_gen.x[0, -10:]
    # u_input = data_gen.u[0, -10:]
    
    x_input = data_gen.x[0, -3:]
    u_input = data_gen.u[0, -3:]
    
    model.input_param = pyo.Var(model.time_input, model.variables, bounds=(LB_input, UB_input))
    model.input_var = pyo.Var(model.time, model.variables, bounds=(LB_input, UB_input)) #t = 0 to t=1
    
else:
    x_input = [1.0, 1.10657895, 1.21388889, 1.32205882, 1.43125, 1.54166667, 1.65357143, 1.76730769, 1.88333333, 2.00227273] #, 2.125]
    u_input = [0.25, 0.26315789, 0.27777778, 0.29411765, 0.3125, 0.33333333, 0.35714286, 0.38461538, 0.41666667, 0.45454545] #, 0.5]
    
    # x_input = data_gen.x[0, -3:-2]
    # u_input = data_gen.u[0, -3:-2]
    
    dict_inputs = {}
    for t, (u_val, x_val) in zip(model.time_input, zip(u_input, x_input)):
        dict_inputs[(t, '0')] = x_val
        dict_inputs[(t, '1')] = u_val
        
    # model.input_param = pyo.Param(model.time_input, model.variables, initialize=dict_inputs)
    model.input_param = pyo.Var(model.time_input, model.variables, bounds=(LB_input, UB_input)) # t=0 to t=prediction time
    
    model.input_param_fixed = pyo.Param(model.time_input, model.variables, initialize=dict_inputs) # t=0 to t=prediction time
    model.x_param_fix_constraints = pyo.ConstraintList()  
    for t in model.time_input:
        for d in model.variables:
            if t < 1:
                model.x_param_fix_constraints.add(expr=model.input_param[t,d] == model.input_param_fixed[t,d])

    X =  [1.0, 1.10657895, 1.21388889, 1.32205882, 1.43125, 1.54166667, 1.65357143, 1.76730769, 1.88333333, 2.00227273, 1.4942082707791076]
    U =  [0.25, 0.26315789, 0.27777778, 0.29411765, 0.3125, 0.33333333, 0.35714286, 0.38461538, 0.41666667, 0.45454545, 0.33872000390697854]

    #X = data_gen.x[0, -3:]
    # U = data_gen.u[0, -3:]
    
    dict_outputs = {}
    for t, (u_out, x_out) in zip(model.time, zip(U, X)):
        dict_outputs[(t, '0')] = x_out
        dict_outputs[(t, '1')] = u_out
    model.input_var = pyo.Var(model.time, model.variables, initialize=dict_outputs, bounds=(LB_input, UB_input)) #t = 0 to t=1


## define transformer sets, vars, params
dict_gamma1 = {(v): val for v,val in zip(model.variables, parameters['layer_normalization_1','gamma'])}
dict_beta1 = {(v): val for v,val in zip(model.variables,  parameters['layer_normalization_1','beta'])}
model.gamma1 = pyo.Param(model.variables, initialize = dict_gamma1)
model.beta1 = pyo.Param(model.variables, initialize = dict_beta1)


dict_gamma2 = {(v): val for v,val in zip(model.variables, parameters['layer_normalization_2','gamma'])}
dict_beta2 = {(v): val for v,val in zip(model.variables,  parameters['layer_normalization_2','beta'])}
model.gamma2 = pyo.Param(model.variables, initialize = dict_gamma2)
model.beta2 = pyo.Param(model.variables, initialize = dict_beta2)

print("GB 1: ", parameters['layer_normalization_1','gamma'], parameters['layer_normalization_1','beta'])
print("GB 2: ", parameters['layer_normalization_2','gamma'], parameters['layer_normalization_2','beta'])

W_q = parameters['multi_head_attention_1','W_q']
W_k = parameters['multi_head_attention_1','W_k']
W_v = parameters['multi_head_attention_1','W_v']
W_o = parameters['multi_head_attention_1','W_o']

b_q = parameters['multi_head_attention_1','b_q']
b_k = parameters['multi_head_attention_1','b_k']
b_v = parameters['multi_head_attention_1','b_v']
b_o = parameters['multi_head_attention_1','b_o']


""" REMOVE FOR TESTING OPTIMIZATION WITH WINDOW OF LAST 10  POINTS"""
## define constraints
model.input_constraints = pyo.ConstraintList()      
if NOT_WARM:
    
    for t_index, t in enumerate(model.time):
        if t == model.time.first():
            model.x_init_constr_x = pyo.Constraint(expr=model.input_param[t,'0'] == x_input[0])
            model.x_init_constr_u = pyo.Constraint(expr=model.input_param[t,'1'] == u_input[0])
            
        elif t < model.time.last():
            # add constraints that x,u = x,u input values
            model.input_constraints.add(expr=model.input_var[t,'0'] == model.input_param[t,'0'])
            model.input_constraints.add(expr=model.input_var[t,'1'] == model.input_param[t,'1'])
            
            # increasing var with time
            model.input_constraints.add(expr=model.input_var[t,'0'] >= model.input_var[model.time.at(t_index ),'0'])
            model.input_constraints.add(expr=model.input_var[t,'1'] >= model.input_var[model.time.at(t_index ),'1']) 
              
else:  
    
    
    for t_index, t in enumerate(model.time):
        if t == model.time.first():
            model.x_init_constr = pyo.Constraint(expr=model.input_var[t,'0'] == 1)
            model.input_constraints.add(expr=model.input_var[t,'1'] == model.input_param[t,'1'])
            
        # create arrays for input x and u values
        elif t < model.time.last():
            # add constraints that x,u = x,u input values
            model.input_constraints.add(expr=model.input_var[t,'0'] == model.input_param[t,'0'])
            model.input_constraints.add(expr=model.input_var[t,'1'] == model.input_param[t,'1'])  
            
            # increasing var with time
            model.input_constraints.add(expr=model.input_var[t,'0'] >= model.input_var[model.time.at(t_index ),'0'])
            model.input_constraints.add(expr=model.input_var[t,'1'] >= model.input_var[model.time.at(t_index ),'1'])    
    
# define integral constraints
def _intX(m, t):
    return m.input_var[t,'0']
def _intU(m, t):
    return m.input_var[t,'1']
model.intX = dae.Integral(model.time, wrt=model.time, rule=_intX)
model.intU = dae.Integral(model.time, wrt=model.time, rule=_intU)

# bound integral of x and u
model.intX.ub = model.input_var[model.time.last(),'0'].ub * model.time.last()
model.intX.lb = model.input_var[model.time.first(),'0'].lb * model.time.last()

model.intU.ub = model.input_var[model.time.last(),'1'].ub * model.time.last()
model.intU.lb = model.input_var[model.time.first(),'1'].lb * model.time.last()

# Set objective function
model.obj = pyo.Objective(
    expr=model.intX - model.intU + model.input_var[model.time.last(),'0'], sense=1
)  # -1: maximize, +1: minimize (default)

