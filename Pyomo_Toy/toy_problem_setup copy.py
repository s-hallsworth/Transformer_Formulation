import pyomo.environ as pyo
from pyomo import dae
import numpy as np
import extract_from_pretrained as extract_from_pretrained
import data_gen

"""
Define toy problem parametrs and var then run from another script like toy_problem.py or transformer_test.py
"""

INIT_ONLY = False
## read model weights
model_path = "..\\Transformer_Toy\\transformer_small_relu_2_TOY.keras" 
layer_names, parameters, TNN_model = extract_from_pretrained.get_learned_parameters(model_path)

## create model
model = pyo.ConcreteModel(name="(TOY_TEST)")

## define problem sets, vars, params
T = 11
time = np.linspace(0, 1, num=T) # entire time t=0:1 including prediction times
model.time_input = dae.ContinuousSet(initialize=time[:-1]) # t < prediction times
model.time = dae.ContinuousSet(initialize=time)


if INIT_ONLY:
    ## USE last 10 of 9000 data points
    x_input = data_gen.x[0, -10:]
    u_input = data_gen.u[ 0, -10:]
else:
    x_input = [1.0, 1.10657895, 1.21388889, 1.32205882, 1.43125, 1.54166667, 1.65357143, 1.76730769, 1.88333333, 2.00227273] #, 2.125]
    u_input = [0.25, 0.26315789, 0.27777778, 0.29411765, 0.3125, 0.33333333, 0.35714286, 0.38461538, 0.41666667, 0.45454545] #, 0.5]

    
transformer_input = np.array([[ [x,u] for x,u in zip(x_input, u_input)]])
set_variables = ['0','1'] ##--- NB: same order as trained input ---##
model.variables = pyo.Set(initialize=set_variables)
dict_inputs = {}
for t, (u_val, x_val) in zip(model.time_input, zip(u_input, x_input)):
    dict_inputs[(t, '0')] = x_val
    dict_inputs[(t, '1')] = u_val
    

if INIT_ONLY:
    model.input_param = pyo.Var(model.time_input, model.variables, bounds=(0, 10))
else:
    model.input_param = pyo.Param(model.time_input, model.variables, initialize=dict_inputs) # t=0 to t=prediction time
    
model.input_var = pyo.Var(model.time, model.variables, bounds=(0, 10)) #t = 0 to t=1

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
if INIT_ONLY:
    model.x_init_constr_x = pyo.Constraint(expr=model.input_param[min(model.time),'0'] == x_input[0])
    model.x_init_constr_u = pyo.Constraint(expr=model.input_param[min(model.time),'1'] == u_input[0])
else:
    model.x_init_constr = pyo.Constraint(expr=model.input_var[min(model.time),'0'] == 1)
    
input_array = []
model.input_constraints = pyo.ConstraintList()      
if INIT_ONLY:
    for t in model.time:
        if t <= 0.9:
            # add constraints that x,u = x,u input values
            model.input_constraints.add(expr=model.input_var[t,'0'] == model.input_param[t,'0'])
            model.input_constraints.add(expr=model.input_var[t,'1'] == model.input_param[t,'1'])

            # create arrays with input values
            input_array.append(
                [ model.input_param[t,'0'], model.input_param[t,'1'],]
            )                 
else:  
    for t in model.time:
        if t == model.time.first():
            model.input_constraints.add(expr=model.input_var[t,'1'] == model.input_param[t,'1'])
            continue
        # create arrays for input x and u values
        if t <= 0.9:
            # add constraints that x,u = x,u input values
            model.input_constraints.add(expr=model.input_var[t,'0'] == model.input_param[t,'0'])
            model.input_constraints.add(expr=model.input_var[t,'1'] == model.input_param[t,'1'])

            # create arrays with input values
            input_array.append(
                [ model.input_param[t,'0'], model.input_param[t,'1'],]
            )  
        
        
# define objective
def _intX(m, t):
    return m.input_var[t,'0']
def _intU(m, t):
    return m.input_var[t,'1']
model.intX = dae.Integral(model.time, wrt=model.time, rule=_intX)
model.intU = dae.Integral(model.time, wrt=model.time, rule=_intU)

model.obj = pyo.Objective(
    expr=model.intX - model.intU + model.input_var[model.time.last(),'0'], sense=1
)  # -1: maximize, +1: minimize (default)

