import pyomo.environ as pyo
from pyomo import dae
import numpy as np
import extract_from_pretrained as extract_from_pretrained
from data_gen import gen_x_u, positional_encoding

"""
Define toy problem parametrs and var then run from another script like toy_problem.py or transformer_test.py
"""


## 
# model_path = "..\\Transformer_Toy\\transformer_small_relu_2_seqlen_2.keras"
# config_file = '.\\data\\toy_config_relu_2_seqlen_2.json' 
# #model_path = "..\\Transformer_Toy\\transformer_small_relu_2_TOY.keras" 
# T = 9000 # time steps
# seq_len = 2
# pred_len = 2
# window = seq_len + pred_len

## create model

def setup_toy( T,start_time, seq_len, pred_len, model_path, config_file):
    model_path = model_path
    config_file = config_file
    model = pyo.ConcreteModel(name="(TOY_OPTIMAL_CONTROL)")
    layer_names, parameters, TNN_model = extract_from_pretrained.get_learned_parameters(model_path)
    window = seq_len + pred_len
    
    ## generate input data
    gen_x, gen_u, _,_ = gen_x_u(T)
    pe = positional_encoding(T, d_model=2)
    gen_x_pe = gen_x + pe[0,0]
    gen_u_pe = gen_u + pe[0,1]
    
    ## define problem sets, vars, params
    time_sample = np.linspace(0, 1, num= T) # entire time t=0:1 including prediction times
    time = time_sample[start_time : start_time + window]
    model.time_input = dae.ContinuousSet(initialize=time[0: seq_len]) # t < prediction times
    model.time_dx = dae.ContinuousSet(initialize=time[-pred_len: ])
    model.time = dae.ContinuousSet(initialize=time)
    set_variables = ['0','1'] ##--- NB: same order as trained input ---##
    model.variables = pyo.Set(initialize=set_variables)
    
    

    # Set bounds based on training dataset
    UB_input = 4
    LB_input = 0 

    # Define inputs
    x_input = gen_x_pe[0, start_time : start_time + window]
    u_input = gen_u_pe[0, start_time : start_time + window]
    print(x_input)
    print(u_input)
    
    print(len(time[0: seq_len]), len(x_input))

    dicseq_lens = {}
    for t, (u_out, x_out) in zip(model.time_input, zip(u_input[0: seq_len], x_input[0: seq_len])):
        dicseq_lens[(t, '0')] = x_out
        dicseq_lens[(t, '1')] = u_out

    #model.input_param = pyo.Var(model.time_input, model.variables, initialize=dicseq_lens, bounds=(LB_input, UB_input))
    model.input_param = pyo.Param(model.time_input, model.variables, initialize=dicseq_lens)#, bounds=(LB_input, UB_input))

    model.X= pyo.Var(model.time, model.variables, bounds=(LB_input, UB_input)) #t = 0 to t=1
    model.dX= pyo.Var(model.time_dx)


    ## define transformer sets, vars, params
    dict_gamma1 = {(v): val for v,val in zip(model.variables, parameters['layer_normalization_1','gamma'])}
    dict_beta1 = {(v): val for v,val in zip(model.variables,  parameters['layer_normalization_1','beta'])}
    model.gamma1 = pyo.Param(model.variables, initialize = dict_gamma1)
    model.beta1 = pyo.Param(model.variables, initialize = dict_beta1)


    dict_gamma2 = {(v): val for v,val in zip(model.variables, parameters['layer_normalization_2','gamma'])}
    dict_beta2 = {(v): val for v,val in zip(model.variables,  parameters['layer_normalization_2','beta'])}
    model.gamma2 = pyo.Param(model.variables, initialize = dict_gamma2)
    model.beta2 = pyo.Param(model.variables, initialize = dict_beta2)
    
    

    W_q = parameters['multi_head_attention_1','W_q']
    W_k = parameters['multi_head_attention_1','W_k']
    W_v = parameters['multi_head_attention_1','W_v']
    W_o = parameters['multi_head_attention_1','W_o']

    b_q = parameters['multi_head_attention_1','b_q']
    b_k = parameters['multi_head_attention_1','b_k']
    b_v = parameters['multi_head_attention_1','b_v']
    b_o = parameters['multi_head_attention_1','b_o']


    
    # initialise X 
    model.input_constraints = pyo.ConstraintList()
    for t_index, t in enumerate(model.time_input):
        model.input_constraints.add(expr=model.input_param[t,'0'] == model.X[t,'0'])
        model.input_constraints.add(expr=model.input_param[t,'1'] == model.X[t,'1'])


    int_factor = 1/(3 * T)
    model.intXU = pyo.Var(model.variables)
    for d in model.variables:
        sum_d = model.X[model.time.first(), d ] + model.X[model.time.last(), d]
        for t_index, t in enumerate(model.time):
            if t < model.time.last() and t > model.time.first():
                # Use Simpsons Rule to find discrete integral
                if t_index % 2 == 0:
                    sum_d += 2 * model.X[model.time.at(t_index + 1), d]
                    
                else:
                    sum_d += 4 * model.X[model.time.at(t_index + 1), d]
        model.input_constraints.add(expr = model.intXU[d] == int_factor * sum_d )

    # Set objective function
    model.obj = pyo.Objective(
        expr= model.intXU['0'] - model.intXU['1'] + model.X[model.time.last(),'0'], sense=1
    )  # -1: maximize, +1: minimize (default)
    
    # # define integral constraints
    # def _intX(m, t):
    #     return m.X[t,'0']
    # def _intU(m, t):
    #     return m.X[t,'1']
    # model.intX = dae.Integral(model.time, wrt=model.time, rule=_intX)
    # model.intU = dae.Integral(model.time, wrt=model.time, rule=_intU)


    # # Set objective function
    # model.obj = pyo.Objective(
    #     expr=model.intX - model.intU + model.X[model.time.last(),'0'], sense=1
    # )  # -1: maximize, +1: minimize (default)

    globals().update(locals()) #make all variables from this function global
    return model