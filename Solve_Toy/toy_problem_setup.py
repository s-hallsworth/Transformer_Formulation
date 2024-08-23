import pyomo.environ as pyo
#from pyomo import dae
import numpy as np
import extract_from_pretrained as extract_from_pretrained


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

def setup_toy( T,start_time, seq_len, pred_len, model_path, config_file, input_data):
    model_path = model_path
    config_file = config_file
    model = pyo.ConcreteModel(name="(TOY_OPTIMAL_CONTROL)")
    layer_names, parameters ,_ = extract_from_pretrained.get_learned_parameters(model_path)
    window = seq_len + pred_len
    
    ## define problem sets, vars, params
    time_sample = np.linspace(0, 1, num= T) # entire time t=0:1 including prediction times
    time = time_sample
    model.time_input = pyo.Set(initialize=time[start_time : start_time + seq_len]) # t < prediction times
    model.time = pyo.Set(initialize=time[start_time : start_time + window])
    set_variables = ['0','1'] ##--- NB: same order as trained input ---##
    model.model_dims = pyo.Set(initialize=set_variables)
    


    # Set bounds based on training dataset
    UB_input = 3.5
    LB_input = 0 

    # Define inputs
    x_input = input_data[0]
    u_input = input_data[1]
    
    print("------------- SET UP ---------------")
    print(x_input)
    print(u_input)

    dicseq_lens = {}
    for t, (u_out, x_out) in zip(model.time_input, zip(u_input, x_input)):
        dicseq_lens[(t, '0')] = x_out
        dicseq_lens[(t, '1')] = u_out
    
    model.input_param = pyo.Param(model.time_input, model.model_dims, initialize=dicseq_lens)#, bounds=(LB_input, UB_input))
    model.input_var = pyo.Var(model.time, model.model_dims , bounds=(LB_input, UB_input)) #t = 0 to t=1

    model.input_constraints = pyo.ConstraintList() 
    for t_index, t in enumerate(model.time_input): 
        model.input_constraints.add(expr=model.input_param[t,'0'] == model.input_var[t,'0'])
        model.input_constraints.add(expr=model.input_param[t,'1'] == model.input_var[t,'1'])  
        
    ## define transformer sets, vars, params
    dict_gamma1 = {(v): val for v,val in zip(model.model_dims, parameters['layer_normalization_1','gamma'])}
    dict_beta1 = {(v): val for v,val in zip(model.model_dims,  parameters['layer_normalization_1','beta'])}
    model.gamma1 = pyo.Param(model.model_dims, initialize = dict_gamma1)
    model.beta1 = pyo.Param(model.model_dims, initialize = dict_beta1)


    dict_gamma2 = {(v): val for v,val in zip(model.model_dims, parameters['layer_normalization_2','gamma'])}
    dict_beta2 = {(v): val for v,val in zip(model.model_dims,  parameters['layer_normalization_2','beta'])}
    model.gamma2 = pyo.Param(model.model_dims, initialize = dict_gamma2)
    model.beta2 = pyo.Param(model.model_dims, initialize = dict_beta2)
    
    
    # define weights and biases
    W_q = parameters['mutli_head_attention_1','W_q']
    W_k = parameters['mutli_head_attention_1','W_k']
    W_v = parameters['mutli_head_attention_1','W_v']
    W_o = parameters['mutli_head_attention_1','W_o']

    b_q = parameters['mutli_head_attention_1','b_q']
    b_k = parameters['mutli_head_attention_1','b_k']
    b_v = parameters['mutli_head_attention_1','b_v']
    b_o = parameters['mutli_head_attention_1','b_o']

        
    # define integral constraints

    # int_factor = (time[-1] - time[0])/(3 * window)
    # model.intXU = pyo.Var(model.model_dims, bounds=(0, UB_input * (time[-1] - time[0])))
    # for d in model.model_dims:
    #     sum_d = model.input_var[model.time.first(), d ] + model.input_var[model.time.last(), d]
    #     for t_index, t in enumerate(model.time):
    #         if t < model.time.last() and t > model.time.first():
    #             # Use Simpsons Rule to find discrete integral
    #             if t_index % 2 == 0:
    #                 sum_d += 2 * model.input_var[model.time.at(t_index + 1), d]
                    
    #             else:
    #                 sum_d += 4 * model.input_var[model.time.at(t_index + 1), d]
    #     model.input_constraints.add(expr = model.intXU[d] == int_factor * sum_d )

    # Set objective function
    # model.obj = pyo.Objective(
    #     expr= model.intXU['0'] - model.intXU['1'] + model.input_var[model.time.last(),'0'], sense=1
    # )  # -1: maximize, +1: minimize (default)


    # model.obj = pyo.Objective(
    #     expr= sum(model.input_var[t,'0'] - model.input_var[t,'1'] for t in model.time) + model.input_var[model.time.last(),'0'], sense=1
    # ) 
    
    model.obj = pyo.Objective(
        expr= sum( (model.input_var[t,'0'] - model.input_var[t,'1'])**2 for t in model.time) + model.input_var[model.time.last(),'0'], sense=1
    ) 
    
    globals().update(locals()) #make all variables from this function global
    return model