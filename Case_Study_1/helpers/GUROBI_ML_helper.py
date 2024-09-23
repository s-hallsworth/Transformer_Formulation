
from omlt.neuralnet.layer import DenseLayer, InputLayer
from omlt.neuralnet.network_definition import NetworkDefinition
import numpy as np
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import re


def weights_to_NetDef(new_name, NN_name, input_shape, model_parameters
):
    input_shape = np.array(input_shape)
    nn = Sequential(name=new_name)
    nn.add(Input(input_shape))
    weights_list = [ ]
    
    # get weights, biases, num inputs, num outputs for each layer of FFN
    for layer_name, val in model_parameters[NN_name].items():
        # print(layer_name)
        
        if "dense" in layer_name or "linear" in layer_name:
            if "linear" in layer_name:
                weights = np.array(val['W']).transpose(1,0)
                bias = np.array(val['b'])
            else:
                weights = np.array(val['W'])
                bias = np.array(val['b'])
            n_layer_inputs, n_layer_nodes = np.shape(weights)
            
            # print(weights.shape, bias.shape)
            # print(val['activation'])
            
            # Determine activation function
            if val['activation'] =='relu':
                weights_list += [[weights, bias]]
                nn.add(Dense(n_layer_nodes, activation='relu',name=layer_name))
            elif val['activation'] == None or val['activation'] == 'linear':
                weights_list += [[weights, bias]]
                nn.add(Dense(n_layer_nodes, activation=None, name=layer_name))
            else:
                raise TypeError(f'Error in layer: {layer_name}. Activation function not currently supported for ', val['activation'])

    # print("model summary",nn.summary())

    # set weights for each layer
    for i, w in enumerate(weights_list):
        nn.layers[i].set_weights(w)
        
    return nn

def get_inputs_gurobipy_FNN(input_nn, output_nn, map_var):
    inputs = []
    outputs = []
    prev_input = {}
    prev_output = {}
    i = ""
    o = ""
    i_flag = False
    o_flag = False
    
    for index, value in map_var.items():
    
        if  str(index).startswith(input_nn.name):
            if "," in str(index):
                if i == "":
                    i = str(index).split(',')[0]
                    prev_input[str(index).split(',')[0]] = [value]
                    i_flag = True
                    
                elif str(index).split(',')[0] != i:
                    inputs += [prev_input[i]]
                    i = str(index).split(',')[0]
                    prev_input[str(index).split(',')[0]] = [value]
                    
                    i_flag = False
                else:
                    prev_input[str(index).split(',')[0]] += [value]
                    i_flag = True
            else:
                inputs += [value]
              
                       
            
        elif str(index).startswith(output_nn.name):
            if "," in str(index):
                if o == "":
                    o = str(index).split(',')[0]
                    prev_output[str(index).split(',')[0]] = [value]
                    o_flag = True
                    
                elif str(index).split(',')[0] != o:
                    outputs += [prev_output[o]]
                    o = str(index).split(',')[0]
                    prev_output[str(index).split(',')[0]] = [value]
                    o_flag = False
                else:
                    prev_output[str(index).split(',')[0]] += [value]
                    o_flag = True
            else:
                outputs += [value]  
                
    # add last vars to list
    if i_flag:
        inputs  += [prev_input[i]]   
    if o_flag:  
        outputs += [prev_output[o]]            

    return inputs, outputs 

def add_envelope(attention_output_name, transformer_block, gurobi_model, block_map):
    
    # constraint lists
    constr_convex = []
    constr_concave = []
    constr_convex_tp = []
    constr_convex_tp_sct = []
    constr_concave_tp = []
    constr_concave_tp_sct = []
    
    # define names of predefined vars (cannot add vars on the fly in gurobi)
    att_name = "Block_"+attention_output_name       
    tie_point_cc = att_name + ".tie_point_cc"
    tie_point_cv = att_name + ".tie_point_cv"
    tie_point_cc_prime = att_name + ".tie_point_cc_prime"
    tie_point_cv_prime = att_name + ".tie_point_cv_prime"
    tp_cv_mult_1 = att_name + ".tp_cv_mult_1"
    tp_cv_mult_2 = att_name + ".tp_cv_mult_2"
    tp_cc_mult_1 = att_name + ".tp_cc_mult_1"
    tp_cc_mult_2 = att_name + ".tp_cc_mult_2"
    
    BigM_s = 0.5
    sct = att_name + ".sct"
    compatibility =  att_name + ".compatibility"
    
    s_cv= att_name + ".s_cv"
    t_cv= att_name + ".t_cv"
    s_cc= att_name + ".s_cc"
    t_cc= att_name + ".t_cc"
    tp_cv =att_name + ".tp_cv"
    tp_cc =att_name + ".tp_cc"

    attention_weight_cc = att_name + ".attention_weight_cc"
    attention_weight_x_cc_prime = att_name + ".attention_weight_x_cc_prime"
    attention_weight_x_cc= att_name + ".attention_weight_x_cc"
    attention_weight_cv = att_name + ".attention_weight_cv"
    attention_weight_x_cv_prime = att_name + ".attention_weight_x_cv_prime"
    attention_weight_x_cv = att_name + ".attention_weight_x_cv"
    attention_weight =   att_name + ".attention_weight"
    
    tp_cv_sct = att_name + ".tp_cv_sct"
    tp_cv_sct_mult_1 = att_name + ".tp_cv_sct_mult_1"
    tp_cv_sct_mult_2 = att_name + ".tp_cv_sct_mult_2"
    tp_cv_sct_mult_1_2 = att_name + ".tp_cv_sct_mult_1_2"
    
    tp_cc_sct = att_name + ".tp_cc_sct"
    tp_cc_sct_mult_1 = att_name + ".tp_cc_sct_mult_1"
    tp_cc_sct_mult_2 = att_name + ".tp_cc_sct_mult_2"
    tp_cc_sct_mult_1_2 = att_name + ".tp_cc_sct_mult_1_2"

    
    for tb_index in transformer_block.index_set():
        att_block =  getattr(transformer_block[tb_index], att_name)
        
        # add indicator variable constraints
        s_cv_pvar = getattr(att_block, "s_cv")
        for index in s_cv_pvar.index_set():
            # get gurobi vars
            i = ",".join(str(x) for x in index)
            s_cv_gvar = block_map[(transformer_block.name, str(tb_index), s_cv, i)]
            s_cc_gvar = block_map[(transformer_block.name, str(tb_index), s_cc, i)]
            attention_weight_gvar = block_map[(transformer_block.name, str(tb_index), attention_weight, i)]
            attention_weight_cv_gvar = block_map[(transformer_block.name, str(tb_index), attention_weight_cv, i)]
            attention_weight_cc_gvar = block_map[(transformer_block.name, str(tb_index), attention_weight_cc, i)]
            compatibility_gvar = block_map[(transformer_block.name, str(tb_index), compatibility, i)]
            sct_gvar = block_map[(transformer_block.name, str(tb_index), sct, i)]
            
            # s_cv
            gurobi_model.cbLazy(attention_weight_gvar.ub <= 0.5  + (BigM_s * s_cv_gvar))
            gurobi_model.cbLazy(attention_weight_gvar.ub >= BigM_s *  s_cv_gvar)
            
            # s_cc
            gurobi_model.cbLazy(attention_weight_gvar.lb >= 0.5 - (BigM_s *  s_cc_gvar)) 
            gurobi_model.cbLazy((BigM_s * s_cc_gvar) <= 0.5 + BigM_s - attention_weight_gvar.lb)
            
            # f(x) >= f_cv(x): attention weight >= convex envelope
            gurobi_model.cbLazy(attention_weight_gvar >= attention_weight_cv_gvar)
            
            # f(x) <= f_cc(x): attention weight <= concave envelope
            gurobi_model.cbLazy(attention_weight_gvar <= attention_weight_cc_gvar)
            
            # sct(x)
            A = ((attention_weight_gvar.ub - attention_weight_gvar.lb) / (compatibility_gvar.ub - compatibility_gvar.lb )) 
            b = ( (compatibility_gvar.ub * attention_weight_gvar.lb) - (compatibility_gvar.lb * attention_weight_gvar.ub)) /(compatibility_gvar.ub - compatibility_gvar.lb )
            gurobi_model.cbLazy(sct_gvar   == (A *  compatibility_gvar) + b )
            
            # # Add concave/convex evelope function constraints
            # # # when f(UB) <= 0.5: convex
            constr_convex.append( attention_weight_cv_gvar == attention_weight_gvar)
            constr_convex.append( attention_weight_cc_gvar == sct_gvar )
            
            for constr in constr_convex:
                gurobi_model.cbLazy(gurobi_model.addGenConstrIndicator(s_cv_gvar, 0, constr))
        
        ####
        # var = getattr(att_block, "tp_cc_sct_mult_1_2")
        # print(var.index_set())
        # for v_index in var.index_set():
        #     v = ",".join(str(x) for x in v_index)
        #     print( block_map[(transformer_block.name, str(tb_index), tp_cc_sct_mult_1_2, v)])
            
