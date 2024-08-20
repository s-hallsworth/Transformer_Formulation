import pyomo.environ as pyo
from pyomo import dae
import numpy as np

import extract_from_pretrained
import transformer_b_exp2 as TNN
import toy_problem_setup_exp_2 as tps
import os
from omlt import OmltBlock
import convert_pyomo
from gurobipy import Model, GRB, GurobiError
from gurobi_ml import add_predictor_constr
from GUROBI_ML_helper import get_inputs_gurobipy_FNN
from print_stats import solve_gurobipy
from data_gen import gen_x_u
import transformer_intermediate_results as tir
import GUROBI_ML_helper

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off


"""
Add transformer instance/constraints to toy problem setup and solve GUROBIPY Model

"""

# Define pyomo model 
model_path = ".\\TNN_enc_0002.keras" #7.keras"
config_file = '.\\data\\toy_config_relu_10.json'#_TNN_7.json' 
T = 9000 # time steps
seq_len = 10
pred_len = 2
gen_x, gen_u, _,_ = gen_x_u(T)

for pred_len in range(2, T-seq_len, 1): #[600]: # \
    print()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    window = seq_len + pred_len
    start_time = T-window - 1
    print("START TIME: ", start_time)
    print("PRED LENGTH: ", pred_len)
    # initialize input and pred lists
    x_input = gen_x[0, start_time : start_time + seq_len]
    u_input = gen_u[0, start_time : start_time + seq_len]
    input_data = [x_input, u_input ]
    preds_u = []
    preds_x = []
    Trained_preds_x=[]
    pred_times = []
    
    print("------------- SET UP ---------------")
    print(x_input)
    print(u_input)

    # initialize otpimization model
    model = tps.setup_toy( T, start_time ,seq_len, pred_len, model_path, config_file, input_data)
    print("- set up complete")

    # Define transformer
    transformer = TNN.Transformer(model, tps.config_file, "TNN") 
    print("- created TNN instance")
     
    def block_rule(b,t):
        # Add TNN acrhicture constraints to block
        transformer.add_input(tps.model.input_param, "input", b)
        transformer.add_layer_norm(model,"input", "layer_norm", "gamma1", "beta1")
        transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
        transformer.add_residual_connection(model,"input", "attention_output", "residual_1")
        transformer.add_layer_norm(model,  "residual_1", "layer_norm_2", "gamma2", "beta2")
        transformer.get_fnn(model, "layer_norm_2", "FFN_1")
        transformer.add_residual_connection(model, "residual_1", "FFN_1", "residual_2")  
        transformer.add_avg_pool(model, "residual_2", "avg_pool")
        transformer.get_fnn(model, "avg_pool", "FFN_2")
    
    print("- defining block rule")
    model.tnn_block = pyo.Block(model.pred_window, rule = block_rule)
    print("- block rule defined")
    
    model.tnn_block_constraints = pyo.ConstraintList() 
   # input linking rule
    for t_index, t in enumerate(model.pred_window):
        index = t_index + 1
        for p, pos in enumerate(model.seq_length):
            index_p = p+1
            for d in model.model_dims:
                # tnn block input == input window 
                model.tnn_block_constraints.add(expr= model.tnn_block[t].input[pos, d] == model.t_inputs[t,pos, d])

                # output of previous time window TNN gives the last value of next time window
                if t != model.pred_window.last():
                    model.tnn_block_constraints.add(expr= model.tnn_block[t].FFN_2[d] == model.t_inputs[model.pred_window.at(index +1),model.seq_length.last(), d])
                      
                # link final tnn output
                else: 
                    model.tnn_block_constraints.add(expr= model.tnn_block[t].FFN_2[d] == model.input_var[model.time.last(), d]) 
                             
                # link time window inputs
                if p > 0  and t < model.pred_window.last():
                    model.tnn_block_constraints.add(expr= model.t_inputs[t, pos, d] == model.t_inputs[model.pred_window.at(index +1), model.seq_length.at(index_p - 1), d])

               
    # # Convert to gurobipy
    print("- converting to gurobipy")
    gurobi_model, map_var = convert_pyomo.to_gurobi(model)
    

    ## Add FNNs to gurobi model
    def get_fnn_details(model, input_var_name, output_var_name, nn_name, input_shape, model_parameters):
        
        input_var = getattr(model, input_var_name)
        output_var = getattr(model, output_var_name)
        nn= GUROBI_ML_helper.weights_to_NetDef(output_var_name, nn_name, input_shape, model_parameters)

        return nn, input_var, output_var

    print("- adding gurobiML FFNs")
    for  t in model.pred_window:
        nn, input_nn, output_nn = get_fnn_details(model.tnn_block[t],"layer_norm_2", "FFN_1", "ffn_1", (seq_len,2), tps.parameters )
        inputs_1, outputs_1 = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
        pred_constr1 = add_predictor_constr(gurobi_model, nn, inputs_1, outputs_1)
        
        nn2, input_nn2, output_nn2 = get_fnn_details(model.tnn_block[t], "avg_pool", "FFN_2", "ffn_2", (1,2), tps.parameters )
        inputs_2, outputs_2 = get_inputs_gurobipy_FNN(input_nn2, output_nn2, map_var)
        pred_constr2 = add_predictor_constr(gurobi_model, nn2, inputs_2, outputs_2)

    print("- updating gurobi model")
    gurobi_model.update()
    #pred_constr.print_stats()

    ## Print Header
    print("------------------------------------------------------")
    print("                  SOLVE INFORMATION ")
    print("------------------------------------------------------")
    print()

    print(f"Problem: Toy Problem")
    print(f"Solver: Gurobi")
    print(f"Exp Approx: Dynamic Outer Bound")
    print(f"NN formulation: Gurobipy NN (using max constraint)")
    print("------------------------------------------------------")
    print()

    ## Optimize
    # gurobi_model.params.SolutionLimit = 10 ##
    # gurobi_model.params.MIPFocus = 1 ## focus on finding feasible solution
    time_limit = 21600 # 6 hrs
    
    
    # gurobi_model.feasRelaxS(0, False, True, True)
    print("- solving optimization model")
    runtime, optimality_gap = solve_gurobipy(gurobi_model, time_limit) ## Solve and print
    
    # print('\nSlack values:')
    # orignumvars = 92994
    # slacks = gurobi_model.getVars()[orignumvars:]
    # for sv in slacks:
    #     if sv.X > 1e-9:
    #         print('%s = %g' % (sv.VarName, sv.X))
    
    ## Print bounds on vars
    # for var in gurobi_model.getVars():
    #     lb = var.LB
    #     ub = var.UB
    #     if ub - lb > 10: 
    #         print(f"variable {var.VarName},  bound: [{lb}, {ub}] (range: {ub-lb}), abs(var): {abs(var.X)}")
            
            

    if gurobi_model.status == GRB.INFEASIBLE:
        print(f"Model at start time {start_time} is infeasible")
        opt = None
        # preds_x += [None, None]
        # preds_u += [None, None]
        # pred_times += [tps.time_sample[start_time + seq_len + i] for i in range(pred_len)]
        # gurobi_model.computeIIS()
        # gurobi_model.write(f"model_{start_time}.ilp")     
        
    else:
        ## Get optimal parameters
        if gurobi_model.status == GRB.OPTIMAL:
            optimal_parameters = {}
            # print(model.intXU)
            
            for v in gurobi_model.getVars():
                
                if "[" in v.varName:
                    name = v.varname.split("[")[0]
                    if name in optimal_parameters.keys():
                        optimal_parameters[name] += [v.x]
                    else:
                        optimal_parameters[name] = [v.x]
                else:    
                    optimal_parameters[v.varName] = v.x
                    
             
            ##
            opt = gurobi_model.getObjective().getValue()
            print(
            "objective value:", gurobi_model.getObjective().getValue()
            )  

        ## Print X, U --> input var, control var 
        input_var_soltuion = np.array(optimal_parameters["input_var"])
        # for item, val  in optimal_parameters.items():
        #     for elem in [".s_cv", ".s_cc", ".t_cv", ".t_cc"]:
        #         if elem in str(item):
        #             print(f"{item}: {val}")

        x = []
        u = []
        for i in range(len(input_var_soltuion)):
            if i % 2 == 0:
                x += [input_var_soltuion[i]]
            else: 
                u += [input_var_soltuion[i]]
        
        preds_x += [x[-pred_len:]]
        preds_u += [u[-pred_len:]]
        pred_times += [tps.time_sample[start_time + seq_len + i] for i in range(pred_len)]

        input_par_soltuion = np.array(optimal_parameters["input_param"])
        xp = []
        up = []
        for i in range(len(input_par_soltuion)):
            if i % 2 == 0:
                xp += [input_par_soltuion[i]]
            else: 
                up += [input_par_soltuion[i]]
        print("X fixed input: ", xp )
        print("U fixed input: ", up )
        print("X: ", x )
        print("U: ", u )
        
        # find result of trained TNN
        # output = tir.generate_TNN_outputs(model_path, input_data)
        # print(output)
        # Trained_preds_x += [output[0]]
        # print("trained X: ", Trained_preds_x )
    
        # # update input
        # x_input = gen_x[0, start_time : start_time  + seq_len]
        # u_input = gen_u[0, start_time : start_time  + seq_len]
        # input_data = [x_input, u_input ]
        # print("next input_data: ", input_data)
            
    print("actual X: ", gen_x[0, start_time : start_time + window])
    print("actual U: ", gen_u[0, start_time : start_time + window])
 
 
    #save to file
    import csv
    file_name = "results_prediction_blocks.csv"   
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        headers = []
        headers.append('T')
        headers.append('seq_len')
        headers.append('pred_len')
        headers.append('pred_times')
        headers.append('preds_x')
        headers.append('preds_u')
        headers.append('actual_x')
        headers.append('actual_u')
        headers.append('runtime')
        headers.append('opt gap')
        headers.append('objective value')
        writer.writerow(headers)
        
        values = []
        values.append(T)
        values.append(seq_len)
        values.append(pred_len)
        values.append(pred_times)
        values.append(preds_x)
        values.append(preds_u)
        values.append( gen_x[0, start_time : start_time + window])
        values.append( gen_u[0, start_time : start_time + window])
        values.append(runtime)
        values.append(optimality_gap)
        values.append(opt)
        values.append('')
        writer.writerow(values)
    
    #plot results  
    preds_x = np.array(preds_x).flatten()
    preds_u = np.array(preds_u).flatten()


    print(preds_x)
    print(preds_u)
    print(pred_times)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    # plt.plot(tps.time_sample, tps.gen_x[0,0:], 's-', label = 'X* Analytical')
    # plt.plot(tps.time_sample, tps.gen_u[0,0:], 's-', label = 'U* Analytical')
    plt.plot(tps.time[start_time : start_time + window], gen_x[0, start_time : start_time + window], 's-', label = 'X* Analytical')
    plt.plot(tps.time[start_time : start_time + window], gen_u[0, start_time : start_time + window], 's-', label = 'U* Analytical')
    # plt.plot(tps.time[start_time : start_time + window], x, '--x', 
    #          linewidth= 2, label = 'X* Solver')
    # plt.plot(tps.time[start_time : start_time + window], u, '--x', 
    #          linewidth= 2, label = 'U* Solver')

    plt.plot(pred_times, preds_x, '--o', 
            linewidth= 2, label = 'X* Solver')
    plt.plot(pred_times, preds_u, '--x', 
            linewidth= 2, label = 'U* Solver')
    # plt.plot(pred_times, Trained_preds_x, '--', 
    #          linewidth= 2, label = 'X* Trained TNN')

    plt.legend()
    plt.savefig(".\\images\\prediction_blocks\\figure_pred_"+str(pred_len))
    plt.show()

# print(optimal_parameters["input_2"])
# print(optimal_parameters["FFN_22"])
# print(optimal_parameters)
# print("-----------------------------")
# print(optimal_parameters)

## Expected values:
# x_input = [1.0, 1.10657895, 1.21388889, 1.32205882, 1.43125, 1.54166667, 1.65357143, 1.76730769, 1.88333333, 2.00227273, 2.125]
# u_input = [0.25, 0.26315789, 0.27777778, 0.29411765, 0.3125, 0.33333333, 0.35714286, 0.38461538, 0.41666667, 0.45454545, 0.5]

# # -------------------- SOLVE MODEL ----------------------------------- #

def get_optimal_dict(result, model):
    optimal_parameters = {}
    if result.solver.status == 'ok' and result.solver.termination_condition == 'optimal':
        for varname, var in model.component_map(pyo.Var).items():
            # Check if the variable is indexed
            if var.is_indexed():
                optimal_parameters[varname] = {index: pyo.value(var[index]) for index in var.index_set()}
            else:
                optimal_parameters[varname] = pyo.value(var)
        #print("Optimal Parameters:", optimal_parameters)
    else:
        print("No optimal solution obtained.")
    
    return optimal_parameters


# from pyomo.core import *
# from pyomo.opt import SolverFactory # run with python3.10
# keepfiles = False  # True prints intermediate file names (.nl,.sol,...)

# solver = SolverFactory('gurobi', solver_io='python')
# #opts = {'halt_on_ampl_error': 'yes','tol': 1e-7, 'bound_relax_factor': 1.0}

# #Create an IMPORT Suffix to store the iis information that willbe returned by gurobi_ampl
# model.iis = Suffix(direction=Suffix.IMPORT)

# ## Send the model to gurobi_ampl and collect the solution
# #The solver plugin will scan the model for all active suffixes
# # valid for importing, which it will store into the results object
# results = solver.solve(model, keepfiles=keepfiles, tee=True)
# print(results)


# print("")
# print("IIS Results")

# for component, value in model.iis.items():
#     print(component.name + " " + str(value))

# model.compatibility.pprint()
#---------------
# import pyomo
# #pyomo.contrib.mis.compute_infeasibility_explanation(model, solver=solver)
# pyomo.contrib.iis.write_iis(model, "iss_file.ilp", solver='scip')

# result = solver.solve(model)
# # get optimal parameters & reformat  --> (1, input_feature, sequence_element)
# optimal_parameters = get_optimal_dict(results, model)
# print(optimal_parameters)
# result = solver.solve(model) #, symbolic_solver_labels=True, tee=True, load_solutions=True, options=opts)

# print(optimal_parameters)
