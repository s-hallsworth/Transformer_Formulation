import pyomo.environ as pyo
import numpy as np

import extract_from_pretrained
import transformer_b_exp as TNN
import toy_problem_setup as tps
import os
from omlt import OmltBlock
import convert_pyomo
from gurobipy import Model, GRB, GurobiError
from gurobi_ml import add_predictor_constr
from GUROBI_ML_helper import get_inputs_gurobipy_FNN
from print_stats import solve_gurobipy
from data_gen import gen_x_u
import transformer_intermediate_results as tir
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off


"""
Add transformer instance/constraints to toy problem setup and solve GUROBIPY Model

"""

# Define pyomo model 
model_path = ".\\TNN_enc_0002.keras" #7.keras"
config_file = '.\\data\\toy_config_relu_10.json'#_TNN_7.json' 
T = 90000 # time steps
seq_len = 10
pred_len = 1
window = seq_len + pred_len

preds_u = []
preds_x = []

Trained_preds_x=[]
pred_times = []

start_indices = list(range(0,T, int(T/12)))
start_indices.append(T - window - 1)

gen_x, gen_u, _,_ = gen_x_u(T)
x_input = gen_x[0, 0 : seq_len]
u_input = gen_u[0, 0 : seq_len]
input_data = [x_input, u_input ]


for start_time in range(0,12) : #start_indices: #[600]: # \
    print("START TIME: ", start_time)

    # define optimization model
    model = tps.setup_toy( T, start_time ,seq_len, pred_len, model_path, config_file, input_data)
    transformer = TNN.Transformer(model, tps.config_file, "time_input") 
    
    # Initialise transformers
    std_list = []
    for i in range(seq_len):
        std_list += [np.std([tps.x_input[i], tps.u_input[i]])]
    std = max(std_list)
    
    transformer.embed_input(model, "input_param", "input_embed")
    transformer.add_layer_norm(model,"input_embed", "layer_norm", "gamma1", "beta1", std)
    transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
    transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
    transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
    nn, input_nn, output_nn = transformer.get_fnn(model, "layer_norm_2", "FFN_1", "ffn_1", (seq_len,2), tps.parameters)
    transformer.add_residual_connection(model,"residual_1", "FFN_1", "residual_2")  
    transformer.add_avg_pool(model, "residual_2", "avg_pool")
    nn2, input_nn2, output_nn2 = transformer.get_fnn(model, "avg_pool", "FFN_2", "ffn_2", (1,2), tps.parameters)

    ## Add constraint to input_var
    model.input_var_constraints = pyo.ConstraintList()

    last_time_1 = False
    d = model.variables.first()
    d2 = model.variables.last()
    delta_T = 1/T
    M = 10 * (delta_T)
        
    for t_index, t in enumerate(model.time):
        
        if t == model.time_input.last():
            last_time_1  = True
            
        elif last_time_1 :
            model.input_var_constraints.add(expr=model.input_var[t,d] == model.FFN_2[d])
            last_time_1  = False
                
        # if  t < model.time.last() and start_time > 0:
        #     model.input_var_constraints.add(expr= model.input_var[model.time.at(t_index + 2),d2] - model.input_var[t ,d2]
        #                                     <= M
        #                                     )
            
        
        
    # # Convert to gurobipy
    gurobi_model, map_var = convert_pyomo.to_gurobi(model)

    ## Add FNN1 to gurobi model
    inputs_1, outputs_1 = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
    pred_constr1 = add_predictor_constr(gurobi_model, nn, inputs_1, outputs_1)

    inputs_2, outputs_2 = get_inputs_gurobipy_FNN(input_nn2, output_nn2, map_var)
    pred_constr2 = add_predictor_constr(gurobi_model, nn2, inputs_2, outputs_2)

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
    time_limit = 21600 
    solve_gurobipy(gurobi_model, time_limit) ## Solve and print
    

    if gurobi_model.status == GRB.INFEASIBLE:
        print(f"Model at start time {start_time} is infeasible")
        # preds_x += [None, None]
        # preds_u += [None, None]
        # pred_times += [tps.time_sample[start_time + seq_len + i] for i in range(pred_len)]
        break
        # gurobi_model.computeIIS()
        # gurobi_model.write(f"model_{start_time}.ilp")     
        
    else:
        ## Get optimal parameters
        if gurobi_model.status == GRB.OPTIMAL:
            optimal_parameters = {}
            print(model.intXU)
            
            for v in gurobi_model.getVars():
                
                if "[" in v.varName:
                    name = v.varname.split("[")[0]
                    if name in optimal_parameters.keys():
                        optimal_parameters[name] += [v.x]
                    else:
                        optimal_parameters[name] = [v.x]
                else:    
                    optimal_parameters[v.varName] = v.x
            print(
            "objective value:", gurobi_model.getObjective().getValue()
            )
    

        ## Print X, U --> input var, control var 
        input_var_soltuion = np.array(optimal_parameters["input_var"])
        input_par_soltuion = np.array(optimal_parameters["input_param"])
        x = []
        u = []
        xx = []
        uu = []
        for i in range(len(input_var_soltuion)):
            if i % 2 == 0:
                x += [input_var_soltuion[i]]
            else: 
                u += [input_var_soltuion[i]]
                
        for i in range(len(input_par_soltuion)):
            if i % 2 == 0:
                xx += [input_par_soltuion[i]]
            else: 
                uu += [input_par_soltuion[i]]
        print("start time: ", start_time)
        print("X: ", x )
        print("U: ", u )
        print("Xx: ", xx )
        print("Uu: ", uu )
        
        preds_x += [x[-1]]
        
        if start_time > 0: #pred at t+1 solves for value of u at t and x and t+2
            preds_u += [u[-2]]
        pred_times += [tps.time_sample[start_time + seq_len + i] for i in range(pred_len)]
        
        # find result of trained TNN
        uu[-1] = u[-2]
        output = tir.generate_TNN_outputs(model_path, [xx, uu])
        print(output)
        Trained_preds_x += [output[0]]
        print("trained X: ", Trained_preds_x )
    
        # update input
        input_data = [ x[1:] , u[1:]]
        print("next input_data: ", input_data)
            
    print("actual X: ", gen_x[0, start_time : start_time + window])
    print("actual U: ", gen_u[0, start_time : start_time + window])
 
#save to file
import csv
file_name = "results_trajectory_seq_2_.csv"   
with open(file_name, 'a', newline='') as file:
    writer = csv.writer(file)
    values = []
    values.append(T)
    values.append(pred_times)
    values.append(preds_x)
    values.append(preds_u)
    
    values.append('')
    writer.writerow(values)
    
#plot results  
preds_x = np.array(preds_x)
preds_u = np.array(preds_u)
 
print(preds_x)
print(pred_times)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
# plt.plot(tps.time_sample, tps.gen_x[0,0:], 's-', label = 'X* Analytical')
# plt.plot(tps.time_sample, tps.gen_u[0,0:], 's-', label = 'U* Analytical')
plt.plot(pred_times, gen_x[0,0: len(pred_times)], 's-', label = 'X* Analytical')
plt.plot(pred_times, gen_u[0,0: len(pred_times)], 's-', label = 'U* Analytical')
plt.plot(pred_times, preds_x, '--x', 
         linewidth= 2, label = 'X* Solver')
plt.plot(pred_times[1:], preds_u, '--x', 
         linewidth= 2, label = 'U* Solver')
plt.plot(pred_times, Trained_preds_x, '--', 
         linewidth= 2, label = 'X* Trained TNN')

plt.legend()
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