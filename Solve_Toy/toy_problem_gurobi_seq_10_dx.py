import pyomo.environ as pyo
from pyomo import dae
import numpy as np

import extract_from_pretrained
import transformer_b as TNN
import toy_problem_setup_dx as tps
import os
from omlt import OmltBlock
import convert_pyomo
from gurobipy import Model, GRB, GurobiError
from gurobi_ml import add_predictor_constr
from GUROBI_ML_helper import get_inputs_gurobipy_FNN
from print_stats import solve_gurobipy


os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0' # turn off floating-point round-off


"""
Add transformer instance/constraints to toy problem setup and solve GUROBIPY Model

"""

# Define pyomo model 
model_path = ".\\TNN_9.keras"
config_file = '.\\data\\toy_config_relu_10_TNN_7.json' 
T = 9000 # time steps
seq_len = 10
pred_len = 2
window = seq_len + pred_len

preds_u = []
preds_x = []
pred_times = []

for start_time in [8987]: #range(0,T, int(T/10)):
    #start_time = 89987
    for i in range(pred_len):
        pred_times += [start_time + seq_len + i + 1]
        
    model = tps.setup_toy( T, start_time ,seq_len, pred_len, model_path, config_file)

    # Define time sets
    Time_start = 1

    model.time_input_2 = dae.ContinuousSet(initialize=tps.time[Time_start : Time_start + seq_len])
    model.input_2 = pyo.Var(model.time_input_2, model.variables, bounds=(tps.LB_input, tps.UB_input))
    
    # Define transformers 
    transformer = TNN.Transformer(model, config_file, "time_input")      
    transformer2 = TNN.Transformer(model, config_file, "time_input_2")

    # Initialise transformers
    std_list = []
    for i in range(seq_len):
        std_list += [np.std([tps.x_input[i], tps.u_input[i]])]
    std = max(std_list)
    
    transformer.embed_input(model, "input_param", "input_embed", "variables")
    transformer.add_layer_norm(model,"input_embed", "layer_norm", "gamma1", "beta1",std)
    transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
    transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
    transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
    nn, input_nn, output_nn = transformer.get_fnn(model, "layer_norm_2", "FFN_1", "ffn_1", (seq_len,2), tps.parameters)
    transformer.add_residual_connection(model,"residual_1", "FFN_1", "residual_2")  
    transformer.add_avg_pool(model, "residual_2", "avg_pool")
    nn2, input_nn2, output_nn2 = transformer.get_fnn(model, "avg_pool", "FFN_2", "ffn_2", (1,2), tps.parameters)

        
    # Define transformer 2

    
    transformer2.embed_input(model, "input_2","input_embed2", "variables")
    transformer2.add_layer_norm(model, "input_embed2", "layer_norm2", "gamma1", "beta1")
    transformer2.add_attention(model, "layer_norm2", "attention_output2" , tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
    transformer2.add_residual_connection(model,"input_embed2", "attention_output2", "residual_12")
    transformer2.add_layer_norm(model, "residual_12", "layer_norm_22", "gamma2", "beta2")
    nn21, input_nn21, output_nn21 = transformer2.get_fnn(model, "layer_norm_22", "FFN_12", "ffn_1", (seq_len,2), tps.parameters)
    transformer2.add_residual_connection(model,"residual_12", "FFN_12", "residual_22")  
    transformer2.add_avg_pool(model, "residual_22", "avg_pool22")
    nn22, input_nn22, output_nn22 = transformer2.get_fnn(model, "avg_pool22", "FFN_22", "ffn_2", (1,2), tps.parameters)


    ## Add constraint to X
    model.X_constraints = pyo.ConstraintList()

    last_time_1 = False
    last_time_2 = False
    d = model.variables.first()
    d2 = model.variables.last()
    delta_T = 1/T
    M = 10 * (delta_T)
    print(d, d2)

    # FNN outputs update X and dX    
    for t_index, t in enumerate(model.time):
        if t == model.time_dx.first(): # first prediction for x(t)
            model.X_constraints.add(expr=model.dX[t] == model.FFN_2[d]) # predict dx
            model.X_constraints.add(expr=model.X[t,d] == model.X[model.time.at(t_index),d] + model.FFN_2[d]) # x = x_prev + dx
            model.X_constraints.add(expr=  model.X[t,d2] -  model.X[model.time.at(t_index),d2] <= M) #u(t) should be smooth (first order) 
        if t == model.time_dx.last(): # second prediction for x(t)
            model.X_constraints.add(expr=model.dX[t] == model.FFN_22[d]) # predict dx
            model.X_constraints.add(expr=model.X[t,d] == model.X[model.time.at(t_index),d] + model.FFN_22[d]) # x = x_prev + dx
            model.X_constraints.add(expr=  model.X[t,d2] -  model.X[model.time.at(t_index),d2] <= M) #u(t) should be smooth (first order) 
        
        
    # # Convert to gurobipy
    gurobi_model, map_var = convert_pyomo.to_gurobi(model)

    ## Add FNN1 to gurobi model
    inputs_1, outputs_1 = get_inputs_gurobipy_FNN(input_nn, output_nn, map_var)
    pred_constr1 = add_predictor_constr(gurobi_model, nn, inputs_1, outputs_1)

    inputs_2, outputs_2 = get_inputs_gurobipy_FNN(input_nn2, output_nn2, map_var)
    pred_constr2 = add_predictor_constr(gurobi_model, nn2, inputs_2, outputs_2)

    # inputs_21, outputs_21 = get_inputs_gurobipy_FNN(input_nn21, output_nn21, map_var)
    # pred_constr21 = add_predictor_constr(gurobi_model, nn21, inputs_21, outputs_21)

    # inputs_22, outputs_22 = get_inputs_gurobipy_FNN(input_nn22, output_nn22, map_var)
    # pred_constr22 = add_predictor_constr(gurobi_model, nn22, inputs_22, outputs_22)
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
    time_limit = 21600 # 24 hrs
    solve_gurobipy(gurobi_model, time_limit) ## Solve and print

    if gurobi_model.status == GRB.INFEASIBLE:
        gurobi_model.computeIIS()
        gurobi_model.write("model.ilp")     
        
    else:
        ## Get optimal parameters
        #if gurobi_model.status == GRB.OPTIMAL:
        optimal_parameters = {}
        for v in gurobi_model.getVars():
            # if 'int' or 'X' in v.varName:
            #     print(f'var name: {v.varName}, var type: {type(v)}, var value: {v.x}')
            if "[" in v.varName:
                name = v.varname.split("[")[0]
                if name in optimal_parameters.keys():
                    optimal_parameters[name] += [v.x]
                else:
                    optimal_parameters[name] = [v.x]
            else:    
                optimal_parameters[v.varName] = v.x
                    
        ##

        print(
        "objective value:", gurobi_model.getObjective().getValue()
        )

        
                

        ## Print X, U --> input var, control var 
        X_soltuion = np.array(optimal_parameters["X"])
        print("dx: ", optimal_parameters["dX"])

        x = []
        u = []
        for i in range(len(X_soltuion)):
            if i % 2 == 0:
                x += [X_soltuion[i]]
            else: 
                u += [X_soltuion[i]]
                
        x = x - tps.pe[0,0]
        u = u - tps.pe[0,1]
        print("X: ", x )
        print("U: ", u )
        preds_x += [x[-pred_len:]]
        preds_u += [u[-pred_len:]]

    print("actual X: ", tps.gen_x[0, -tps.window :])
    print("actual U: ", tps.gen_u[0, -tps.window :])
 
#save to file
import csv
file_name = "results_trajectory_2.csv"   
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
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))

plt.plot(tps.time_sample, tps.gen_x[0,:], 's-', label = 'X* Analytical')
plt.plot(tps.time_sample, tps.gen_u[0,:], 's-', label = 'U* Analytical')
plt.plot(pred_times, preds_x, '-', label = 'X* Solver')
plt.plot(pred_times, preds_u, '-', label = 'U* Solver')

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