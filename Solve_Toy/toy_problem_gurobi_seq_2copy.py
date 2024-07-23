import pyomo.environ as pyo
from pyomo import dae
import numpy as np
import transformer
import extract_from_pretrained
import transformer_b as TNN
import toy_problem_setup as tps
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

# create transformer instance   
model = tps.model  
#config_file = '.\\data\\toy_config_relu_2_seqlen_2.json' 
transformer = TNN.Transformer(model, tps.config_file, "time_input")      
        
# Define tranformer layers
std=0.774

transformer.embed_input(model, "input_param","input_embed", "variables")
transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")


transformer.add_attention(model, "layer_norm","attention_output", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
nn, input_nn, output_nn = transformer.get_fnn(model, "layer_norm_2", "FFN_1", "ffn_1", (2,2), tps.parameters)
transformer.add_residual_connection(model,"residual_1", "FFN_1", "residual_2")  
transformer.add_avg_pool(model, "residual_2", "avg_pool")
nn2, input_nn2, output_nn2 = transformer.get_fnn(model, "avg_pool", "FFN_2", "ffn_2", (1,2), tps.parameters)

## Add constraint to input_var
model.input_var_constraints = pyo.ConstraintList()

next = False
for d in model.variables:
    
    for t in model.time:
        
        if t == model.time_input.last():
            next = True
            continue
            
        if next:
            model.input_var_constraints.add(expr=model.input_var[t,d] == model.FFN_2[d])
            next = False

    
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
gurobi_model.params.MIPFocus = 1 ## focus on finding feasible solution
time_limit = 86400 # 24 hrs
solve_gurobipy(gurobi_model, time_limit) ## Solve and print

if gurobi_model.status == GRB.INFEASIBLE:
    gurobi_model.computeIIS()
    gurobi_model.write("model.ilp")     
    
else:
    ## Get optimal parameters
    #if gurobi_model.status == GRB.OPTIMAL:
    optimal_parameters = {}
    for v in gurobi_model.getVars():
        #print(f'var name: {v.varName}, var type {type(v)}')
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
input_var_soltuion = np.array(optimal_parameters["input_var"])

x = []
u = []
for i in range(len(input_var_soltuion)):
    if i % 2 == 0:
        x += [input_var_soltuion[i]]
    else: 
        u += [input_var_soltuion[i]]
print("X: ", x )
print("U: ", u )

print("actual X: ", tps.x_input)
print("actual U: ", tps.u_input)

#print((optimal_parameters))
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
