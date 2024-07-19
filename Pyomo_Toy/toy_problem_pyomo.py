import pyomo.environ as pyo
from pyomo import dae
import numpy as np
import transformer
import extract_from_pretrained
from toy_problem_setup import *
from pyomo.opt import SolverFactory 
from print_stats import solve_pyomo
import toy_problem_setup as tps
from omlt import OmltBlock

"""
Add transformer instance/constraints to toy problem setup and solve PYOMI MODEL

# Commands:
model.pprint() # view model
pyomo solve toy_problem.py --solver=gurobi --stream-solver --summary # run model in terminal ('ipopt' for NLP)

"""

# create transformer instance     
transformer = transformer.Transformer(model, '.\\data\\toy_config_relu_2_seqlen_2.json' )# ".\\data\\toy_config_relu_2.json")

# add trnasformer layers and constraints
transformer.embed_input(model, "input_param","input_embed", "variables")
transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1", std=0.774)
#transformer.add_attention(model, "layer_norm", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)


transformer.add_attention_approx(model, "layer_norm", tps.W_q, tps.W_k, tps.W_v, tps.W_o, tps.b_q, tps.b_k, tps.b_v, tps.b_o)
transformer.add_residual_connection(model,"input_embed", "attention_output", "residual_1")
transformer.add_layer_norm(model, "residual_1", "layer_norm_2", "gamma2", "beta2")
transformer.add_FFN_2D(model, "layer_norm_2", "ffn_1", (10,2), tps.parameters) 
transformer.add_residual_connection(model,"residual_1", "ffn_1", "residual_2")  
transformer.add_avg_pool(model, "residual_2", "avg_pool")
transformer.add_FFN_2D(model, "avg_pool", "ffn_2", (1,2), tps.parameters)

# output constraints
for enum_d, d in enumerate(model.variables):
    model.input_constraints.add(expr=model.ffn_2[d] == model.input_var[model.time.last(),d])
            
                   
# Discretize model using Backward Difference method
discretizer = pyo.TransformationFactory("dae.finite_difference")
discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")


#Solve
solver = pyo.SolverFactory("gurobi")
# solver = pyo.SolverFactory('mindtpy').solve(model,
#                                    strategy='FP',
#                                    mip_solver='cplex',
#                                    nlp_solver='ipopt',
#                                    tee=True
#                                    )
#solver = SolverFactory('gurobi', solver_io='python')
time_limit = None
result = solve_pyomo(model, solver, time_limit)

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
#----------------------------
optimal_parameters = get_optimal_dict(result, model) # get optimal parameters & reformat  --> (1, input_feature, sequence_element)

input_var_soltuion = optimal_parameters["input_var"]
print(input_var_soltuion)

#print(optimal_parameters)  
     
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
