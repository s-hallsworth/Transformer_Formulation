import pyomo.environ as pyo
from pyomo import dae
import numpy as np
import transformer as transformer
import extract_from_pretrained as extract_from_pretrained
from toy_problem_setup import *

"""
Add transformer instance/constraints to toy problem setup and solve

# Commands:
model.pprint() # view model
pyomo solve toy_problem.py --solver=gurobi --stream-solver --summary # run model in terminal ('ipopt' for NLP)

"""

# create transformer instance     
transformer = transformer.Transformer(model, ".\\data\\toy_config.json")

# add trnasformer layers and constraints
transformer.embed_input(model, "input_param","input_embed", "variables")
transformer.add_layer_norm(model, "input_embed", "layer_norm", "gamma1", "beta1")

# transformer_input = np.array([[ [x,u] for x,u in zip(x_input, u_input)]])
# layer_outputs_dict = extract_from_pretrained.get_intermediate_values(model_path, transformer_input) 
# transformer.add_FFN_2D(model, "layer_norm", "ffn_1", layer_outputs_dict['layer_normalization_2'], parameters)
transformer.add_attention(model, "layer_norm", W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
# transformer.add_residual_connection(model, model.input_embed, model.layer_norm, "mha_residual")
#transformer.add_output_constraints(model, model.mha_residual)


# Discretize model using Backward Difference method
discretizer = pyo.TransformationFactory("dae.finite_difference")
discretizer.apply_to(model, nfe=T - 1, wrt=model.time, scheme="BACKWARD")



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

from pyomo.core import *
from pyomo.opt import SolverFactory # run with python3.10
keepfiles = False  # True prints intermediate file names (.nl,.sol,...)

solver = SolverFactory('scip', solver_io='python')

#Create an IMPORT Suffix to store the iis information that willbe returned by gurobi_ampl
model.iis = Suffix(direction=Suffix.IMPORT)

## Send the model to gurobi_ampl and collect the solution
#The solver plugin will scan the model for all active suffixes
# valid for importing, which it will store into the results object
results = solver.solve(model, keepfiles=keepfiles, tee=True)
print(results)


print("")
print("IIS Results")

for component, value in model.iis.items():
    print(component.name + " " + str(value))

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
