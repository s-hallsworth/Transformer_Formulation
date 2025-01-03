from gurobipy import GRB, GurobiError
from pyomo.opt import TerminationCondition, SolverStatus
import time
import csv
import os
import matplotlib.pyplot as plt



def save_gurobi_results(model, file_name, descr_short, rep, tnn_config):
    """
    Saves Gurobi optimization results to a CSV file.

    Args:
        model (gurobipy.Model): The solved Gurobi model.
        file_name (str): Name of the CSV file (without extension) to save results.
        descr_short (str): Short description of the experiment.
        rep (int): The repetition number of the experiment.
        tnn_config (dict): Configuration dictionary containing TNN-specific parameters.

    Writes:
        A CSV file containing various metrics and results of the optimization process.

    Notes:
        - Includes model statistics such as the number of variables, constraints, runtime, and optimality gap.
        - Handles MIP-specific metrics like the number of nodes explored and solutions found.
    """

    # Collect the required information
    time_limit = model.Params.TimeLimit
    threads = model.Params.Threads
    num_vars = model.NumVars
    num_constraints = model.NumConstrs
    num_integer_vars = sum(1 for v in model.getVars() if v.VType == GRB.INTEGER)
    num_binary_vars = sum(1 for v in model.getVars() if v.VType == GRB.BINARY)
    num_continuous_vars = sum(1 for v in model.getVars() if v.VType == GRB.CONTINUOUS)
    runtime_seconds = model.Runtime
    runtime_work_units = model.Work
    num_iterations = model.IterCount
    num_nodes_explored = model.NodeCount
    num_solutions_found = model.SolCount
    optimality_gap = model.MIPGap if model.IsMIP else None
    objective_value = model.ObjVal if model.Status == GRB.OPTIMAL else None
    root_node_relax_obj_value = model.ObjBoundC
    #root_node_relax_time = model.NodeWork
    root_node_iterations = model.BarIterCount

    # Write to CSV
    output_data = [
        ['Name', file_name],
        ['Exp', descr_short],
        ['Rep', rep],
    ]

    for key, value in tnn_config.items():
        output_data += [[key, value]]

    framework = "gurobipy"
    solver = "gurobi"
    output_data += [
        ['Framework', framework],
        ['Solver', solver],
        ['Time Limit (seconds)', time_limit],
        ['Num Threads', threads],
        ['Num Vars', num_vars],
        ['Num Constraints', num_constraints],
        ['Num Cont Vars', num_continuous_vars],
        ['Num Int Vars', num_integer_vars],
        ['Num Bin Vars', num_binary_vars],
        ['Run Time (s)', runtime_seconds],
        ['Run Time (wu)', runtime_work_units],
        ['Num Iters', num_iterations],
        ['Num Nodes Expl', num_nodes_explored],
        ['Num Solns', num_solutions_found],
        ['Opt Gap', optimality_gap],
        ['Obj Value', objective_value],
        ['RNR Obj Value', root_node_relax_obj_value],
        ['RNR Iters', root_node_iterations],
    ]

    # Save the data to a CSV file
    with open(f'{file_name}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Metric', 'Value'])
        csvwriter.writerows(output_data)

    print("Solve results saved to ", file_name+".csv")


def solve_gurobipy(model, time_limit, callback=None):
    """
    Solves a Gurobi optimization model and prints the results.

    Args:
        model (gurobipy.Model): The Gurobi model to be solved.
        time_limit (float or None): Time limit in seconds for the solver. If None, no limit is applied.
        callback (optional): Callback function for advanced solver interactions.

    Returns:
        tuple: A tuple containing:
            - runtime (float): The runtime of the optimization in seconds.
            - optimality_gap (float or None): The optimality gap, if applicable.

    Notes:
        - Supports solving MIP and continuous models.
        - Prints key solver metrics such as solve status, runtime, number of solutions found, and optimality gap.
    """

    # Set a time limit
    if time_limit is not None:
        model.setParam('TimeLimit',time_limit)
        print("------------------------------------------------------")
        print()
    
    try:
        # Optimize the model
        model.optimize(callback)
        
        print("------------------------------------------------------")
        print()
        # Check if the optimization was successful
        if model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        elif model.status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        elif model.status == GRB.TIME_LIMIT:
            print("Time limit reached.")
        else:
            print("Optimization ended with status %d" % model.status)
        
        # Print runtime, number of solutions, and optimality gap
        runtime = model.Runtime
        num_solutions = model.SolCount
        try:
            optimality_gap = model.MIPGap
            print(f"Optimality gap: {optimality_gap:.4f}")
        except:
            optimality_gap = None
        
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Number of solutions found: {num_solutions}")
        

    except GurobiError as e:
        print(f"Gurobi Error: {e.errno} - {e}")
    except AttributeError as e:
        print(f"Attribute Error: {e}")
        
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print()
    return runtime, optimality_gap
    
def solve_pyomo(model, solver, time_limit):
    """
    Solves a Pyomo optimization model and prints the results.

    Args:
        model (pyomo.ConcreteModel): The Pyomo model to be solved.
        solver (pyomo.opt.SolverFactory): The Pyomo solver to be used.
        time_limit (float or None): Time limit in seconds for the solver. If None, no limit is applied.

    Returns:
        pyomo.opt.SolverResults: The results object containing solution information.

    Notes:
        - Supports time limits via solver options.
        - Extracts and prints key metrics such as runtime, solver status, termination condition, and optimality gap.
    """

    if time_limit is not None:
        solver.options['timelimit'] = time_limit
    print('Time Limit: ', time_limit)
    print("------------------------------------------------------")
    print()
    
    # Record start time
    start_time = time.time()
    
    # Solve the model
    results = solver.solve(model, tee=True)

    # Record end time
    end_time = time.time()
    
    # Calculate runtime
    runtime = end_time - start_time
    
    # Extract solver status and termination condition
    solver_status = results.solver.status
    termination_condition = results.solver.termination_condition
    
    # Print runtime
    print(f"Runtime: {runtime} seconds")
    
    # Number of solutions found
    if hasattr(results.solver, 'num_solutions'):
        num_solutions = results.solver.num_solutions
        print(f"Number of solutions: {num_solutions}")
    else:
        print("Number of solutions: N/A (Solver-specific feature)")
    
    # Optimality gap
    if solver_status == SolverStatus.ok and termination_condition == TerminationCondition.optimal:
        print("Optimal solution found.")
        optimality_gap = 0.0
    elif solver_status == SolverStatus.ok and termination_condition in [TerminationCondition.feasible, TerminationCondition.maxTimeLimit]:
        # Extract the optimality gap
        if 'MipGap' in results.solver:
            optimality_gap = results.solver.MipGap
        else:
            optimality_gap = None  # This might vary depending on the solver
        print(f"Optimality gap: {optimality_gap}")
    else:
        print(f"Solver status: {solver_status}")
        print(f"Termination condition: {termination_condition}")
        optimality_gap = None
        
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print()
    
    return results

def save_results(exp_name, exp_set_name, path, dict, expected_value, opt_value, expected_x_value, opt_x_value, show_plt=True):
    """
    Saves experiment results to a CSV file and generates a plot.

    Args:
        exp_name (str): Name of the experiment.
        exp_set_name (str): Name of the experiment set.
        path (str): Directory path where the results will be saved.
        dict (dict): Dictionary containing experiment-specific metrics and configurations.
        expected_value (np.ndarray): Array of expected output values.
        opt_value (np.ndarray): Array of optimized output values.
        expected_x_value (np.ndarray): Corresponding x-values for the expected results.
        opt_x_value (np.ndarray): Corresponding x-values for the optimized results.
        show_plt (bool): Whether to display the plot. Defaults to True.

    Saves:
        - A CSV file with experiment metrics and configurations.
        - A PNG plot comparing expected values from trained TNN and values from optimised TNN.

    Notes:
        - Automatically appends to the CSV file if it exists; otherwise, creates a new one.
        - Generates a plot comparing expected and solved results.
    """

    
    path = path
    
    file_name = f".\\{path}\\{exp_set_name}_results.csv"  
    if os.path.exist(file_name):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            values = []
            values.append(exp_name)
            
            for key, value in dict.items():
                values.append(value)
                
            values.append('')
            writer.writerow(values)

    else:
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            headers = []
            headers.append('Exp Name')
            
            values = []
            values.append(exp_name)
            
            for key, value in dict.items():
                headers.append(key)
                values.append(value)
                
            values.append('')
            writer.writerow(headers)
            writer.writerow(values)
    
    #plot results  
    plt.figure(figsize=(6, 4))
    for e in expected_value.shape[0]:
        plt.plot(expected_x_value, expected_value[e], 's-', label = f'{e} Expected')
    for o in opt_value.shape[0]:
        plt.plot(opt_x_value, opt_value[e], '--x', label = f'{o} Solved')

    plt.legend()
    plt.savefig(f".\\{path}\\{exp_set_name}_image_{exp_name}.png") #save plot
    if show_plt:
        plt.show()