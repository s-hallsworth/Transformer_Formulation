from gurobipy import Model, GRB, GurobiError
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition, SolverStatus
import pyomo.environ as pyo
import time

def solve_gurobipy(model, time_limit):
    # Set a time limit
    model.setParam('TimeLimit',time_limit)
    print("------------------------------------------------------")
    print()
    
    try:
        # Optimize the model
        model.optimize()
        
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
        optimality_gap = model.MIPGap
        
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Number of solutions found: {num_solutions}")
        print(f"Optimality gap: {optimality_gap:.4f}")

    except GurobiError as e:
        print(f"Gurobi Error: {e.errno} - {e}")
    except AttributeError as e:
        print(f"Attribute Error: {e}")
        
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print()
    
    
def solve_pyomo(model, solver, time_limit):
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