from gurobipy import Model, GRB, GurobiError

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