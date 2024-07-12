import pyomo.environ as pyo

def solve_pyomo(model, solver, time_limit=None):
    # Example of setting solver options
    # solver.options['mip_solver'] = 'CPLEX'
    if time_limit is not None:
         solver.options['timelimit'] = time_limit
    
    try:
        result = solver.solve(model)
        return result
    except AttributeError as e:
        print(f"Solver error: {str(e)}")
        raise

# Define a simple Pyomo model
model = pyo.ConcreteModel()
model.x = pyo.Var(bounds=(0, None), initialize=1.0)
model.y = pyo.Var(bounds=(0, None), initialize=1.0)
model.obj = pyo.Objective(expr=model.x**2 + model.y**2, sense=pyo.minimize)
model.constr = pyo.Constraint(expr=model.x + model.y >= 1)

# Create a solver object for Octeract
solver = pyo.SolverFactory('octeract')

# Solve the model

result = solve_pyomo(model, solver)
print(result)