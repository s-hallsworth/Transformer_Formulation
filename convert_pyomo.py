import pyomo.environ as pyo
import pyomo.core.expr as pyo_expr
from gurobipy import Model, GRB
from gurobipy import LinExpr, QuadExpr
import math

def convert_pyomo_to_gurobipy(pyomo_model):
    
    # Create gurobi model with same name as pyomo model
    gurobi_model = Model(pyomo_model.name)
    
    # Mapping of Pyomo variables to Gurobi variables
    var_map = {}
    
    # Convert pyo.Var to Gurobi Var
    for var in pyomo_model.component_data_objects(pyo.Var, active=True):
        print(var)
        if var.is_indexed():
            for index in var:
                print("var: ", var)
                print("index: ", index)
                pyomo_var = var[index]
                print("pyomo_var: ", var)
                
                lb = pyomo_var.lb if pyomo_var.lb is not None else -GRB.INFINITY
                ub = pyomo_var.ub if pyomo_var.ub is not None else GRB.INFINITY
                vtype = get_gurobi_vtype(pyomo_var)
                gurobi_var = gurobi_model.addVar(lb=lb, ub=ub, name=str(pyomo_var), vtype=vtype)
                var_map[pyomo_var.name] = gurobi_var
        else:
            lb = var.lb if var.lb is not None else -GRB.INFINITY
            ub = var.ub if var.ub is not None else GRB.INFINITY
            vtype = get_gurobi_vtype(var)
            gurobi_var = gurobi_model.addVar(lb=lb, ub=ub, name=str(var), vtype=vtype)
            var_map[var.name] = gurobi_var
    
    # Update model
    gurobi_model.update()
    print(gurobi_model.getVars())
    
    # Convert objective
    for obj in pyomo_model.component_objects(pyo.Objective, active=True):
        print("objective: ",obj, obj.expr)
        expr = obj.expr
        sense = GRB.MINIMIZE if obj.sense == pyo.minimize else GRB.MAXIMIZE
        gurobi_model.setObjective(expr_to_gurobi(expr, var_map), sense=sense)
    
    # Convert constraints
    for con in pyomo_model.component_objects(pyo.Constraint, active=True):
        for index in con:
            gurobi_expr = expr_to_gurobi(con[index].body, var_map)
            
            print("constraint: ",con, con[index].body)
            if con[index].equality:
                gurobi_model.addConstr(gurobi_expr == con[index].upper)
            else:
                if con[index].has_lb():
                    gurobi_model.addConstr(gurobi_expr >= con[index].lower)
                if con[index].has_ub():
                    gurobi_model.addConstr(gurobi_expr <= con[index].upper)
    
    return gurobi_model

def get_gurobi_vtype(pyomo_var):
    if pyomo_var.domain == pyo.NonNegativeReals:
        return GRB.CONTINUOUS
    elif pyomo_var.domain == pyo.Binary:
        return GRB.BINARY
    elif pyomo_var.domain == pyo.Integers or pyomo_var.domain == pyo.NonNegativeIntegers:
        return GRB.INTEGER
    else:
        return GRB.CONTINUOUS  # Default to continuous if the domain is not specified or different

def expr_to_gurobi(expr, var_map):
    
    ## INT
    if isinstance(expr, int):
        return expr
    
    ## FLOAT
    if isinstance(expr, float):
        return expr
    
    ## Parameter
    if isinstance(expr, pyo.Param):
        return expr()
    
    ## VARIABLE
    elif isinstance(expr, pyo.Var):
        return var_map[expr.name]
    
    ## LINEAR EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.LinearExpression):
        gurobi_expr = LinExpr()
        for coef, var in zip(expr.linear_coefs, expr.linear_vars):
            gurobi_expr += coef * var_map[var.name]
        return gurobi_expr
    
    ## PRODUCT EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.ProductExpression):
        gurobi_expr = QuadExpr()
        for sub_expr in expr.args:
            gurobi_expr *= expr_to_gurobi(sub_expr, var_map)
        return gurobi_expr
    
    ## SUM EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.SumExpression):
        gurobi_expr = LinExpr()
        for sub_expr in expr.args:
            gurobi_expr += expr_to_gurobi(sub_expr, var_map)
        return gurobi_expr
    
    ## POW EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.PowExpression):
        base, exponent = expr.args
        return expr_to_gurobi(base, var_map) ** expr_to_gurobi(exponent, var_map)
    
    ## DIVISION EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.DivisionExpression):
        numerator, denominator = expr.args
        return expr_to_gurobi(numerator, var_map) / expr_to_gurobi(denominator, var_map)
    
    ## MAX EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.MaxExpression):
        return max(expr_to_gurobi(arg, var_map) for arg in expr.args)
    
    ## MIN EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.MinExpression):
        return min(expr_to_gurobi(arg, var_map) for arg in expr.args)
    
    ## NEGATION EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.NegationExpression):
        return -expr_to_gurobi(expr.args[0], var_map)
    
    ## UNARY FUNCTIONS
    elif isinstance(expr, pyo_expr.numeric_expr.UnaryFunctionExpression):
        func = expr.getname()
        arg = expr.args[0]
        if func == 'exp':
            return math.exp(expr_to_gurobi(arg, var_map))
        elif func == 'sqrt':
            return math.sqrt(expr_to_gurobi(arg, var_map))
        elif func == 'log':
            return math.log(expr_to_gurobi(arg, var_map))
        else:
            raise ValueError(f"Unsupported unary function: {func}")
    
    ## BOOLEAN EXPR
    elif isinstance(expr, pyo_expr.logical_expr.BooleanExpression):
        return expr_to_gurobi(expr.args[0], var_map)
    
    ## NOT EXPR
    elif isinstance(expr, pyo_expr.logical_expr.NotExpression):
        return not expr_to_gurobi(expr.args[0], var_map)
    
    ## XOR EXPR
    elif isinstance(expr, pyo_expr.logical_expr.XorExpression):
        arg1, arg2 = expr.args
        return expr_to_gurobi(arg1, var_map) ^ expr_to_gurobi(arg2, var_map)
    
    ## EQUIVALENT EXPR
    elif isinstance(expr, pyo_expr.logical_expr.EquivalenceExpression):
        arg1, arg2 = expr.args
        return expr_to_gurobi(arg1, var_map) == expr_to_gurobi(arg2, var_map)
    
    ## AND EXPR
    elif isinstance(expr, pyo_expr.logical_expr.AndExpression):
        return all(expr_to_gurobi(arg, var_map) for arg in expr.args)
    
    ## OR EXPR
    elif isinstance(expr, pyo_expr.logical_expr.OrExpression):
        return any(expr_to_gurobi(arg, var_map) for arg in expr.args)
    
    ## NUMERIC VALUE
    if isinstance(expr, pyo_expr.numeric_expr.NumericValue):
        return expr
    
    ## BOOLEAN VALUE
    if isinstance(expr, pyo_expr.logical_expr.BooleanValue):
        return expr
    
    else:
        raise ValueError(f"Unsupported expression type: {type(expr)}")

# Example usage
pyomo_model = pyo.ConcreteModel()
pyomo_model.heads = pyo.RangeSet(1, 10)
pyomo_model.x = pyo.Var(within=pyo.NonNegativeReals)
pyomo_model.y = pyo.Var(within=pyo.Binary)
pyomo_model.z = pyo.Var(within=pyo.NonNegativeIntegers)
pyomo_model.obj = pyo.Objective(expr=2*pyomo_model.x + 3*pyomo_model.y + 4*pyomo_model.z, sense=1)
pyomo_model.con1 = pyo.Constraint(expr=pyomo_model.x + pyomo_model.y + pyomo_model.z <= 10)
pyomo_model.con2 = pyo.Constraint(expr=pyomo_model.x - pyomo_model.y >= 3)
pyomo_model.con3 = pyo.Constraint(expr=pyomo_model.x**2 >= 3)
pyomo_model.con4 = pyo.Constraint(expr=pyomo_model.x/2 <= 3)
pyomo_model.con5 = pyo.Constraint(expr= pyo.exp(pyomo_model.x) <= 100)
pyomo_model.con6 = pyo.Constraint(expr= pyomo_model.y <= sum(pyomo_model.heads[d] for d in [1,2,3,4,5,6,7,8]))

gurobi_model = convert_pyomo_to_gurobipy(pyomo_model)
# gurobi_model.optimize()

# if gurobi_model.status == GRB.OPTIMAL:
#     for v in gurobi_model.getVars():
#         print(f'{v.varName}: {v.x}')
#     print(f'Objective: {gurobi_model.objVal}')
