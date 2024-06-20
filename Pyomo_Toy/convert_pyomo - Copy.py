import pyomo.environ as pyo
import pyomo.core.expr as pyo_expr
import pyomo.core.base as pyo_base
from pyomo import dae
from gurobipy import Model, GRB, LinExpr, QuadExpr
import math

def convert_pyomo_to_gurobipy(pyomo_model, func_nonlinear=1):
    
    # Create gurobi model with same name as pyomo model
    gurobi_model = Model(pyomo_model.name)
    gurobi_model.params.FuncNonlinear = func_nonlinear # 0: piecewise linear approx, 1: outer approx of function, -1: controlled by FuncNonLinear
    
    # Mapping of Pyomo variables to Gurobi variables
    var_map = {}
    
    # Convert pyo.Var to Gurobi Var
    for var in pyomo_model.component_data_objects(pyo.Var, active=True):

        if var.is_indexed():
            for index in var:
                pyomo_var = var[index]
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
    
    # Convert objective
    for obj in pyomo_model.component_objects(pyo.Objective, active=True):
        expr = obj.expr
        sense = GRB.MINIMIZE if obj.sense == pyo.minimize else GRB.MAXIMIZE
        expr, _= expr_to_gurobi(expr, var_map, gurobi_model)
        gurobi_model.setObjective(expr, sense=sense)
       
    # Convert constraints
    for con in pyomo_model.component_objects(pyo.Constraint, active=True):
        for index in con:
            gurobi_expr, add_constraint = expr_to_gurobi(con[index].body, var_map, gurobi_model)
            
            if add_constraint :
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

def expr_to_gurobi(expr, var_map, gurobi_model):
    print("expression", expr, type(expr))
    
    ## INT
    if isinstance(expr, int):
        return expr, True
    
    ## FLOAT
    elif isinstance(expr, float):
        return expr, True
    
    ## INDEXED
    try:
        if expr.is_indexed():
           for i,sub_expr in enumerate(expr.args):
            var = gurobi_model.addVar(name=arg.name+"_index_"+str(i))
            gurobi_model.addConstr(var == expr_to_gurobi(sub_expr, var_map, gurobi_model)[0])
            
            gurobi_model.update()
        return var, False
    except:
        pass
    
    ## PARAMETER
    if isinstance(expr, pyo.Param):
        func = expr.getname()
        arg = expr.args[0]
        var = gurobi_model.setParam(arg.name, expr.args[0])
        return var, True
    
    ## VARIABLE
    elif isinstance(expr, (pyo.Var,pyo_base.var._GeneralVarData)):
        print("var")
        lb = expr.lb if expr.lb is not None else -GRB.INFINITY
        ub = expr.ub if expr.ub is not None else GRB.INFINITY
        vtype = get_gurobi_vtype(expr)
        gurobi_var = gurobi_model.addVar(lb=lb, ub=ub, name=str(expr), vtype=vtype)
        return gurobi_var , True
    
    ## LINEAR EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.LinearExpression):
        gurobi_expr = 0.0
        for coef, var in zip(expr.linear_coefs, expr.linear_vars):
            print("linear expr ",var.name, var_map[var.name], type(var_map[var.name]))
            print("coeff",coef, type(coef) )
            gurobi_expr += coef * var_map[var.name]
        return gurobi_expr, True
    
    ## PRODUCT EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.ProductExpression):
        gurobi_expr = 1.0
        for sub_expr in expr.args:
            sub_gurobi_expr, _ = expr_to_gurobi(sub_expr, var_map, gurobi_model)
            gurobi_expr *= sub_gurobi_expr
        return gurobi_expr, True
    
    ## SUM EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.SumExpression):
        gurobi_expr = 0.0
        for sub_expr in expr.args:
            gurobi_expr += expr_to_gurobi(sub_expr, var_map, gurobi_model)[0]
        return gurobi_expr, True
    
    ## POW EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.PowExpression):
        base, exponent = expr.args
        return  expr_to_gurobi(base, var_map, gurobi_model)[0] ** expr_to_gurobi(exponent, var_map, gurobi_model)[0], True
    
    ## DIVISION EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.DivisionExpression):
        numerator, denominator = expr.args
        return expr_to_gurobi(numerator, var_map, gurobi_model)[0] / expr_to_gurobi(denominator, var_map, gurobi_model)[0], True
    
    ## MAX EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.MaxExpression):
        new_expr = []
        for arg in expr.args:
            new_expr += [expr_to_gurobi(arg, var_map, gurobi_model)[0]]
        return max( new_expr), True
    
    ## MIN EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.MinExpression):
        new_expr = []
        for arg in expr.args:
            new_expr += [expr_to_gurobi(arg, var_map, gurobi_model)[0]]
        return min(new_expr ), True
    
    ## NEGATION EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.NegationExpression):
        return -expr_to_gurobi(expr.args[0], var_map, gurobi_model)[0], True
    
    ## UNARY FUNCTIONS
    elif isinstance(expr, pyo_expr.numeric_expr.UnaryFunctionExpression):
        func = expr.getname()
        arg = expr.args[0]
        arg_gurobi = expr_to_gurobi(arg, var_map, gurobi_model)[0]
        if func == 'exp':
            expx = gurobi_model.addVar(name=arg.name+"_exp")
            gurobi_model.addGenConstrExp(arg_gurobi, expx)
            gurobi_model.update()
            return expx, False
        elif func == 'sqrt':
            sqrtx = gurobi_model.addVar(name=arg.name+"_sqrt")
            gurobi_model.addConstr(sqrtx * sqrtx == arg_gurobi)
            gurobi_model.update()
            return sqrtx, False
        elif func == 'log': #natural log
            logx = gurobi_model.addVar(name=arg.name+"_log")
            gurobi_model.addGenConstrLog(arg_gurobi, logx)
            gurobi_model.update()
            return logx, False
        else:
            raise ValueError(f"Unsupported unary function: {func}")
        
    ## DAE.INTEGRAL EXPR
    elif isinstance(expr, dae.Integral):
        print("intefgral",expr, expr.getname(), expr.args)
        func = expr.getname()
        arg = expr.args[0]

        integrand_gurobi_expr, _ = expr_to_gurobi(arg, var_map, gurobi_model)
        
        return integrand_gurobi_expr, True
    
    ## BOOLEAN EXPR
    elif isinstance(expr, pyo_expr.logical_expr.BooleanExpression):
        return expr_to_gurobi(expr.args[0], var_map, gurobi_model)[0], True
    
    ## NOT EXPR
    elif isinstance(expr, pyo_expr.logical_expr.NotExpression):
        return not expr_to_gurobi(expr.args[0], var_map, gurobi_model)[0], True
    
    ## XOR EXPR
    elif isinstance(expr, pyo_expr.logical_expr.XorExpression):
        arg1, arg2 = expr.args
        return expr_to_gurobi(arg1, var_map, gurobi_model)[0] ^ expr_to_gurobi(arg2, var_map, gurobi_model)[0], True
    
    ## EQUIVALENT EXPR
    elif isinstance(expr, pyo_expr.logical_expr.EquivalenceExpression):
        arg1, arg2 = expr.args
        return expr_to_gurobi(arg1, var_map, gurobi_model)[0] == expr_to_gurobi(arg2, var_map, gurobi_model)[0], True
    
    ## AND EXPR
    elif isinstance(expr, pyo_expr.logical_expr.AndExpression):
        return all(expr_to_gurobi(arg, var_map, gurobi_model)[0] for arg in expr.args), True
    
    ## OR EXPR
    elif isinstance(expr, pyo_expr.logical_expr.OrExpression):
        return any(expr_to_gurobi(arg, var_map, gurobi_model)[0] for arg in expr.args), True
    
    ## NUMERIC VALUE
    elif isinstance(expr, pyo_expr.numeric_expr.NumericValue):
        return expr, True
    
    ## BOOLEAN VALUE
    elif isinstance(expr, pyo_expr.logical_expr.BooleanValue):
        return expr, True
    
    else:
        raise ValueError(f"Unsupported expression type: {type(expr)}")



############################################################################
# Example usage
############################################################################

# pyomo_model = pyo.ConcreteModel()
# pyomo_model.heads = pyo.RangeSet(1, 10)
# pyomo_model.x = pyo.Var(within=pyo.NonNegativeReals)
# pyomo_model.y = pyo.Var(within=pyo.Binary)
# pyomo_model.z = pyo.Var(within=pyo.NonNegativeIntegers)
# pyomo_model.obj = pyo.Objective(expr=2*pyomo_model.x + 3*pyomo_model.y + 4*pyomo_model.z, sense=1)

# pyomo_model.con1 = pyo.Constraint(expr=pyomo_model.x + pyomo_model.y + pyomo_model.z <= 10)
# pyomo_model.con2 = pyo.Constraint(expr=pyomo_model.x + pyomo_model.y >= 2)
# pyomo_model.con3 = pyo.Constraint(expr=pyomo_model.x**2 <= 4)
# pyomo_model.con4 = pyo.Constraint(expr=pyomo_model.x/2 <= 1)
# pyomo_model.con5 = pyo.Constraint(expr= pyo.exp(pyomo_model.x) <= 100)
# pyomo_model.con6 = pyo.Constraint(expr= pyomo_model.y <= sum(pyomo_model.heads[d] for d in [1,2,3,4,5,6,7,8]))
# pyomo_model.con7 = pyo.Constraint(expr= pyo.log(pyo.exp(pyomo_model.x)) <= 100)
# pyomo_model.con8 = pyo.Constraint(expr= pyo.sqrt(pyomo_model.x**2) <= 100)


# gurobi_model = convert_pyomo_to_gurobipy(pyomo_model)
# gurobi_model.optimize()

# if gurobi_model.status == GRB.OPTIMAL:
#     for v in gurobi_model.getVars():
#         print(f'{v.varName}: {v.x}')
#     print(f'Objective: {gurobi_model.objVal}')
