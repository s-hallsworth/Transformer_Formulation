import pyomo.environ as pyo
import pyomo.core.expr as pyo_expr
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
    
    ## INT
    if isinstance(expr, int):
        return expr, True
    
    ## FLOAT
    if isinstance(expr, float):
        return expr, True
    
    ## PARAMETER
    if isinstance(expr, pyo.Param):
        return expr(), True
    
    ## VARIABLE
    elif isinstance(expr, pyo.Var):
        return var_map[expr.name], True
    
    ## LINEAR EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.LinearExpression):
        gurobi_expr = LinExpr()
        for coef, var in zip(expr.linear_coefs, expr.linear_vars):
            gurobi_expr += coef * var_map[var.name]
        return gurobi_expr, True
    
    ## PRODUCT EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.ProductExpression):
        gurobi_expr = QuadExpr()
        for sub_expr in expr.args:
            gurobi_expr *= expr_to_gurobi(sub_expr, var_map, gurobi_model)[0]
        return gurobi_expr, True
    
    ## SUM EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.SumExpression):
        gurobi_expr = LinExpr()
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
        return expr_to_gurobi(numerator, var_map)[0] / expr_to_gurobi(denominator, var_map, gurobi_model)[0], True
    
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
    # elif isinstance(expr, dae.Integral):
    #     integral_var = gurobi_model.addVar(name=f"{expr.name}_integral")
        
    #     # Convert the integrand expression to Gurobi expression
    #     integrand_gurobi_expr, _ = expr_to_gurobi(expr.integrand, var_map, gurobi_model)
        
    #     # Add the integral constraint
    #     gurobi_model.addConstr(integral_var == gurobi_model.addLConstr(
    #         gurobi_model.integral(0, expr.tau, integrand_gurobi_expr), GRB.EQUAL, expr.upper_bound
    #     ))
        
    #     gurobi_model.update()
        
    #     return integral_var, True
    
    ## SCALAR INTEGRAL EXPR (ScalarIntegral)
    # elif isinstance(expr, pyo.ScalarIntegral):
    #     integral_var = gurobi_model.addVar(name=f"{expr.name}_integral", lb=expr.lb, ub=expr.ub)
        
    #     # Convert the integrand expression to Gurobi expression
    #     integrand_gurobi_expr, _ = expr_to_gurobi(expr.f, var_map, gurobi_model)
        
    #     # Add the integral constraint
    #     gurobi_model.addConstr(integral_var == gurobi_model.integral(0, expr.t, integrand_gurobi_expr))
        
    #     gurobi_model.update()
        
    #     return integral_var, True
    
    ## BOOLEAN EXPR
    elif isinstance(expr, pyo_expr.logical_expr.BooleanExpression):
        return expr_to_gurobi(expr.args[0], var_map, gurobi_model)[0]
    
    ## NOT EXPR
    elif isinstance(expr, pyo_expr.logical_expr.NotExpression):
        return not expr_to_gurobi(expr.args[0], var_map, gurobi_model)[0]
    
    ## XOR EXPR
    elif isinstance(expr, pyo_expr.logical_expr.XorExpression):
        arg1, arg2 = expr.args
        return expr_to_gurobi(arg1, var_map, gurobi_model)[0] ^ expr_to_gurobi(arg2, var_map, gurobi_model)[0]
    
    ## EQUIVALENT EXPR
    elif isinstance(expr, pyo_expr.logical_expr.EquivalenceExpression):
        arg1, arg2 = expr.args
        return expr_to_gurobi(arg1, var_map, gurobi_model)[0] == expr_to_gurobi(arg2, var_map, gurobi_model)[0]
    
    ## AND EXPR
    elif isinstance(expr, pyo_expr.logical_expr.AndExpression):
        return all(expr_to_gurobi(arg, var_map, gurobi_model)[0] for arg in expr.args)
    
    ## OR EXPR
    elif isinstance(expr, pyo_expr.logical_expr.OrExpression):
        return any(expr_to_gurobi(arg, var_map, gurobi_model)[0] for arg in expr.args)
    
    ## NUMERIC VALUE
    elif isinstance(expr, pyo_expr.numeric_expr.NumericValue):
        return expr
    
    ## BOOLEAN VALUE
    elif isinstance(expr, pyo_expr.logical_expr.BooleanValue):
        return expr
    
    else:
        raise ValueError(f"Unsupported expression type: {type(expr)}")

# Test toy transformer

from pyomo import dae
import numpy as np
from transformer import *
import extract_from_pretrained as extract_from_pretrained
from toy_problem import *
from toy_problem_setup import *
from omlt import OmltBlock
from omlt.neuralnet import NetworkDefinition, ReluBigMFormulation
from omlt.io.keras import keras_reader
import omlt
import OMLT_helper

gurobi_model = convert_pyomo_to_gurobipy(model)
gurobi_model.optimize()

if gurobi_model.status == GRB.OPTIMAL:
    # for v in gurobi_model.getVars():
    #     print(f'{v.varName}: {v.x}')
    print(f'Objective: {gurobi_model.objVal}')

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
