import pyomo.environ as pyo
import pyomo.core.expr as pyo_expr
import pyomo.core.base as pyo_base
from pyomo import dae
from gurobipy import Model, GRB
import numpy as np
import omlt
import re

"""
This file contains helper functions to parse a Pyomo model and convert it to a Gurobi model.
"""

def to_gurobi(pyomo_model, func_nonlinear=1):
    """
    Converts a Pyomo optimization model to a Gurobi optimization model.

    Args:
        pyomo_model (pyomo.ConcreteModel): 
            A Pyomo model containing variables, constraints, and an objective function.
        func_nonlinear (int, optional): 
            Specifies how nonlinear functions (like exp()) are handled. Defaults to 1.
            - 0: Piecewise linear approximation.
            - 1: Outer approximation of the function.
            - -1: Controlled by Gurobi's `FuncNonlinear` parameter.

    Returns:
        tuple: 
            - gurobi_model (gurobipy.Model): A Gurobi model equivalent to the input Pyomo model.
            - var_map (dict): A mapping from Pyomo variables to Gurobi variables.
            - block_map (dict): A mapping from Pyomo blocks to corresponding Gurobi representations.

    Raises:
        ValueError: If unsupported functions or expressions are encountered.

    Notes:
        - Handles Pyomo variables, parameters, and blocks, including OMLT blocks for neural networks.
        - Supports a wide range of Pyomo expression types, including:
            - LinearExpression
            - ProductExpression
            - SumExpression
            - UnaryFunctionExpression (exp, log, sqrt, abs)
            - MaxExpression, MinExpression, and more (See: expr_to_gurobi()).
    """
    
    # Create gurobi model with same name as pyomo model
    gurobi_model = Model(pyomo_model.name)
    gurobi_model.params.FuncNonlinear = func_nonlinear # 0: piecewise linear approx, 1: outer approx of function, -1: controlled by FuncNonLinear
    
    # Mapping of Pyomo variables to Gurobi variables
    var_map = {}
    block_map = {}
        
    # Convert Vars and Params
    for attr in dir(pyomo_model):
        var = getattr(pyomo_model, attr)
        
        if isinstance(var, (omlt.OmltBlock, pyo.Block)): # Check for tnn Block()
            
            if var.is_indexed():
                index_set = list(var.index_set().data())
                
                for index in index_set:
                    
                    for attr in dir(var[index]): # iterate over Block attributes
                        block_attr = getattr(var[index], attr)
                       
                        if isinstance(block_attr, (omlt.OmltBlock, pyo.Block, pyo.Var, pyo_base.var._VarData, pyo.Param)):
                            if "NN_Block" in str(attr):
                                
                                
                                # Handle OMLT NN Block layer
                                if "NN_Block.layer" in str(block_attr):
                                    
                                    if isinstance(block_attr, (omlt.OmltBlock, pyo.Block)):
                                        for attr2 in dir(block_attr): # iterate over Block attributes
                                            block_attr2 = getattr(block_attr, attr2)
                                            
                                            if isinstance(block_attr2, dict) and isinstance(list(block_attr2.keys())[0], int):
                                                for index, obj in block_attr2.items():    
                                                    var_map, block_map = convert_block(obj, var_map, gurobi_model, block_map)       
                            else:
                                var_map, block_map = convert_block(var[index], var_map, gurobi_model, block_map)    
            else:
                var_map, block_map = convert_block(var, var_map, gurobi_model, block_map) 
                         
        else:  
            # Map & convert model params and vars to gurobi vars  
            var_map, _ = create_gurobi_var(var, var_map, gurobi_model)

    gurobi_model.update()
    # Convert objective
    for obj in pyomo_model.component_objects(pyo.Objective, active=True):
        expr = obj.expr
        sense = GRB.MINIMIZE if obj.sense == pyo.minimize else GRB.MAXIMIZE
        expr, _= expr_to_gurobi(expr, var_map, gurobi_model)
        gurobi_model.setObjective(expr=expr, sense=sense)
        
    gurobi_model.update()
       
    # Convert constraints
    for con in pyomo_model.component_objects(pyo.Constraint, active=True): 
        for index in con:
            lhs, rhs = con[index].expr.args
            unary = False
            if isinstance(lhs, pyo_expr.numeric_expr.UnaryFunctionExpression):
                func_equ = lhs
                res = rhs
                unary = True
            elif isinstance(rhs, pyo_expr.numeric_expr.UnaryFunctionExpression):
                func_equ = rhs
                res = lhs
                unary = True
                
            if unary:
                func = func_equ.getname()
                arg = func_equ.args[0]
                result = var_map[res.name]
                arg_gurobi = expr_to_gurobi(arg, var_map, gurobi_model)[0]
                
                if func == 'exp':
                    gurobi_model.addGenConstrExp(arg_gurobi, result)
                    

                elif func == 'sqrt':
                    gurobi_model.addConstr(arg_gurobi * arg_gurobi == result)


                elif func == 'log': #natural log
                    gurobi_model.addGenConstrLog(arg_gurobi, result)
                    
                elif func == 'abs': 
                    gurobi_model.addGenConstrAbs(result, arg_gurobi )
                    
                else:
                    raise ValueError(f"Unsupported unary function: {func}")
                
            # OTHER FUNCTION
            else:     
                
                lhs_gurobi_expr, add_constraint = expr_to_gurobi(lhs, var_map, gurobi_model)
                rhs_gurobi_expr, add_constraint = expr_to_gurobi(rhs, var_map, gurobi_model)
                    
            if add_constraint :
                
                if con[index].equality:
                    
                    gurobi_model.addConstr(lhs_gurobi_expr == rhs_gurobi_expr)
                else:
                    if con[index].has_lb():
                        
                        gurobi_model.addConstr(lhs_gurobi_expr >= rhs_gurobi_expr)
                    if con[index].has_ub():
                        
                        gurobi_model.addConstr(lhs_gurobi_expr <= rhs_gurobi_expr)
    
    gurobi_model.update()
    return gurobi_model, var_map, block_map

def get_gurobi_vtype(pyomo_var):
    """
    Determines the Gurobi variable type corresponding to a Pyomo variable's domain.

    Args:
        pyomo_var (pyomo.Var): A Pyomo variable.

    Returns:
        gurobipy.GRB.VarType: The Gurobi variable type, one of:
            - GRB.CONTINUOUS
            - GRB.BINARY
            - GRB.INTEGER

    Examples:
        >>> get_gurobi_vtype(pyomo_model.x)
        GRB.CONTINUOUS
    """

    try:
        domain =  pyomo_var.domain
    except:
        domain = None
        
    if domain == pyo.NonNegativeReals:
        return GRB.CONTINUOUS
    elif domain == pyo.Binary:
        return GRB.BINARY
    elif domain == pyo.Integers or domain == pyo.NonNegativeIntegers:
        return GRB.INTEGER
    else:
        return GRB.CONTINUOUS  # Default to continuous if the domain is not specified or different

def convert_block(var, var_map, gurobi_model, block_map):
    """
    Converts a Pyomo block to its Gurobi equivalent and maps its variables. 
    The function recursively handles blocks with block attributes.

    Args:
        var (pyomo.Block): The Pyomo block to be converted.
        var_map (dict): Mapping from Pyomo variables to Gurobi variables.
        gurobi_model (gurobipy.Model): The Gurobi model where the block will be added.
        block_map (dict): Mapping from Pyomo blocks to corresponding Gurobi representations.

    Returns:
        tuple: Updated `var_map` and `block_map`.
    """

    for attr in dir(var): # iterate over Block attributes
        block_attr = getattr(var, attr)
        
        # Map & convert block params and vars to gurobi vars
        if isinstance(block_attr, (pyo.Var, pyo_base.var._VarData, pyo.Param)):
            var_map, block_map = create_gurobi_var(block_attr, var_map, gurobi_model, block_map) 
            
        # Check for sub-block
        elif isinstance(block_attr, (omlt.OmltBlock, pyo.Block)):
            var_map, block_map = convert_block(block_attr, var_map, gurobi_model, block_map)

    return var_map, block_map

def create_nested_dict(dict, var_name, gurobi_var, reg_expr="[\[\]]"):
    """
    Creates or updates a nested dictionary structure for mapping Pyomo variables to Gurobi variables.
    This is necessary to convert Pyomo's indexed variables to Gurobi variables.

    Args:
        dict (dict): The dictionary to update.
        var_name (str): Name of the Pyomo variable.
        gurobi_var (gurobipy.Var): The Gurobi variable corresponding to the Pyomo variable.
        reg_expr (str, optional): Regular expression to split variable names. Defaults to "[\\[\\]]".

    Returns:
        dict: Updated nested dictionary with the mapping.
    """
    name_split = re.split(reg_expr, var_name) # split name on [, ., ]
    name_split = [elem for elem in name_split if elem] # remove empties
    
    name = []
    for elem in name_split:
        if elem[0] == '.':
            name.append(elem[1:])
        else:
            name.append(elem)
    
    dict[(tuple(name))] = gurobi_var
    return dict
                    
def create_gurobi_var(var, var_map, gurobi_model, block_map = None):
    """
    Creates equivalent Pyomo variables or parameters for the Gurobi model and updates mappings.

    Args:
        var (Union[pyomo.Var, pyomo.Param]): A Pyomo variable or parameter to convert.
        var_map (dict): Mapping from Pyomo variables to Gurobi variables.
        gurobi_model (gurobipy.Model): The Gurobi model where variables are added.
        block_map (dict, optional): Mapping from Pyomo blocks to Gurobi representations. Defaults to None.

    Returns:
        tuple: Updated `var_map` and `block_map`.
    """

    
    # Variables
    if isinstance(var, (pyo.Var,pyo_base.var._VarData)):
        if var.is_indexed():
            index_set = list(var.index_set().data())
            vtype = get_gurobi_vtype(var[index_set[0]])
            
            gurobi_var = gurobi_model.addVars(index_set, name=str(var), vtype=vtype)
            
            # add bounds
            for index in index_set:
                pyomo_var = var[index]
                gurobi_var[index].lb = pyomo_var.lb if pyomo_var.lb is not None else -GRB.INFINITY
                gurobi_var[index].ub = pyomo_var.ub if pyomo_var.ub is not None else GRB.INFINITY

                var_map[pyomo_var.name] = gurobi_var[index] 
                if block_map is not None:
                    block_map = create_nested_dict(block_map, pyomo_var.name, gurobi_var[index])
        else:
            lb = var.lb if var.lb is not None else -GRB.INFINITY
            ub = var.ub if var.ub is not None else GRB.INFINITY
            vtype = get_gurobi_vtype(var)
            gurobi_var = gurobi_model.addVar(lb=lb, ub=ub, name=str(var), vtype=vtype)
            var_map[var.name] = gurobi_var
                
            if block_map is not None:
                block_map = create_nested_dict(block_map, var.name, gurobi_var)
        
    # Parameters   
    elif isinstance(var, pyo.Param):
        if var.is_indexed():
            index_set = list(var.index_set().data())
            try:
                vtype = get_gurobi_vtype(var[index_set[0]])
            except:
                print(var, index_set )
            gurobi_var = gurobi_model.addVars(index_set, name=str(var), vtype=vtype)
            var_map[var.name] = gurobi_var
            
            if block_map is not None:
                block_map = create_nested_dict(block_map, var.name, gurobi_var)
            
            # add bounds
            for index in index_set:
                pyomo_var = var[index]
                
                
                if isinstance(pyomo_var, (int, float, np.int32, np.int64,  np.float32, np.float64)):
                    gurobi_var[index].lb = pyomo_var
                    gurobi_var[index].ub = pyomo_var
                    
                     
                    var_map[var.name+str([index])] = gurobi_var[index]
                    
                    
                    if block_map is not None:
                        block_map = create_nested_dict(block_map, var.name+str([index]), gurobi_var[index])
                    
                elif isinstance(pyomo_var, pyo_base.param._ParamData):
                    gurobi_var[index].lb = pyomo_var.value
                    gurobi_var[index].ub = pyomo_var.value
                    var_map[pyomo_var] = gurobi_var[index] 
                    
                    if block_map is not None:
                        block_map = create_nested_dict(block_map, pyomo_var.name, gurobi_var[index])
                        
                else:
                    gurobi_var[index].lb = pyomo_var
                    gurobi_var[index].ub = pyomo_var
                    var_map[str(var)+str(pyomo_var)] = gurobi_var[index] 
                    if block_map is not None:
                        block_map = create_nested_dict(block_map, pyomo_var.name, gurobi_var[index])
        else:
            vtype = get_gurobi_vtype(var)
            gurobi_var = gurobi_model.addVar(name=str(var), vtype=vtype)
            gurobi_var.lb = var.value
            gurobi_var.ub = var.value
            var_map[var.name] = gurobi_var
            if block_map is not None:
                        block_map = create_nested_dict(block_map, var.name, gurobi_var[index])
            
    return var_map, block_map
                
def expr_to_gurobi(expr, var_map, gurobi_model):
    """
    Recursively converts a Pyomo expression into its Gurobi equivalent.

    Args:
        expr (pyomo.Expression): The Pyomo expression to convert.
        var_map (dict): Mapping from Pyomo variables to Gurobi variables.
        gurobi_model (gurobipy.Model): The Gurobi model where the expression will be added.

    Returns:
        tuple: 
            - Gurobi expression equivalent to the Pyomo expression.
            - bool: Indicator of successful conversion.

    Raises:
        ValueError: If the expression type is unsupported.
        
    Notes:
        - Supports a wide range of Pyomo expression types, including:
            - LinearExpression
            - ProductExpression
            - PowExpression
            - SumExpression
            - MaxExpression, MinExpression, and more.
    """
    
    ## INT
    if isinstance(expr, (int, np.int32, np.int64 )):
        return expr, True
    
    ## FLOAT
    elif isinstance(expr, (float, np.float32, np.float64)):
        return expr, True
    
    ## PARAMETER
    if isinstance(expr, (pyo.Param, pyo_base.param._ParamData)):
        gurobi_var = var_map[expr.name]
        return  gurobi_var, True
    
    ## VARIABLE
    elif isinstance(expr, (pyo.Var,pyo_base.var._VarData)):
        
        gurobi_var = var_map[expr.name]
            
        return gurobi_var , True
    
    ## LINEAR EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.LinearExpression):
        gurobi_expr = 0.0
        #print("lin expr: ", expr)
        for sub_expr in expr.args:
            sub_gurobi_expr, _ = expr_to_gurobi(sub_expr, var_map, gurobi_model)
            gurobi_expr += sub_gurobi_expr

        return gurobi_expr, True
    
    ## PRODUCT EXPR
    elif isinstance(expr, pyo_expr.numeric_expr.ProductExpression):
        gurobi_expr = 1.0
        for sub_expr in expr.args:
            sub_gurobi_expr, _ = expr_to_gurobi(sub_expr, var_map, gurobi_model)
            gurobi_expr *= (sub_gurobi_expr)
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
    
    ## DAE.INTEGRAL EXPR
    elif isinstance(expr, dae.Integral):
        #print("intefgral",expr, expr.getname(), expr.args)
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


# gurobi_model = to_gurobi(pyomo_model)
# gurobi_model.optimize()

# if gurobi_model.status == GRB.OPTIMAL:
#     optimal_parameters = {}
#     for v in gurobi_model.getVars():
#         #print(f'{v.varName}: {v.x}')
#         optimal_parameters[v.varName] = v.x
#     #print(f'Objective: {gurobi_model.objVal}')
    
#     print(optimal_parameters)
