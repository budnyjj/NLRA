import math

import sympy as sp
import numpy as np

###########
# Helpers #
###########

# Caches results of symbolical equation solving
__CACHE_SOLUTIONS = {}
# Caches result of symbolic parameter substitutions
__CACHE_SUBSTITUTIONS = {}

def __solve(f, parameters):
    """Return result of symbolic equation solving.

    Parameters:
        f --- tuple of sympy equations to solve

        parameters --- tuple of sympy parameters of equation

    Return:
        solution --- first sympy solution of equation

    """
    key_solution = (f, parameters)
    solution = None
    try:
        solution = __CACHE_SOLUTIONS[key_solution]
    except KeyError:
        solution = sp.solve(f, parameters, dict=True)
        __CACHE_SOLUTIONS[key_solution] = solution
    if type(solution) is list:
        # get only first solution
        solution = solution[0]
    return solution

def __subs(sym_f, subs):
    """
    Computes values of function by value substitution.

    Parameters:
        sym_f --- symbolic function to evaluate, for example:
        (y0 - y1)/(x0 - x1)

        subs --- dict of symbolic/numeric substitutions, for example:
        {x0: 8.9882534228582163, y0: 8.9868654253346634, x1: 9.3128376287378849, y1: 9.314557492970053}

    Return:
        val_f --- function value as the result of numeric parameter substitutions
    """

    sym_subs, val_subs = tuple(subs.keys()), subs.values()
    key_sub = (sym_f, sym_subs)
    f = None
    try:
        f = __CACHE_SUBSTITUTIONS[key_sub]
    except KeyError:
        f = sp.lambdify(sym_subs, sym_f, modules='numpy')
        __CACHE_SUBSTITUTIONS[key_sub] = f
    val_f = f(*val_subs)
    return val_f

def __gen_equations(delta_expression, values, num_params):
    """Return system of equations from delta_expression for every parameter and
    list of symbolic substitutions for each equation.

    Parameters:
        delta_expression --- sympy object, which represents equation,
    for example, 'a * exp(-alpha*x) - y', which is should be equal to 0

        values --- dict of values, keyed by sympy objects
    for example, {x: [x1, x2, ...], y: [y1, y2, ...])

        num_params --- number of target parameters in delta_expression

    Return:
        f --- list of equations, generated from delta expression

        f_subs --- list of dicts with of indexed symbolical_values,
    keyed by these symbolical values, for example:
    [{x: x1, y: y1}, {x: x2, y: y2}]

    """
    # vector function
    f = []

    # vector function substitutions
    f_subs = []

    # generate system of equations
    for param_i in range(num_params):
        # generate substitutions: x -> x1, y -> y1, ...
        # for each equation
        subs = {}
        for sym_value in values:
            subs[sym_value] = sp.Symbol('{}{}'.format(sym_value, param_i))
        f.append(delta_expression.subs(subs))
        f_subs.append(subs)

    return tuple(f), tuple(f_subs)

def __gen_subs(f_subs, values):
    """Generate dicts of substitutions of f_subs symbolic variables by
    corresponding values, with equal distance between them.

    Parameters:
        f_subs --- list of dicts with of indexed symbolical_values,
    keyed by these symbolical values, for example:
    [{x: x1, y: y1}, {x: x2, y: y2}]

        values --- dict of values, keyed by sympy objects
    for example, {x: [x1, x2, ...], y: [y1, y2, ...])

    Yield:
        res --- dict of substitutions, for example:
    {x1: 12, x2: 21, y1: 43, y2: 34}

    """

    # get sample parameter as first key of dict
    s_p = next(iter(values.keys()))

    # determine distance between neighbour values,
    # for example x0 and x1
    distance = len(values[s_p]) // len(f_subs)

    # result substitution
    res = {}
    for val_i in range(distance):
        for f_subs_i, f_sub in enumerate(f_subs):
            for sym_value in values:
                # param to substitute
                f_subs_param = f_sub[sym_value]

                # index of current value
                cur_val_i = val_i + f_subs_i * distance
                res[f_subs_param] = values[sym_value][cur_val_i]
        yield res


#######################
# Statistical methods #
#######################
def search_basic(delta_expression, parameters, values):
    """Search estimates by solving system of equations.

    Parameters:
        delta_expression --- sympy object, which represents equation,
    for example, 'a * exp(-alpha*x) - y', which is should be equal to 0

        parameters --- list of sympy objects, whose estimates we should find,
    for example, (a, alpha)

        values --- dict of values, keyed by sympy objects
    for example, {x: [x1, x2, ...], y: [y1, y2, ...])

    Return:
        estimates --- numpy array of estimates of specified symbolic variables
    """

    # number of parameters
    num_params = len(parameters)

    f, f_subs = __gen_equations(delta_expression, values, num_params)

    # symbolic expressions of parameters
    sym_expr_params = __solve(f, parameters)

    estimates = np.zeros(num_params)

    # executes only once
    for subs in __gen_subs(f_subs, values):
        for i_param, sym_param in enumerate(parameters):
            estimates[i_param] = __subs(sym_expr_params[sym_param], subs)

    return estimates

def search_lse2(expression, parameters, values,
                result_values, init_estimates,
                err_stds=None, num_iter=1):
    """Search estimates of ditribution parameters with LSE method.

    Parameters:
        expression --- sympy object, which represents target expression,
    for example, 'a * exp(-alpha*x)'

        parameters --- list of sympy objects, whose estimates we
    should find, for example, (a, alpha)

        values --- dict of values, keyed by sympy objects
    for example, {x: [x1, x2, ...]}

        result_values --- dict of result values, keyed by sympy objects
    for example, {y: [y1, y2, ...]}

        init_estimates --- dict of init values of estimates, used in
    iterational search of estimates, keyed by sympy objects
    for example, {x: 0, y: 0}

        err_stds --- dict of standart error deviations of values,
    keyed by sympy objects, for example:
    {x: 0.2, y: 0.2}

        num_iter --- number of method iterations

    Yield:
        cur_num_iter, cur_estimates --- list of target
    estimates by number of iteration

    """

    # get list of symbolic values
    sym_vals = tuple(values.keys())

    # get list of symbolic result values
    sym_res_vals = tuple(result_values.keys())

    # get array of real values
    vals = []
    for sym_val in sym_vals:
        vals.append(values[sym_val])
    vals = np.array(vals).T

    # get result values as first value of dict
    res_vals = [next(iter(result_values.values()))]
    res_vals = np.array(res_vals).T

    # init effective estimates with basic values
    cur_estimates = []
    for parameter in parameters:
        cur_estimates.append(init_estimates[parameter])
    cur_estimates = np.array([cur_estimates]).T

    # get matrix of symbolic derivatives of expression
    sym_diff_funcs = []
    for parameter in parameters:
        sym_diff_funcs.append(sp.diff(expression, parameter))

    for i in range(num_iter):
        # substitute current parameter values into sym_expr
        subs = {}
        for i_param, sym_param in enumerate(parameters):
            subs[sym_param] = cur_estimates[i_param]

        cur_f = sp.lambdify(sym_vals, expression.subs(subs), 'numpy')

        cur_appr = np.vectorize(cur_f)(vals)

        # compute derivates of sym_expr by sym_params
        diff_funcs = []
        for param_i, parameter in enumerate(parameters):
            diff_func = sym_diff_funcs[param_i].subs(subs)
            diff_funcs.append(sp.lambdify(sym_vals, diff_func, 'numpy'))

        # construct Q from rows
        q_rows = []

        for diff_func in diff_funcs:
            q_rows.append(
                np.vectorize(diff_func)(vals)
            )

        Q = np.hstack(q_rows)
        Q_t = Q.T

        # calculate addition = ((Q_t*Q)^-1)*Q_t*(res_vals - cur_appr)
        add = np.linalg.inv(np.dot(Q_t, Q))
        add = np.dot(add, Q_t)
        add = np.dot(add, res_vals - cur_appr)

        cur_estimates += add

        # yield first row
        yield i + 1, cur_estimates.T[0]

def search_lse(*args, **kwargs):
    estimates = None
    for _, it_estimates in search_lse2(*args, **kwargs):
        estimates = it_estimates
    return estimates

def search_mrt(delta_expression, parameters, values, err_stds):
    """Search estimates by Taylor method.

    Parameters:
        delta_expression --- sympy object, which represents equation,
    for example, 'a * exp(-alpha*x) - y', which is should be equal to 0

        parameters --- list of sympy objects, whose estimates we should find,
    for example, (a, alpha)

        values --- dict of values, keyed by sympy objects,
    for example, {x: [x1, x2, ...], y: [y1, y2, ...])

        err_stds --- dict of standart error deviations of values,
    keyed by sympy objects, for example:
    {x: 0.2, y: 0.2}

    Return:
        res -- list of estimates of specified symbolic variables

    """

    # number of symbolic parameters
    num_params = int(len(parameters))
    # get list of symbolic values
    sym_vals = tuple(values.keys())
    # number of symbolic values
    num_sym_vals = int(len(sym_vals))

    f, f_subs = __gen_equations(delta_expression, values, num_params)

    # symbolic expressions of parameters
    sym_expr_params = __solve(f, parameters)

    # matrix of symbolic derivatives
    sym_G = []
    # compute derivarives of parameter expressions,
    # and place the in matrix, for example:
    # G = [
    #       [ diff(expr1, x0), diff(expr1, y0), diff(expr1, x1), diff(expr1, y1)],
    #       [ diff(expr2, x0), diff(expr2, y0), diff(expr2, x1), diff(expr2, y1)],
    #     ], where x0, x1, y0, y1 are sym_values
    for parameter in parameters:
        g_row = []
        for f_sub in f_subs:
            for sym_val in sym_vals:
                g_row.append(
                    sp.diff(sym_expr_params[parameter], f_sub[sym_val]))
        sym_G.append(g_row)

    # create diagonal matrix of error dispersions
    err_disps = np.zeros(num_params * num_sym_vals)
    err_disp_i = 0
    for _ in parameters:
        for sym_val in sym_vals:
            err_disps[err_disp_i] = err_stds[sym_val] ** 2
            err_disp_i += 1
    R_err = np.diagflat(err_disps)

    # accumulation matrix with summary of R
    sum_R_inv = np.zeros((num_params, num_params))
    # accumulation matrix with summary of R_inv * theta
    sum_R_inv_theta = np.zeros(num_params)

    # replace symbolic values of f_subs with real values
    for val_subs in __gen_subs(f_subs, values):
        # TODO: optimize
        G = np.zeros((len(sym_G), len(sym_G[0])))
        for i, sym_g_row in enumerate(sym_G):
            for j, sym_diff in enumerate(sym_g_row):
                G[i, j] = __subs(sym_diff, val_subs)

        R = np.dot(np.dot(G, R_err), G.T)

        R_inv = np.linalg.inv(R)

        sum_R_inv += R_inv

        # substitute values into sym_expr_param to get theta
        theta = np.zeros((num_params, 1))
        for (i_param, param) in enumerate(parameters):
            theta[i_param, 0] = __subs(sym_expr_params[param], val_subs)

        R_inv_theta = np.dot(R_inv, theta)
        sum_R_inv_theta = sum_R_inv_theta + R_inv_theta

    res_matrix = np.dot(np.linalg.inv(sum_R_inv), sum_R_inv_theta)
    estimates = res_matrix[:,0]
    return estimates
