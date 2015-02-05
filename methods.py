import math

import sympy as sp
import numpy as np

#######################
# Statistical methods #
#######################
def search_basic(sym_expr, sym_params, sym_values,
                 base_values, base_estimates, precision=10):
    '''Search estimates by solving system of equations.
    Parameters:
        sym_expr --- sympy object, represents equation,
    for example, 'a * exp(-alpha*x) - y',
    which is should be equal to 0

        sym_params --- tuple of sympy objects, whose estimates we should find,
    for example, (a, alpha)

        sym_values --- tuple of sympy objects, whose values we have,
    for example, (x, y)

        base_values --- tuple of list of values, in same order,
    as sym_values, for example, ([x1, x2, ...], [y1, y2, ...]),
    used to compute estimate

        base_estimates --- tuple of start values of estimates, used in
    iterational search of estimates, in same order, as sym_params

        precision --- precision of numeric solution, specified by
    number of digits after dot

    Return:
        Estimates of specified symbolic variables
    '''        

    # vector function
    f = []

    # set precision
    sp.mpmath.mp.dps = precision
    
    # substitute values into sym_expr
    for i_param, sym_param in enumerate(sym_params):
        subs = {}
        for i_val, sym_val in enumerate(sym_values):
            subs[sym_val] = base_values[i_val][i_param]

        f.append(sym_expr.subs(subs))

    return sp.nsolve(f, sym_params, (10, 0))

def search_mnk(sym_expr, sym_params,
               sym_values, values,
               sym_res_value, res_values,
               init_estimates, num_iter=1):
    '''Search estimates of ditribution parameters with MNK method

    Parameters:
        sym_expr --- sympy object, which represents equation,
    for example, 'a * exp(-alpha*x)'

        sym_params --- tuple of sympy objects, whose estimates we 
    should find, for example, (a, alpha)

        sym_values --- tuple of sympy objects, whose values we have,
    for example, (x)

        values --- tuple of lists of values in same order, 
    as sym_values, for example, ([x1, x2, ...], [t1, t2, ...]),
    used to compute estimate

        sym_res_value --- tuple of sympy objects, whose values we
    have as a result of sym_expr, for example, y

        res_values --- list of values in same order, 
    as sym_res_values, for example, [y1, y2, ...], used to compute
    estimate

        init_estimates --- tuple of init values of estimates, used in
    iterational search of estimates, in same order, as sym_params

        num_iter --- number of method iterations

    Yield values:
        cur_num_iter, estimates --- range of tuples contains target
    estimates by number of iteration
    '''        

    res_values_t = res_values.T
    
    # init effective estimates with basic values
    cur_vals = np.array(init_estimates)
    
    for i in range(num_iter):
        # substitute current parameter values into sym_expr
        subs = {}
        for i_param, sym_param in enumerate(sym_params):
            subs[sym_param] = cur_vals[i_param]

        cur_f = sp.lambdify(sym_values, sym_expr.subs(subs))
        
        cur_appr_t = np.vectorize(cur_f)(values).T
        
        # compute derivates of sym_expr by sym_params
        diff_funcs = []
        for sym_param in sym_params:
            diff_func = sp.diff(sym_expr, sym_param).subs(subs)

            diff_funcs.append(sp.lambdify(sym_values, diff_func))

        # construct Q from rows
        q_rows = []
            
        for diff_func in diff_funcs:
            q_rows.append(
                np.vectorize(diff_func)(values)
            )
            
        Q_t = np.vstack(q_rows)
        Q = Q_t.T
                
        # calculate addition = ((Q_t*Q)^-1)*Q_t*(y_t - appr_f_t)
        add = np.linalg.inv(np.dot(Q_t, Q))
        add = np.dot(add, Q_t)
        add = np.dot(add, res_values_t - cur_appr_t)

        cur_vals += add

        yield i+1, cur_vals


def search_taylor(sym_expr, sym_params, sym_values, values, err_stds):
    '''Search estimates by Taylor method.
    Parameters:
        sym_expr --- sympy object, represents equation,
    for example, 'a * exp(-alpha*x) - y',
    which is should be equal to 0

        sym_params --- tuple of sympy objects, whose estimates we should find,
    for example, (a, alpha)

        sym_values --- tuple of sympy objects, whose values we have,
    for example, (x, y)

        values --- tuple of list of values, in same order,
    as sym_values, for example, ([x1, x2, ...], [y1, y2, ...]),
    used to compute estimate

        err_stds --- tuple of standart error deviations of values,
    in same order, as sym_values

    Return:
        Estimates of specified symbolic variables
    '''        
    def gen_subs(sym_values, f_subs, values):
        '''Generate dicts with substitutions of f_subs symbolic
        variables by corresponding values, with equal distance
        between them. 
        '''
        # determine distance between neighbour values,
        # for example x0 and x1
        dist = len(values[0]) // len(f_subs)

        # result substitution
        res = {}
        for val_i in range(dist):
            for f_subs_i, f_sub in enumerate(f_subs):
                for sym_i, sym_value in enumerate(sym_values):
                    # param to substitute
                    f_subs_param = f_sub[sym_value]
                    # index of current value
                    cur_val_i = val_i + dist*f_subs_i
                    res[f_subs_param] = values[sym_i][cur_val_i]
            yield res
    
    # number of symbolic parameters
    num_params = len(sym_params)
    
    err_cnt = 0

    # vector function
    f = []

    # vector function substitutions
    f_subs = []

    # generate system of equations
    for i in range(num_params):
        # generate substitutions: x -> x1, y -> y1, ...
        # for each equation
        subs = {}
        for sym_value in sym_values:
            subs[sym_value] = sp.Symbol(
                str(sym_value) + str(i)
            )
        f.append(sym_expr.subs(subs))
        f_subs.append(subs)

    # symbolic expressions of parameters
    sym_expr_params = sp.solve(f, sym_params)[0] # get first solution

    # matrix of symbolic derivatives
    sym_G = []

    # compute derivarives of parameter expressions,
    # and place the in matrix, for example:
    # G = [
    #       [ diff(expr1, x0), diff(expr1, y0), diff(expr1, x1), diff(expr1, y1)],
    #       [ diff(expr2, x0), diff(expr2, y0), diff(expr2, x1), diff(expr2, y1)],
    #     ], where x0, x1, y0, y1 are sym_values
    for sym_expr_param in sym_expr_params:
        g_row = []
        for f_sub in f_subs:
            for sym_value in sym_values:
                g_row.append(sp.diff(sym_expr_param, f_sub[sym_value]))
        sym_G.append(g_row)

    # create diagonal matrix of error dispersions
    err_disps = []
    for _ in sym_params:
        for err_std in err_stds:
            err_disps.append(err_std ** 2)
            
    R_err = np.diagflat(err_disps)

    # error counter
    err_cnt = 0

    # accumulation matrix with summary of R
    sum_R_inv = np.zeros((num_params, num_params))

    # accumulation matrix with summary of R_inv * theta
    sum_R_inv_theta = np.zeros(num_params)

    # replace symbolic values of f_subs with real values
    for val_subs in gen_subs(sym_values, f_subs, values):
        try:
            G = []
            for sym_g_row in sym_G:
                g_row = []
                for sym_diff in sym_g_row:
                    g_row.append(sym_diff.subs(val_subs))
                G.append(g_row)
            G = np.array(G)
            R = np.dot(np.dot(G, R_err), G.T)

            R_inv = np.linalg.inv(R)
            # print(R_inv)
            
            sum_R_inv = sum_R_inv + R_inv

            # substitute values into sym_expr_param to get theta
            theta = []
            for sym_expr_param in sym_expr_params:
                theta_row = [sym_expr_param.subs(val_subs)]
                theta.append(theta_row)

            theta = np.array(theta)

            R_inv_theta = np.dot(R_inv, theta)
            sum_R_inv_theta = sum_R_inv_theta + R_inv_theta

            # print(R_inv_theta)
            # print()
        except (ValueError, np.linalg.linalg.LinAlgError): # singular matrix, skip it
            err_cnt += 1
            continue

    res_matrix = np.dot(np.linalg.inv(sum_R_inv), sum_R_inv_theta)

    # extract estimates from matrix
    res = []
    for row in res_matrix:
        res.append(row[0])
        
    return err_cnt, res
