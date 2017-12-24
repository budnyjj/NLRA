import numpy as np

def with_errors(real_x, real_y, err_std_x, err_std_y):
    # add X errors with current normal distribution
    x = real_x + np.random.normal(0, err_std_x, real_x.shape)
    # add Y errors with current normal distribution
    y = real_y + np.random.normal(0, err_std_y, real_y.shape)
    return x, y

def precise(
        expression_vectorized,
        num_vals,
        min_x,
        max_x):
    """Provides specified number of values (x, y),
       linked with specified relation, using the following scheme:
        y = expression(x)

    Parameters:
        f_expression --- functional relation between free and dependent variable
        num_vals     --- number of values to generate
        min_x        --- minimal value of free variable
        max_x        --- maximal value of free variable
        err_std_x    --- standart deviation of errors applied to the free variable
        err_std_y    --- standart deviation of errors applied to the result values

    Returns:
         a pair of numpy arrays with measurements of the free and dependent variables.
    """
    x = np.linspace(min_x, max_x, num_vals)
    y = expression_vectorized(x)
    return x, y

def determined(
        expression_vectorized,
        num_vals,
        min_x,
        max_x,
        err_std_x = 0,
        err_std_y = 0):
    """Provides specified number of values (x, y),
       linked with specified relation, using the following scheme:
        x* in [min_x, max_x], x*_i - x*_{i-1} = (max_x - min_x) / num_vals
        y* = expression(x*)
        x = x* + gauss(err_std_x)
        y = y* + gauss(err_std_y)

    Parameters:
        expression_vectorized --- functional relation between
                                  free and dependent variable
        num_vals              --- number of values to generate
        min_x                 --- minimal value of free variable
        max_x                 --- maximal value of free variable
        err_std_x             --- standart deviation of errors
                                  applied to the free variable
        err_std_y             --- standart deviation of errors
                                  applied to the result values

    Returns:
         a pair of numpy arrays with measurements of the free and dependent variables.
    """
    # real X values without errors
    real_x, real_y = precise(expression_vectorized, num_vals, min_x, max_x)
    if err_std_x == 0 and err_std_y == 0:
        return real_x, real_y
    return with_errors(real_x, real_y, err_std_x, err_std_y)

def uniform(
        expression_vectorized,
        num_vals,
        min_x,
        max_x,
        err_std_x=0,
        err_std_y=0):
    """Provides specified number of values (x, y),
       linked with specified relation, using the following scheme:
        x* is a uniform value from [min_x, max_x)
        y* = expression(x*)
        x = x* + gauss(err_std_x)
        y = y* + gauss(err_std_y)

    Parameters:
        expression_vectorized --- functional relation between
                                  free and dependent variable
        num_vals              --- number of values to generate
        min_x                 --- minimal value of free variable
        max_x                 --- maximal value of free variable
        err_std_x             --- standart deviation of errors
                                  applied to the free variable
        err_std_y             --- standart deviation of errors
                                  applied to the result values

    Returns:
         a pair of numpy arrays with measurements of the free and
         dependent variables.
    """
    # real X values without errors
    real_x = np.random.uniform(min_x, max_x, num_vals)
    real_y = expression_vectorized(real_x)
    if err_std_x == 0 and err_std_y == 0:
        return real_x, real_y
    return with_errors(real_x, real_y, err_std_x, err_std_y)

def normal(
        expression_vectorized,
        num_vals,
        min_x,
        max_x,
        err_std_x=0,
        err_std_y=0):
    """Provides specified number of values (x, y),
       linked with specified relation, using the following scheme:
        x* is a normal value from [min_x, max_x)
        y* = expression(x*)
        x = x* + gauss(err_std_x)
        y = y* + gauss(err_std_y)

    Parameters:
        expression_vectorized --- functional relation between
                                  free and dependent variable
        num_vals              --- number of values to generate
        min_x                 --- minimal value of free variable
        max_x                 --- maximal value of free variable
        err_std_x             --- standart deviation of errors
                                  applied to the free variable
        err_std_y             --- standart deviation of errors
                                  applied to the result values
    Returns:
         a pair of numpy arrays with measurements of the free and dependent variables.
    """
    # real X values without errors
    avg_x = (min_x + max_x) / 2
    std_x = (max_x - min_x) / 6
    real_x = np.random.normal(avg_x, std_x, num_vals)
    real_y = expression_vectorized(real_x)
    if err_std_x == 0 and err_std_y == 0:
        return real_x, real_y
    return with_errors(real_x, real_y, err_std_x, err_std_y)
