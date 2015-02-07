#!/usr/bin/env python

import math
import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import stats.methods as methods
from stats.utils import *

#############
# Functions #
#############
 
SYM_X, SYM_Y = SYM_VALUES = sp.symbols('x y')
SYM_A, SYM_ALPHA = SYM_PARAMS = sp.symbols('a alpha')
 
# SYM_EXPR = sp.sympify('a * exp(-alpha*x)')
# SYM_EXPR_DELTA = sp.sympify('y - a * exp(-alpha*x)')

# SYM_EXPR = sp.sympify('a / exp(-alpha*x)')
# SYM_EXPR_DELTA = sp.sympify('y - a / exp(-alpha*x)')
 
# linear function
# SYM_EXPR = sp.sympify('a + alpha*x')
# SYM_EXPR_DELTA = sp.sympify('y - a - alpha*x')
 
# quadratic function
# SYM_EXPR = sp.sympify('a*(x**2) + alpha*x')
# SYM_EXPR_DELTA = sp.sympify('y - a*(x**2) - alpha*x')
 
# logarithmic function
# SYM_EXPR = sp.sympify('a + alpha*log(x)')
# SYM_EXPR_DELTA = sp.sympify('y - a - alpha*log(x)')

# sinusoidal function
SYM_EXPR = sp.sympify('a + alpha*sin(x)')
SYM_EXPR_DELTA = sp.sympify('y - (a + alpha*sin(x))')

MIN_X = 1
MAX_X = 20    
NUM_VALS = 20              # number of source values
 
REAL_A = 2                # real 'a' value of source distribution
REAL_ALPHA = 0.5          # real 'alpha' value of source distiribution
 
ERR_X_AVG = 0              # average of X error values
ERR_X_STD = 0              # std of X error values
 
ERR_Y_AVG = 0              # average of Y error values
ERR_Y_STD = 0.2            # std of Y error values
 
NUM_ITER = 10              # number of realizations
 
# real X values without errors
real_x = np.linspace(MIN_X, MAX_X, NUM_VALS ,dtype=np.float)
 
# real Y values without errors
real_y = np.vectorize(
    sp.lambdify(
        SYM_X,
        SYM_EXPR.subs({SYM_A: REAL_A, SYM_ALPHA: REAL_ALPHA})
    )
)(real_x)
 
print('Error X std:   {}'.format(ERR_X_STD))
print('Error Y std:   {}\n'.format(ERR_Y_STD))
 
# iterate by error standart derivation values
for iter_i in range(NUM_ITER):
    # add X errors with current normal distribution
    x = np.vectorize(
        lambda v: v + random.gauss(ERR_X_AVG, ERR_X_STD)
    )(real_x)

    half_len = len(x) / 2
    
    # add Y errors with current normal distribution
    y = np.vectorize(
        lambda v: v + random.gauss(ERR_Y_AVG, ERR_Y_STD)
    )(real_y)

    # plot real values
    plt.plot(x, y,
             color='b', linestyle=' ',
             marker='.', markersize=10, mfc='r')

    ################################
    # Base values for basic search #
    ################################
    
    # get base values as first pairs of values
    base_values_first = (
        [x[0], x[1]],
        [y[0], y[1]]
    )

    # get base values as half-distant pairs of values
    base_values_dist = (
        [x[0], x[half_len]],
        [y[0], y[half_len]]
    )

    # get base values as averages of two half-length subgroups
    base_values_avg = (
        [avg(x[:half_len]), avg(x[half_len:])],
        [avg(y[:half_len]), avg(y[half_len:])]
    )

    ################
    # Basic search #
    ################

    # find params with basic method
    basic_a, basic_alpha = methods.search_basic(
        sym_expr=SYM_EXPR_DELTA,
        sym_params=(SYM_A, SYM_ALPHA),
        sym_values=(SYM_X, SYM_Y),
        base_values=base_values_avg
    )
    
    basic_y = np.vectorize(
        sp.lambdify(
            SYM_X,
            SYM_EXPR.subs({SYM_A: basic_a, SYM_ALPHA: basic_alpha})
        )
    )(real_x)
    
    basic_disp = disp(basic_y, y)
    basic_std = std(basic_y, y)

    print('Basic a:       {}'.format(basic_a))
    print('Basic alpha:   {}'.format(basic_alpha))
    print('Dispersion:    {}'.format(basic_disp))
    print('Std:           {}\n'.format(basic_std))

    plt.plot(x, basic_y,
             color='g', linestyle='-',
             marker='.', markersize=5, mfc='g')

    ##############
    # MNK search #
    ##############
    
    # use basic estimates as init estimates for MNK
    for i, (mnk_a, mnk_alpha) in methods.search_mnk(
            sym_expr=SYM_EXPR,
            sym_params=(SYM_A, SYM_ALPHA),
            sym_values=(SYM_X),
            values=(x),
            sym_res_value=SYM_Y,
            res_values=y,
            init_estimates=(basic_a, basic_alpha)
    ):
        mnk_y = np.vectorize(
            sp.lambdify(
                SYM_X,
                SYM_EXPR.subs({SYM_A: mnk_a,
                               SYM_ALPHA: mnk_alpha})
            )
        )(real_x)
        mnk_disp = disp(mnk_y, y)
        mnk_std = std(mnk_y, y)

        print('MNK({}) a:      {}'.format(i, mnk_a))
        print('MNK({}) alpha:  {}'.format(i, mnk_alpha))
        print('Dispersion:    {}'.format(mnk_disp))
        print('Std:           {}\n'.format(mnk_std))

        plt.plot(x, mnk_y,
                 color='b', linestyle='-',
                 marker='.', markersize=5, mfc='b')

    #################
    # Taylor search #
    #################

    # find params with taylor method
    taylor_a, taylor_alpha = methods.search_taylor(
        sym_expr=SYM_EXPR_DELTA,
        sym_params=(SYM_A, SYM_ALPHA),
        sym_values=(SYM_X, SYM_Y),
        values=(x, y),
        err_stds=(ERR_X_STD, ERR_Y_STD)
    )
 
    taylor_y = np.vectorize(
        sp.lambdify(
            SYM_X,
            SYM_EXPR.subs({SYM_A: taylor_a,
                           SYM_ALPHA: taylor_alpha})
        )
    )(real_x)
     
    taylor_disp = disp(taylor_y, y)
    taylor_std = std(taylor_y, y)
     
    print('Taylor a:      {}'.format(taylor_a))
    print('Taylor alpha:  {}'.format(taylor_alpha))
    print('Dispersion:    {}'.format(taylor_disp))
    print('Std:           {}\n'.format(taylor_std))
     
    plt.plot(x, taylor_y,
             color='r', linestyle='-',
             marker='.', markersize=5, mfc='r')
     
plt.xlabel('x')
plt.ylabel('y')

plt.grid(True)
plt.show()
