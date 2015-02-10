#!/usr/bin/env python

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

SYM_EXPR = sp.sympify('a * exp(-alpha*x)')
SYM_EXPR_DELTA = sp.sympify('y - a * exp(-alpha*x)')

MIN_X = 0
MAX_X = 24
NUM_VALS = 20              # number of source values

REAL_A = 10                # real 'a' value of source distribution
REAL_ALPHA = 0.01          # real 'alpha' value of source distiribution

ERR_X_AVG = 0              # average of X error values
ERR_X_MIN_STD = 0.1        # minimal std of X error values
ERR_X_MAX_STD = 0.5        # maximal std of X error values

ERR_Y_AVG = 0              # average of Y error values
ERR_Y_MIN_STD = 0.1        # minimal std of Y error values
ERR_Y_MAX_STD = 0.5        # maximal std of Y error values

ERR_NUM_STD_ITER = 10      # number of stds iterations  

# real X values without errors
real_x = np.linspace(MIN_X, MAX_X, NUM_VALS ,dtype=np.float)

# real Y values without errors
real_y = np.vectorize(
    sp.lambdify(SYM_X,
                SYM_EXPR.subs({SYM_A: REAL_A, SYM_ALPHA: REAL_ALPHA})
    )
)(real_x)

# generate array of X error stds
err_x_stds = np.linspace(ERR_X_MIN_STD, ERR_X_MAX_STD, ERR_NUM_STD_ITER)

# generate array of Y error stds
err_y_stds = np.linspace(ERR_Y_MIN_STD, ERR_Y_MAX_STD, ERR_NUM_STD_ITER)
    
# collect dispersions of estimates
basic_stds = []
mnk_stds = []
taylor_stds = []

# iterate by error standart derivation values
for err_x_std, err_y_std in zip(err_x_stds, err_y_stds):
    print('Error X std:   {}'.format(err_x_std))
    print('Error Y std:   {}\n'.format(err_y_std))

    # add X errors with current normal distribution
    x = np.vectorize(
        lambda v: v + random.gauss(ERR_X_AVG, err_x_std)
    )(real_x)

    # add Y errors with current normal distribution
    y = np.vectorize(
        lambda v: v + random.gauss(ERR_Y_AVG, err_y_std)
    )(real_y)

    # get base values as first pairs of values
    base_values_first = {
        SYM_X: [x[0], x[1]],
        SYM_Y: [y[0], y[1]]
    }

    half_len = len(x) / 2
    
    # get base values as half-distant pairs of values
    base_values_dist = {
        SYM_X: [x[0], x[half_len]],
        SYM_Y: [y[0], y[half_len]]
    }

    # get base values as averages of two half-length subgroups
    base_values_avg = {
        SYM_X: [avg(x[:half_len]), avg(x[half_len:])],
        SYM_Y: [avg(y[:half_len]), avg(y[half_len:])]
    }
    
    ################
    # Basic search #
    ################
    
    # find params with basic method
    basic_a, basic_alpha = methods.search_basic(
        delta_expression=SYM_EXPR_DELTA,
        parameters=(SYM_A, SYM_ALPHA),
        values=base_values_first)
    
    basic_y = np.vectorize(
        sp.lambdify(SYM_X,
                    SYM_EXPR.subs({SYM_A: basic_a, SYM_ALPHA: basic_alpha})
                )
    )(real_x)
    
    basic_disp = disp(basic_y, y)
    basic_std = std(basic_y, y)
    basic_stds.append(basic_std)
    
    print('Basic a:       {}'.format(basic_a))
    print('Basic alpha:   {}'.format(basic_alpha))
    print('Dispersion:    {}'.format(basic_disp))
    print('Std:           {}\n'.format(basic_std))
    
    ##############
    # MNK search #
    ##############
    for i, (mnk_a, mnk_alpha) in methods.search_mnk(
            expression=SYM_EXPR,
            parameters=(SYM_A, SYM_ALPHA),
            values={SYM_X: x},
            result_values={SYM_Y: y},
            init_estimates={SYM_A: basic_a, SYM_ALPHA: basic_alpha}
    ):
        mnk_y = np.vectorize(
            sp.lambdify(SYM_X,
                        SYM_EXPR.subs({SYM_A: mnk_a,
                                       SYM_ALPHA: mnk_alpha})
                    )
        )(real_x)
        mnk_disp = disp(mnk_y, y)
        mnk_std = std(mnk_y, y)
        mnk_stds.append(mnk_std) # only if one iteration

        print('MNK({}) a:      {}'.format(i, mnk_a))
        print('MNK({}) alpha:  {}'.format(i, mnk_alpha))
        print('Dispersion:    {}'.format(mnk_disp))
        print('Std:           {}\n'.format(mnk_std))
    
    # find params with taylor method
    taylor_a, taylor_alpha = methods.search_taylor(
        delta_expression=SYM_EXPR_DELTA,
        parameters=(SYM_A, SYM_ALPHA),
        values={SYM_X: x, SYM_Y: y},
        err_stds={SYM_X: err_x_std, SYM_Y: err_y_std}
    )
    
    taylor_y = np.vectorize(
            sp.lambdify(SYM_X,
                        SYM_EXPR.subs({SYM_A: taylor_a,
                                       SYM_ALPHA: taylor_alpha})
                    )
    )(real_x)
    taylor_disp = disp(taylor_y, y)
    taylor_std = std(taylor_y, y)
    taylor_stds.append(taylor_std)

    print('Taylor a:       {}'.format(taylor_a))
    print('Taylor alpha:   {}'.format(taylor_alpha))
    print('Dispersion:    {}'.format(taylor_disp))
    print('Std:           {}\n'.format(taylor_std))
    
plt.plot(err_x_stds, basic_stds,
         color='g', linestyle='-',
         marker='.', markersize=5, mfc='g')
plt.plot(err_x_stds, mnk_stds,
         color='b', linestyle='-',
         marker='.', markersize=5, mfc='b')
plt.plot(err_x_stds, taylor_stds,
         color='r', linestyle='-',
         marker='.', markersize=5, mfc='r')
    
plt.xlabel('D(Y)')
plt.ylabel('ro')

plt.grid(True)
plt.show()
