#!/usr/bin/env python

import math
import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import stats.methods as methods
from stats.utils import *

##########################
# 3-parametric functions #
##########################

SYM_X, SYM_Y = SYM_VALUES = sp.symbols('x y')
SYM_A, SYM_B, SYM_C = SYM_PARAMS = sp.symbols('a b c')

# quadratic function
SYM_EXPR = sp.sympify('a*(x**2) + b*x + c')
SYM_EXPR_DELTA = sp.sympify('y - (a*(x**2) + b*x + c)')

MIN_X = -10
MAX_X = 20
NUM_VALS = 30              # number of source values

REAL_A = 0.2               # real 'a' value of source distribution
REAL_B = -2                # real 'b' value of source distiribution
REAL_C = 5                 # real 'c' value of source distiribution

ERR_X_AVG = 0              # average of X error values
ERR_X_STD = 0              # std of X error values

ERR_Y_AVG = 0              # average of Y error values
ERR_Y_STD = 5              # std of Y error values

NUM_ITER = 10              # number of realizations

# real X values without errors
real_x = np.linspace(MIN_X, MAX_X, NUM_VALS, dtype=np.float)

# real Y values without errors
real_y = np.vectorize(
    sp.lambdify(
        SYM_X,
        SYM_EXPR.subs(
            {
                SYM_A: REAL_A,
                SYM_B: REAL_B,
                SYM_C: REAL_C
            }
        )
    )
)(real_x)

print('Expression:    {}'.format(SYM_EXPR))
print('Real A:        {}'.format(REAL_A))
print('Real B:        {}'.format(REAL_B))
print('Real C:        {}'.format(REAL_C))
print('Error X std:   {}'.format(ERR_X_STD))
print('Error Y std:   {}'.format(ERR_Y_STD))
print('Number of iterations: {}'.format(NUM_ITER))
print('-' * 40, '\n')

# iterate by error standart derivation values
for iter_i in range(NUM_ITER):
    print('Iteration #{}:'.format(iter_i))

    # add X errors with current normal distribution
    x = np.vectorize(
        lambda v: v + random.gauss(ERR_X_AVG, ERR_X_STD)
    )(real_x)

    third_len = len(x) / 3

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
    base_values_first = {
        SYM_X: [x[0], x[1], x[2]],
        SYM_Y: [y[0], y[1], x[2]]
    }

    # get base values as half-distant pairs of values
    base_values_dist = {
        SYM_X: [x[0], x[third_len], x[third_len * 2]],
        SYM_Y: [y[0], y[third_len], y[third_len * 2]]
    }

    # get base values as averages of two half-length subgroups
    base_values_avg = {
        SYM_X: [
            avg(x[:third_len]),
            avg(x[third_len:third_len * 2]),
            avg(x[third_len * 2:])
        ],
        SYM_Y: [
            avg(y[:third_len]),
            avg(y[third_len:third_len * 2]),
            avg(y[third_len * 2:])
        ]
    }

    ################
    # Basic search #
    ################

    # find params with basic method
    basic_a, basic_b, basic_c = methods.search_basic(
        delta_expression=SYM_EXPR_DELTA,
        parameters=(SYM_A, SYM_B, SYM_C),
        values=base_values_avg
    )

    basic_y = np.vectorize(
        sp.lambdify(
            SYM_X,
            SYM_EXPR.subs(
                {
                    SYM_A: basic_a,
                    SYM_B: basic_b,
                    SYM_C: basic_c
                }
            )
        )
    )(real_x)

    basic_disp = disp(basic_y, y)
    basic_std = std(basic_y, y)

    print('Basic a:       {}'.format(basic_a))
    print('Basic b:       {}'.format(basic_b))
    print('Basic c:       {}'.format(basic_c))
    print('Dispersion:    {}'.format(basic_disp))
    print('Std:           {}\n'.format(basic_std))

    plt.plot(x, basic_y,
             color='g', linestyle='-',
             marker='.', markersize=5, mfc='g')

    ##############
    # MNK search #
    ##############

    # use basic estimates as init estimates for MNK
    for i, (mnk_a, mnk_b, mnk_c) in methods.search_mnk(
            expression=SYM_EXPR,
            parameters=(SYM_A, SYM_B, SYM_C),
            values={SYM_X: x},
            result_values={SYM_Y: y},
            init_estimates={
                SYM_A: basic_a,
                SYM_B: basic_b,
                SYM_C: basic_c
            }
    ):
        mnk_y = np.vectorize(
            sp.lambdify(
                SYM_X,
                SYM_EXPR.subs(
                    {
                        SYM_A: mnk_a,
                        SYM_B: mnk_b,
                        SYM_C: mnk_c
                    }
                )
            )
        )(real_x)
        mnk_disp = disp(mnk_y, y)
        mnk_std = std(mnk_y, y)

        print('MNK({}) a:     {}'.format(i, mnk_a))
        print('MNK({}) b:     {}'.format(i, mnk_b))
        print('MNK({}) c:     {}'.format(i, mnk_c))
        print('Dispersion:    {}'.format(mnk_disp))
        print('Std:           {}\n'.format(mnk_std))

        plt.plot(x, mnk_y,
                 color='b', linestyle='-',
                 marker='.', markersize=5, mfc='b')

    #################
    # Taylor search #
    #################

    # find params with mrt method
    mrt_a, mrt_b, mrt_c = methods.search_mrt(
        delta_expression=SYM_EXPR_DELTA,
        parameters=(SYM_A, SYM_B, SYM_C),
        values={SYM_X: x, SYM_Y: y},
        err_stds={SYM_X: ERR_X_STD, SYM_Y: ERR_Y_STD}
    )

    mrt_y = np.vectorize(
        sp.lambdify(
            SYM_X,
            SYM_EXPR.subs(
                {
                    SYM_A: mrt_a,
                    SYM_B: mrt_b,
                    SYM_C: mrt_c
                }
            )
        )
    )(real_x)

    mrt_disp = disp(mrt_y, y)
    mrt_std = std(mrt_y, y)

    print('Taylor a:      {}'.format(mrt_a))
    print('Taylor b:      {}'.format(mrt_b))
    print('Taylor c:      {}'.format(mrt_c))
    print('Dispersion:    {}'.format(mrt_disp))
    print('Std:           {}'.format(mrt_std))

    plt.plot(x, mrt_y,
             color='r', linestyle='-',
             marker='.', markersize=5, mfc='r')

    print('-' * 40, '\n')

plt.xlabel('x')
plt.ylabel('y')

plt.grid(True)
plt.show()
