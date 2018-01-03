#!/usr/bin/env python

import os.path
import argparse
import math
import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)

import stats.methods as methods
import stats.utils as utils

################
# Declarations #
################

SYM_PARAMS = sp.symbols('a b c')
PRECISE_PARAMS = (0, 0.07, 0.01)
SYM_X, SYM_Y = sp.symbols('x y')

# SYM_EXPR = sp.sympify('a * exp(-b*x)')
# SYM_EXPR_DELTA = sp.sympify('y - a * exp(-b*x)')

# SYM_EXPR = sp.sympify('a * exp(b*x)')
# SYM_EXPR_DELTA = sp.sympify('y - a * exp(b*x)')

# linear function
# SYM_EXPR = sp.sympify('a + b*x')
# SYM_EXPR_DELTA = sp.sympify('y - a - b*x')

# quadratic function
# SYM_EXPR = sp.sympify('a + b*x + c*(x**2)')
# SYM_EXPR_DELTA = sp.sympify('y - (a + b*x + c*(x**2))')

# inverse function
SYM_EXPR = sp.sympify('a + 1/(b + c*x)')
SYM_EXPR_DELTA = sp.sympify('y - (a + 1/(b + c*x))')

# logarithmic function
# SYM_EXPR = sp.sympify('a + b*log(x)')
# SYM_EXPR_DELTA = sp.sympify('y - a - b*log(x)')

# sinusoidal function
# SYM_EXPR = sp.sympify('a + b*sin(x)')
# SYM_EXPR_DELTA = sp.sympify('y - (a + b*sin(x))')

MIN_X = 0
MAX_X = 10
NUM_VALS = 20              # number of source values

ERR_X_AVG = 0              # average of X error values
ERR_X_STD = 2              # std of X error values

ERR_Y_AVG = 0              # average of Y error values
ERR_Y_STD = 2              # std of Y error values

NUM_ITER = 10              # number of realizations

LSE_NUM_ITER = 1           # number of LSE iterations

################
# Program code #
################

DESCRIPTION = 'Use this script to determine estimates quality'
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('-w', '--write-to', metavar='PATH',
                    type=str, help='file to write plot in')

# parse cli options
args = parser.parse_args()

# precise X values without errors
precise_x = np.linspace(MIN_X, MAX_X, NUM_VALS, dtype=np.float)

# precise Y values without errors
precise_y = np.vectorize(
    sp.lambdify(
        SYM_X,
        SYM_EXPR.subs(dict(zip(SYM_PARAMS, PRECISE_PARAMS))),
        'numpy'
    )
)(precise_x)

print('Expression:               {}'.format(SYM_EXPR))
print('Symbolic parameters:      {}'.format(SYM_PARAMS))
print('Precise parameter values: {}'.format(PRECISE_PARAMS))
print('Error X std:              {}'.format(ERR_X_STD))
print('Error Y std:              {}'.format(ERR_Y_STD))
print('Number of iterations:     {}'.format(NUM_ITER))

# iterate by error standart derivation values
for iter_i in range(NUM_ITER):
    # add X errors with current normal distribution
    x = np.vectorize(
        lambda v: v + random.gauss(ERR_X_AVG, ERR_X_STD)
    )(precise_x)

    half_len = len(x) // 2

    # add Y errors with current normal distribution
    y = np.vectorize(
        lambda v: v + random.gauss(ERR_Y_AVG, ERR_Y_STD)
    )(precise_y)

    # plot precise values on all figures
    for i in range(4):
        plt.figure(i)
        plt.xlabel('$ X_o $')
        plt.ylabel('$ Y_o $')
        plt.grid(True)
        plt.plot(x, y,
                 color='b', linestyle=' ',
                 marker='.', markersize=10,
                 mfc='r', label='values')

    ################################
    # Base values for basic search #
    ################################

    # set base values to subgroup averages
    base_values = utils.base_values_avg(SYM_X, SYM_Y, x, y, len(SYM_PARAMS))

    print("Base values: {}\n".format(base_values))

    ################
    # Basic search #
    ################

    # find params with basic method
    basic_params = methods.search_basic(
        delta_expression=SYM_EXPR_DELTA,
        parameters=SYM_PARAMS,
        values=base_values
    )

    basic_y = np.vectorize(
        sp.lambdify(
            SYM_X,
            SYM_EXPR.subs(zip(SYM_PARAMS, basic_params)),
            'numpy'
        )
    )(precise_x)

    print('Basic params:       {}'.format(basic_params))

    plt.figure(1)
    plt.plot(precise_x, basic_y,
             color='g', linestyle='-',
             marker='.', markersize=5,
             mfc='g')

    ##############
    # LSE search #
    ##############

    # use basic estimates as init estimates for LSE
    for i, lse_params in methods.search_lse2(
            expression=SYM_EXPR,
            parameters=SYM_PARAMS,
            values={SYM_X: x},
            result_values={SYM_Y: y},
            init_estimates=dict(zip(SYM_PARAMS, basic_params)),
            num_iter=LSE_NUM_ITER
    ):
        lse_y = np.vectorize(
            sp.lambdify(
                SYM_X,
                SYM_EXPR.subs(zip(SYM_PARAMS, lse_params)),
                'numpy'
            )
        )(precise_x)

        print('LSE({}) params:      {}'.format(i, lse_params))

    plt.figure(2)
    # plot only last iteration
    plt.plot(precise_x, lse_y,
             color='b', linestyle='-',
             marker='.', markersize=5,
             mfc='b')

    #################
    # Mrt search #
    #################

    # find params with mrt method
    mrt_params = methods.search_mrt(
        delta_expression=SYM_EXPR_DELTA,
        parameters=SYM_PARAMS,
        values={SYM_X: x, SYM_Y: y},
        err_stds={SYM_X: ERR_X_STD, SYM_Y: ERR_Y_STD}
    )

    mrt_y = np.vectorize(
        sp.lambdify(
            SYM_X,
            SYM_EXPR.subs(zip(SYM_PARAMS, mrt_params)),
            'numpy'
        )
    )(precise_x)

    print('MRT params:         {}'.format(mrt_params))

    plt.figure(3)
    plt.plot(precise_x, mrt_y,
             color='r', linestyle='-',
             marker='.', markersize=5,
             mfc='r')

    print('-' * 40, '\n')

if args.write_to:
    file_name, file_ext = os.path.splitext(args.write_to)

    plt.figure(0)
    plt.savefig('{}_values{}'.format(file_name, file_ext),
                dpi=100)

    plt.figure(1)
    plt.savefig('{}_basic{}'.format(file_name, file_ext),
                dpi=100)

    plt.figure(2)
    plt.savefig('{}_lse{}'.format(file_name, file_ext),
                dpi=100)

    plt.figure(3)
    plt.savefig('{}_mrt{}'.format(file_name, file_ext),
                dpi=100)

plt.show()
