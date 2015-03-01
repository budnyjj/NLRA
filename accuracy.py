#!/usr/bin/env python

import os.path
import argparse
import math
import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import stats.methods as methods
from stats.utils import *

################
# Declarations #
################

SYM_X, SYM_Y = SYM_VALUES = sp.symbols('x y')
SYM_A, SYM_ALPHA = SYM_PARAMS = sp.symbols('a alpha')

# SYM_EXPR = sp.sympify('a * exp(-alpha*x)')
# SYM_EXPR_DELTA = sp.sympify('y - a * exp(-alpha*x)')

SYM_EXPR = sp.sympify('a * exp(alpha*x)')
SYM_EXPR_DELTA = sp.sympify('y - a * exp(alpha*x)')

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
# SYM_EXPR = sp.sympify('a + alpha*sin(x)')
# SYM_EXPR_DELTA = sp.sympify('y - (a + alpha*sin(x))')

MIN_X = 0
MAX_X = 10
NUM_VALS = 20              # number of source values

REAL_A = 31                # real 'a' value of source distribution
REAL_ALPHA = 0.5           # real 'alpha' value of source distiribution

ERR_X_AVG = 0              # average of X error values
ERR_X_STD = 0.02              # std of X error values

ERR_Y_AVG = 0              # average of Y error values
ERR_Y_STD = 7              # std of Y error values

NUM_ITER = 10              # number of realizations

MNK_NUM_ITER = 1             # number of MNK iterations

################
# Program code #
################

DESCRIPTION = 'Use this script to determine estimates accuracy'
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('-w', '--write-to', metavar='PATH',
                    type=str, help='file to write plot in')

# parse cli options
args = parser.parse_args()

# real X values without errors
real_x = np.linspace(MIN_X, MAX_X, NUM_VALS, dtype=np.float)

# real Y values without errors
real_y = np.vectorize(
    sp.lambdify(
        SYM_X,
        SYM_EXPR.subs({SYM_A: REAL_A, SYM_ALPHA: REAL_ALPHA}),
        'numpy'
    )
)(real_x)

print('Expression:    {}'.format(SYM_EXPR))
print('Real A:        {}'.format(REAL_A))
print('Real ALPHA:    {}'.format(REAL_ALPHA))
print('Error X std:   {}'.format(ERR_X_STD))
print('Error Y std:   {}'.format(ERR_Y_STD))
print('Number of iterations: {}'.format(NUM_ITER))
print('-' * 40, '\n')

# plot real parameters
plt.figure(0)
plt.plot(REAL_A, REAL_ALPHA,
         color='m', linestyle=' ',
         marker='x', markersize=10,
         mfc='r')

# current accuracies for this stds
cur_basic_acc = 0
cur_mnk_acc = 0
cur_mrt_acc = 0

# iterate by error standart derivation values
for iter_i in range(NUM_ITER):
    print('Iteration #{}:'.format(iter_i + 1))

    # add X errors with current normal distribution
    x = np.vectorize(
        lambda v: v + random.gauss(ERR_X_AVG, ERR_X_STD)
    )(real_x)

    half_len = len(x) / 2

    # add Y errors with current normal distribution
    y = np.vectorize(
        lambda v: v + random.gauss(ERR_Y_AVG, ERR_Y_STD)
    )(real_y)

    ################################
    # Base values for basic search #
    ################################

    # get base values as first pairs of values
    base_values_first = {
        SYM_X: [x[0], x[1]],
        SYM_Y: [y[0], y[1]]
    }

    # get base values as half-distant pairs of values
    base_values_half_dist = {
        SYM_X: [x[0], x[half_len]],
        SYM_Y: [y[0], y[half_len]]
    }

    # get base values as maximal distant pairs of values
    base_values_max_dist = {
        SYM_X: [x[0], x[-1]],
        SYM_Y: [y[0], y[-1]]
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
        values=base_values_max_dist
    )

    print('Basic a:       {}'.format(basic_a))
    print('Basic alpha:   {}'.format(basic_alpha))

    # add distance between estimates and real values
    cur_basic_dst = (basic_a - REAL_A)**2 + (basic_alpha - REAL_ALPHA)**2
    cur_basic_acc += math.sqrt(cur_basic_dst)

    plt.figure(0)
    plt.plot(basic_a, basic_alpha,
             color='g', linestyle=' ',
             marker='.', markersize=10,
             mfc='g', label='values')

    ##############
    # MNK search #
    ##############

    # use basic estimates as init estimates for MNK
    for i, (mnk_a, mnk_alpha) in methods.search_mnk(
            expression=SYM_EXPR,
            parameters=(SYM_A, SYM_ALPHA),
            values={SYM_X: x},
            result_values={SYM_Y: y},
            init_estimates={SYM_A: basic_a, SYM_ALPHA: basic_alpha},
            num_iter=MNK_NUM_ITER
    ):
        mnk_y = np.vectorize(
            sp.lambdify(
                SYM_X,
                SYM_EXPR.subs({SYM_A: mnk_a,
                               SYM_ALPHA: mnk_alpha}),
                'numpy'
            )
        )(real_x)

        print('MNK({}) a:      {}'.format(i, mnk_a))
        print('MNK({}) alpha:  {}'.format(i, mnk_alpha))

    # add distance between estimates and real values
    cur_mnk_dst = (mnk_a - REAL_A)**2 + (mnk_alpha - REAL_ALPHA)**2
    cur_mnk_dst += math.sqrt(cur_mnk_dst)

    plt.figure(0)
    plt.plot(mnk_a, mnk_alpha,
             color='b', linestyle=' ',
             marker='.', markersize=10,
             mfc='b')

    #################
    # Mrt search #
    #################

    # find params with mrt method
    mrt_a, mrt_alpha = methods.search_mrt(
        delta_expression=SYM_EXPR_DELTA,
        parameters=(SYM_A, SYM_ALPHA),
        values={SYM_X: x, SYM_Y: y},
        err_stds={SYM_X: ERR_X_STD, SYM_Y: ERR_Y_STD}
    )

    print('MRT a:         {}'.format(mrt_a))
    print('MRT alpha:     {}'.format(mrt_alpha))

    # add distance between estimates and real values
    cur_mrt_dst = (mrt_a - REAL_A)**2 + (mrt_alpha - REAL_ALPHA)**2
    cur_mrt_acc += math.sqrt(cur_mrt_dst)

    plt.figure(0)
    plt.plot(mrt_a, mrt_alpha,
             color='r', linestyle=' ',
             marker='.', markersize=10,
             mfc='r')

    print('-' * 40, '\n')

print('Basic accuracy: {}'.format(cur_basic_acc))
print('MNK accuracy:   {}'.format(cur_mnk_acc))
print('MRT accuracy:   {}'.format(cur_mrt_acc))

plt.figure(0)
plt.xlabel('$ a $')
plt.ylabel('$ \\alpha $')
plt.grid(True)

if args.write_to:
    plt.savefig(args.write_to, dpi=100)

plt.show()
