#!/usr/bin/env python3

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
NUM_VALS = 20          # number of source values

REAL_A = 31            # real 'a' value of source distribution
REAL_ALPHA = 0.5       # real 'alpha' value of source distiribution

ERR_X_AVG = 0          # average of X error values
ERR_X_MIN_STD = 0.001  # minimal std of X error values
ERR_X_MAX_STD = 0.1  # maximal std of X error values

ERR_Y_AVG = 0          # average of Y error values
ERR_Y_MIN_STD = 10      # minimal std of Y error values
ERR_Y_MAX_STD = 10      # maximal std of Y error values

ERR_NUM_STD_ITER = 10   # number of stds iterations
NUM_ITER = 50           # number of realizations

MNK_NUM_ITER = 1       # number of MNK iterations

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

# generate array of X error stds
err_x_stds = np.linspace(ERR_X_MIN_STD, ERR_X_MAX_STD, ERR_NUM_STD_ITER)

# generate array of Y error stds
err_y_stds = np.linspace(ERR_Y_MIN_STD, ERR_Y_MAX_STD, ERR_NUM_STD_ITER)

# collect accuracies of estimates
basic_accs = []
mnk_accs = []
mrt_accs = []

print('Expression:    {}'.format(SYM_EXPR))
print('Real A:        {}'.format(REAL_A))
print('Real ALPHA:    {}'.format(REAL_ALPHA))
print('Number of iterations: {}'.format(ERR_NUM_STD_ITER * NUM_ITER))
print('-' * 40, '\n')

# iterate by error standart derivation values
for err_x_std, err_y_std in zip(err_x_stds, err_y_stds):
    print('Error X std:   {}'.format(err_x_std))
    print('Error Y std:   {}\n'.format(err_y_std))

    # current accuracies for this std
    cur_basic_acc = 0
    cur_mnk_acc = 0
    cur_mrt_acc = 0

    # iterate by error standart derivation values
    for iter_i in range(NUM_ITER):
        print('Iteration #{}:'.format(iter_i + 1))

        # add X errors with current normal distribution
        x = np.vectorize(
            lambda v: v + random.gauss(ERR_X_AVG, err_x_std)
        )(real_x)

        half_len = len(x) / 2

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(ERR_Y_AVG, err_y_std)
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
        cur_mnk_acc += math.sqrt(cur_mnk_dst)

        #################
        # Mrt search #
        #################

        # find params with mrt method
        mrt_a, mrt_alpha = methods.search_mrt(
            delta_expression=SYM_EXPR_DELTA,
            parameters=(SYM_A, SYM_ALPHA),
            values={SYM_X: x, SYM_Y: y},
            err_stds={SYM_X: err_x_std, SYM_Y: err_y_std}
        )

        print('MRT a:         {}'.format(mrt_a))
        print('MRT alpha:     {}'.format(mrt_alpha))

        # add distance between estimates and real values
        cur_mrt_dst = (mrt_a - REAL_A)**2 + (mrt_alpha - REAL_ALPHA)**2
        cur_mrt_acc += math.sqrt(cur_mrt_dst)

        print('-' * 40, '\n')

    print('Basic accuracy: {}'.format(cur_basic_acc))
    print('MNK accuracy:   {}'.format(cur_mnk_acc))
    print('MRT accuracy:   {}'.format(cur_mrt_acc))

    print('-' * 40, '\n')

    # append current accuracy to accuracies
    basic_accs.append(cur_basic_acc / NUM_ITER)
    mnk_accs.append(cur_mnk_acc / NUM_ITER)
    mrt_accs.append(cur_mrt_acc / NUM_ITER)

# basic_accs_plot, = plt.plot(err_y_stds, basic_accs,
#                             color='g', linestyle='-',
#                             marker='o', markersize=5,
#                             mfc='g', label="base")

mnk_accs_plot, = plt.plot(err_x_stds, mnk_accs,
                          color='b', linestyle='-',
                          marker='s', markersize=5,
                          mfc='b', label='HMHK({})'.format(MNK_NUM_ITER))

mrt_accs_plot, = plt.plot(err_x_stds, mrt_accs,
                          color='r', linestyle='-',
                          marker='v', markersize=5,
                          mfc='r', label='MPT')

plt.legend(handles=[  # basic_accs_plot,
    mnk_accs_plot, mrt_accs_plot], fontsize=16)

plt.axis([ERR_X_MIN_STD, ERR_X_MAX_STD, 0, 6])
plt.xlabel('$ \\sigma_x $')
plt.ylabel('$ \\rho $')
plt.grid(True)

if args.write_to:
    plt.savefig(args.write_to, dpi=100)

plt.show()
