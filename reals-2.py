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
MAX_X = 7
NUM_VALS = 14              # number of source values

REAL_A = 31                # real 'a' value of source distribution
REAL_ALPHA = 0.5           # real 'alpha' value of source distiribution

ERR_X_AVG = 0              # average of X error values
ERR_X_STD = 0.1              # std of X error values

ERR_Y_AVG = 0              # average of Y error values
ERR_Y_STD = 10             # std of Y error values

NUM_ITER = 10             # number of realizations

MNK_NUM_ITER = 1           # number of MNK iterations

################
# Program code #
################

DESCRIPTION = 'Use this script to determine estimates quality'
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

    # plot real values on all figures
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

    # set base values as max distant values
    base_values = base_values_max_dist
    
    print("Base values: {}\n".format(base_values))
    
    ################
    # Basic search #
    ################

    # find params with basic method
    basic_a, basic_alpha = methods.search_basic(
        delta_expression=SYM_EXPR_DELTA,
        parameters=(SYM_A, SYM_ALPHA),
        values=base_values
    )
    
    basic_y = np.vectorize(
        sp.lambdify(
            SYM_X,
            SYM_EXPR.subs({SYM_A: basic_a, SYM_ALPHA: basic_alpha}),
            'numpy'
        )
    )(real_x)

    basic_disp = disp(basic_y, real_y)
    basic_std = std(basic_y, real_y)

    print('Basic a:       {}'.format(basic_a))
    print('Basic alpha:   {}'.format(basic_alpha))
    print('Dispersion:    {}'.format(basic_disp))
    print('Std:           {}\n'.format(basic_std))

    plt.figure(1)
    plt.plot(real_x, basic_y,
             color='g', linestyle='-',
             marker='.', markersize=5,
             mfc='g')

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
        mnk_disp = disp(mnk_y, real_y)
        mnk_std = std(mnk_y, real_y)

        print('MNK({}) a:      {}'.format(i, mnk_a))
        print('MNK({}) alpha:  {}'.format(i, mnk_alpha))
        print('Dispersion:    {}'.format(mnk_disp))
        print('Std:           {}\n'.format(mnk_std))

    plt.figure(2)
    # plot only last iteration
    plt.plot(real_x, mnk_y,
             color='b', linestyle='-',
             marker='.', markersize=5,
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

    mrt_y = np.vectorize(
        sp.lambdify(
            SYM_X,
            SYM_EXPR.subs({SYM_A: mrt_a,
                           SYM_ALPHA: mrt_alpha}),
            'numpy'
        )
    )(real_x)

    mrt_disp = disp(mrt_y, real_y)
    mrt_std = std(mrt_y, real_y)

    print('Mrt a:         {}'.format(mrt_a))
    print('Mrt alpha:     {}'.format(mrt_alpha))
    print('Dispersion:    {}'.format(mrt_disp))
    print('Std:           {}'.format(mrt_std))

    plt.figure(3)
    plt.plot(real_x, mrt_y,
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
    plt.savefig('{}_mnk{}'.format(file_name, file_ext),
                dpi=100)

    plt.figure(3)
    plt.savefig('{}_mrt{}'.format(file_name, file_ext),
                dpi=100)

plt.show()
