#!/usr/bin/env python3

import os.path
import argparse
import math
import random
import numpy as np
import sympy as sp

from sympy.core.sympify import SympifyError

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
ERR_X_MAX_STD = 0.101  # maximal std of X error values

ERR_Y_AVG = 0          # average of Y error values
ERR_Y_MIN_STD = 0.1    # minimal std of Y error values
ERR_Y_MAX_STD = 10.1   # maximal std of Y error values

ERR_NUM_STD_ITER = 10  # number of stds iterations
NUM_ITER = 50          # number of realizations

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

# create meshgrid
err_x_stds, err_y_stds = np.meshgrid(err_x_stds, err_y_stds)

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
for err_std_row in np.dstack((err_x_stds, err_y_stds)):
    row_basic_accs = []
    row_mnk_accs = []
    row_mrt_accs = []
    
    for err_x_std, err_y_std in err_std_row:
        print('Error X std:   {}'.format(err_x_std))
        print('Error Y std:   {}'.format(err_y_std))
        print('=' * 40, '\n')
            
        # current accuracies for this std
        cur_basic_acc = 0
        cur_mnk_acc = 0
        cur_mrt_acc = 0

        # number of successful iterations
        basic_num_success_iter = 0
        mnk_num_success_iter = 0
        mrt_num_success_iter = 0
        
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

            # set base values as max distant values
            base_values = base_values_max_dist

            print("Base values: {}\n".format(base_values))

            ################
            # Basic search #
            ################

            try:
                # find params with basic method
                basic_a, basic_alpha = methods.search_basic(
                    delta_expression=SYM_EXPR_DELTA,
                    parameters=(SYM_A, SYM_ALPHA),
                    values=base_values
                )

                print('Basic a:       {}'.format(basic_a))
                print('Basic alpha:   {}'.format(basic_alpha))

                # add distance between estimates and real values
                cur_basic_dst = (basic_a - REAL_A)**2 + (basic_alpha - REAL_ALPHA)**2
                cur_basic_acc += math.sqrt(cur_basic_dst)

            except (TypeError, SympifyError):
                # If got complex estimate, pass it
                print("Got complex estimates. Skip it...")
            else:
                # increase number of successfull iterations
                basic_num_success_iter += 1
                
            ##############
            # MNK search #
            ##############

            try:
                mnk_a = mnk_alpha = 0
            
                # use basic estimates as init estimates for MNK
                for i, (mnk_tmp_a, mnk_tmp_alpha) in methods.search_mnk(
                        expression=SYM_EXPR,
                        parameters=(SYM_A, SYM_ALPHA),
                        values={SYM_X: x},
                        result_values={SYM_Y: y},
                        init_estimates={SYM_A: basic_a, SYM_ALPHA: basic_alpha},
                        num_iter=MNK_NUM_ITER
                ):
                    mnk_a, mnk_alpha = mnk_tmp_a, mnk_tmp_alpha

                print('MNK({}) a:      {}'.format(MNK_NUM_ITER, mnk_a))
                print('MNK({}) alpha:  {}'.format(MNK_NUM_ITER, mnk_alpha))

                # add distance between estimates and real values
                cur_mnk_dst = (mnk_a - REAL_A)**2 + (mnk_alpha - REAL_ALPHA)**2
                cur_mnk_acc += math.sqrt(cur_mnk_dst)

            except (TypeError, SympifyError):
                # If got complex estimate, pass it
                print("Got complex estimates. Skip it...")
            else:
                # increase number of successfull iterations
                mnk_num_success_iter += 1
                
            ##############
            # Mrt search #
            ##############
            try:
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

            except (TypeError, SympifyError):
                # If got complex estimate, pass it
                print("Got complex estimates. Skip it...")
            else:
               # increase number of successfull iterations
                mrt_num_success_iter += 1

            print('-' * 40, '\n')

        print('Basic accuracy:         {}'.format(cur_basic_acc))
        print('Successfull iterations: {}\n'.format(basic_num_success_iter))

        print('MNK accuracy:           {}'.format(cur_mnk_acc))
        print('Successfull iterations: {}\n'.format(mnk_num_success_iter))

        print('MRT accuracy:           {}'.format(cur_mrt_acc))
        print('Successfull iterations: {}\n'.format(mrt_num_success_iter))

        print('#' * 40, '\n')

        # append average accuracy to row accuracies
        avg_basic_accs = 0
        if basic_num_success_iter > 0:
            avg_basic_accs = cur_basic_acc / basic_num_success_iter
        row_basic_accs.append(avg_basic_accs)

        avg_mnk_accs = 0
        if mnk_num_success_iter > 0:
            avg_mnk_accs = cur_mnk_acc / mnk_num_success_iter
        row_mnk_accs.append(avg_mnk_accs)

        avg_mrt_accs = 0
        if mrt_num_success_iter > 0:
            avg_mrt_accs = cur_mrt_acc / mrt_num_success_iter
        row_mrt_accs.append(avg_mrt_accs)

    # append row accuracies to accumulator
    basic_accs.append(row_basic_accs)
    mnk_accs.append(row_mnk_accs)
    mrt_accs.append(row_mrt_accs)

# convert to numpy array
basic_accs = np.array(basic_accs)
mnk_accs = np.array(mnk_accs)
mrt_accs = np.array(mrt_accs)

# print(err_x_stds)
# print(err_y_stds)
# print(basic_accs)
# print(mnk_accs)
# print(mrt_accs)

basic_fig = plt.figure(0)
basic_ax = basic_fig.gca(projection='3d')
basic_ax.view_init(elev=15., azim=240)
basic_ax.set_xlabel('$ \\sigma_x $')
basic_ax.set_ylabel('$ \\sigma_y $')
basic_ax.set_zlabel('$ \\rho $')
basic_accs_surf = basic_ax.plot_surface(
    err_x_stds, err_y_stds, basic_accs,
    rstride=1, cstride=1,
    cmap=cm.coolwarm,
    label='MPT'
)
basic_ax.zaxis.set_major_locator(LinearLocator(10))
basic_ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
basic_ax.set_zlim(0, 6)


mnk_fig = plt.figure(1)
mnk_ax = mnk_fig.gca(projection='3d')
mnk_ax.view_init(elev=15., azim=240)
mnk_ax.set_xlabel('$ \\sigma_x $')
mnk_ax.set_ylabel('$ \\sigma_y $')
mnk_ax.set_zlabel('$ \\rho $')
mnk_accs_surf = mnk_ax.plot_surface(
    err_x_stds, err_y_stds, mnk_accs,
    rstride=1, cstride=1,
    cmap=cm.coolwarm,
    label='MHK({})'.format(MNK_NUM_ITER)
)
mnk_ax.zaxis.set_major_locator(LinearLocator(10))
mnk_ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
mnk_ax.set_zlim(0, 6)


mrt_fig = plt.figure(2)
mrt_ax = mrt_fig.gca(projection='3d')
mrt_ax.view_init(elev=15., azim=240)
mrt_ax.set_xlabel('$ \\sigma_x $')
mrt_ax.set_ylabel('$ \\sigma_y $')
mrt_ax.set_zlabel('$ \\rho $')
mrt_accs_surf = mrt_ax.plot_surface(
    err_x_stds, err_y_stds, mrt_accs,
    rstride=1, cstride=1,
    cmap=cm.coolwarm,
    label='MPT'
)
mrt_ax.zaxis.set_major_locator(LinearLocator(10))
mrt_ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
mrt_ax.set_zlim(0, 6)

if args.write_to:
    file_name, file_ext = os.path.splitext(args.write_to)

    plt.figure(0)
    plt.savefig('{}_basic{}'.format(file_name, file_ext),
                dpi=200)
    
    plt.figure(1)
    plt.savefig('{}_mnk{}'.format(file_name, file_ext),
                dpi=200)

    plt.figure(2)
    plt.savefig('{}_mrt{}'.format(file_name, file_ext),
                dpi=200)

plt.show()
