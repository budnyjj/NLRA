#!/usr/bin/env python3

import os.path
import argparse
import numpy as np
import sympy as sp

from sympy.core.sympify import SympifyError

import stats.estimators as estimators
import stats.methods as methods
import stats.accuracy as accuracy

################
# Declarations #
################

DESCRIPTION = 'Use this script to determine estimates accuracy'

SYM_X, SYM_Y = SYM_VALUES = sp.symbols('x y')
SYM_ALPHA, SYM_BETA = SYM_PARAMS = sp.symbols('a b')

# SYM_EXPR = sp.sympify('a * exp(-alpha*x)')
# SYM_EXPR_DELTA = sp.sympify('y - a * exp(-alpha*x)')

# linear function
# SYM_EXPR = sp.sympify('a + b*x')
# SYM_EXPR_DELTA = sp.sympify('y - a - b*x')

# quadratic function
SYM_EXPR = sp.sympify('a*x + b*(x**2)')
SYM_EXPR_DELTA = sp.sympify('y - a*x - b*(x**2)')

# cubic function
# SYM_EXPR = sp.sympify('a*(x**2) + b*(x**3)')
# SYM_EXPR_DELTA = sp.sympify('y - b*(x**3) - a*(x**2)')

# logarithmic function
# SYM_EXPR = sp.sympify('a + alpha*log(x)')
# SYM_EXPR_DELTA = sp.sympify('y - a - alpha*log(x)')

# exponential function
# SYM_EXPR = sp.sympify('a * exp(b*x)')
# SYM_EXPR_DELTA = sp.sympify('y - a * exp(b*x)')

# sinusoidal function
# SYM_EXPR = sp.sympify('a + alpha*sin(x)')
# SYM_EXPR_DELTA = sp.sympify('y - (a + alpha*sin(x))')

MIN_X = 0
MAX_X = 10
NUM_VALS = 100          # number of source values

PRECISE_ALPHA = 0       # real 'alpha' value of source distribution
PRECISE_BETA = -5       # real 'beta' value of source distiribution

ERR_NUM_STD_ITER = 20   # number of stds iterations

ERR_MIN_STD_X = 0.001   # minimal std of X error values
ERR_MAX_STD_X = 2.001   # maximal std of X error values
ERR_STEP_STD_X = (ERR_MAX_STD_X - ERR_MIN_STD_X) / ERR_NUM_STD_ITER

ERR_MIN_STD_Y = 0.001   # minimal std of Y error values
ERR_MAX_STD_Y = 2.001   # maximal std of Y error values
ERR_STEP_STD_Y = (ERR_MAX_STD_Y - ERR_MIN_STD_Y) / ERR_NUM_STD_ITER

NUM_ITER = 100          # number of realizations
LSE_NUM_ITER = 1        # number of LSE iterations

################
# Program code #
################

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    '-o', '--output', metavar='PATH',
    type=str, required=True,
    help='base path to write data')
args = parser.parse_args()
output_path, _ = os.path.splitext(args.output)

print('Expression:               {}'.format(SYM_EXPR))
print('Precise ALPHA:            {}'.format(PRECISE_ALPHA))
print('Precise BETA:             {}'.format(PRECISE_BETA))
print('Real X:                   {}..{}'.format(MIN_X, MAX_X))
print('STD X:                    {}..{}'.format(ERR_MIN_STD_X, ERR_MAX_STD_X))
print('STD X step:               {}'.format(ERR_STEP_STD_X))
print('STD Y:                    {}..{}'.format(ERR_MIN_STD_Y, ERR_MAX_STD_Y))
print('STD Y step:               {}'.format(ERR_STEP_STD_Y))
print('Number of iterations:     {}'.format(NUM_ITER))
print('Output path:              {}'.format(output_path))

# build precise values
precise_expr = sp.lambdify(
    SYM_X,
    SYM_EXPR.subs({SYM_ALPHA: PRECISE_ALPHA, SYM_BETA: PRECISE_BETA}),
    'numpy')
precise_vectorized = np.vectorize(precise_expr)
# get precise values
precise_vals_x, precise_vals_y = estimators.precise(
    precise_vectorized, NUM_VALS,
    MIN_X, MAX_X)

# generate array of X error stds
err_stds_x = np.linspace(ERR_MIN_STD_X, ERR_MAX_STD_X, ERR_NUM_STD_ITER)
# generate array of Y error stds
err_stds_y = np.linspace(ERR_MIN_STD_Y, ERR_MAX_STD_Y, ERR_NUM_STD_ITER)
# create meshgrid
err_stds_x, err_stds_y = np.meshgrid(err_stds_x, err_stds_y)

# collect accuracies of estimates
basic_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
lse_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
mrt_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))

num_std_iter = ERR_NUM_STD_ITER**2

half_num_vals = int(NUM_VALS/2)
# iterate by error standard derivation values
std_iter = 0
for std_i, err_std_row in enumerate(np.dstack((err_stds_x, err_stds_y))):
    for std_j, (err_std_x, err_std_y) in enumerate(err_std_row):
        std_iter += 1
        print("Iteration {}/{}: std X: {:.2f}, std Y: {:.2f}".format(
              std_iter, num_std_iter, err_std_x, err_std_y))

        # iterate by error standart derivation values
        for iter_i in range(NUM_ITER):
            measured_vals_x, measured_vals_y = estimators.uniform(
                precise_vectorized, NUM_VALS,
                MIN_X, MAX_X,
                err_std_x, err_std_y)

            ################################
            # Base values for basic search #
            ################################

            # get base values as first pairs of values
            base_values_first = {
                SYM_X: [measured_vals_x[0], measured_vals_x[1]],
                SYM_Y: [measured_vals_y[0], measured_vals_y[1]]
            }

            # get base values as half-distant pairs of values
            base_values_half_dist = {
                SYM_X: [measured_vals_x[0], measured_vals_x[half_num_vals]],
                SYM_Y: [measured_vals_y[0], measured_vals_y[half_num_vals]]
            }

            # get base values as maximal distant pairs of values
            base_values_max_dist = {
                SYM_X: [measured_vals_x[0], measured_vals_x[-1]],
                SYM_Y: [measured_vals_y[0], measured_vals_y[-1]]
            }

            # get base values as averages of two half-length subgroups
            base_values_avg = {
                SYM_X: [
                    np.average(measured_vals_x[:half_num_vals]),
                    np.average(measured_vals_x[half_num_vals:])
                ],
                SYM_Y: [
                    np.average(measured_vals_y[:half_num_vals]),
                    np.average(measured_vals_y[half_num_vals:])
                ]
            }

            # set base values as max distant values
            base_values = base_values_max_dist

            ################
            # Basic search #
            ################
            # find params with basic method
            basic_alpha, basic_beta = methods.search_basic(
                delta_expression=SYM_EXPR_DELTA,
                parameters=(SYM_ALPHA, SYM_BETA),
                values=base_values
            )
            # print('Basic alpha: {}'.format(basic_alpha))
            # print('Basic beta:  {}'.format(basic_beta))
            # add distance between estimates and real values
            basic_acc = accuracy.avg_euclidean_dst(
                np.array(((PRECISE_ALPHA), (PRECISE_BETA))),
                np.array(((basic_alpha), (basic_beta))))
            basic_accs[std_i, std_j] += basic_acc

            ##############
            # LSE search #
            ##############
            # use basic estimates as init estimates for LSE
            lse_alpha, lse_beta = methods.search_lse(
                    expression=SYM_EXPR,
                    parameters=(SYM_ALPHA, SYM_BETA),
                    values={SYM_X: measured_vals_x},
                    result_values={SYM_Y: measured_vals_y},
                    init_estimates={SYM_ALPHA: basic_alpha, SYM_BETA: basic_beta},
                    num_iter=LSE_NUM_ITER)
            # print('LSE({}) alpha: {}'.format(LSE_NUM_ITER, lse_alpha))
            # print('LSE({}) beta:  {}'.format(LSE_NUM_ITER, lse_beta))
            lse_acc = accuracy.avg_euclidean_dst(
                np.array(((PRECISE_ALPHA), (PRECISE_BETA))),
                np.array(((lse_alpha), (lse_beta))))
            lse_accs[std_i, std_j] += lse_acc

            ##############
            # Mrt search #
            ##############
            # find params with mrt method
            mrt_alpha, mrt_beta = methods.search_mrt(
                delta_expression=SYM_EXPR_DELTA,
                parameters=(SYM_ALPHA, SYM_BETA),
                values={SYM_X: measured_vals_x, SYM_Y: measured_vals_y},
                err_stds={SYM_X: err_std_x, SYM_Y: err_std_y}
            )
            # print('MRT alpha:    {}'.format(mrt_alpha))
            # print('MRT beta:     {}'.format(mrt_beta))
            mrt_acc = accuracy.avg_euclidean_dst(
                np.array(((PRECISE_ALPHA), (PRECISE_BETA))),
                np.array(((mrt_alpha), (mrt_beta))))
            mrt_accs[std_i, std_j] += mrt_acc

basic_accs /= NUM_ITER
lse_accs /= NUM_ITER
mrt_accs /= NUM_ITER

np.save(
    '{}_err-stds-x.npy'.format(output_path),
    err_stds_x)
np.save(
    '{}_err-stds-y.npy'.format(output_path),
    err_stds_y)
np.save(
    '{}_lse-accs.npy'.format(output_path),
    lse_accs)
np.save(
    '{}_mrt-accs.npy'.format(output_path),
    mrt_accs)
