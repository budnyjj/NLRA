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
SYM_A, SYM_B = SYM_PARAMS = sp.symbols('a b')

# SYM_EXPR = sp.sympify('a * exp(-a*x)')
# SYM_EXPR_DELTA = sp.sympify('y - a * exp(-a*x)')

# linear function
# SYM_EXPR = sp.sympify('a + b*x')
# SYM_EXPR_DELTA = sp.sympify('y - a - b*x')

# quadratic function
# SYM_EXPR = sp.sympify('a*x + b*(x**2)')
# SYM_EXPR_DELTA = sp.sympify('y - a*x - b*(x**2)')

# hyperbolic function
SYM_EXPR = sp.sympify('a + 1/(b*(x+1))')
SYM_EXPR_DELTA = sp.sympify('y - a - 1/(b*(x+1))')

# sinusoidal function
# SYM_EXPR = sp.sympify('a + b*sin(0.2*x)')
# SYM_EXPR_DELTA = sp.sympify('y - (a + b*sin(0.2*x))')

# logarithmic function
# SYM_EXPR = sp.sympify('a + a*log(x)')
# SYM_EXPR_DELTA = sp.sympify('y - a - a*log(x)')

# exponential function
# SYM_EXPR = sp.sympify('a * exp(b*x)')
# SYM_EXPR_DELTA = sp.sympify('y - a * exp(b*x)')


MIN_X = 0
MAX_X = 10
NUM_VALS = 100          # number of source values

PRECISE_A = 0           # real 'a' value of source distribution
PRECISE_B = 0.1       # real 'b' value of source distiribution

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
print('Precise A:                {}'.format(PRECISE_A))
print('Precise B:                {}'.format(PRECISE_B))
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
    SYM_EXPR.subs({SYM_A: PRECISE_A, SYM_B: PRECISE_B}),
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
            basic_a, basic_b = methods.search_basic(
                delta_expression=SYM_EXPR_DELTA,
                parameters=(SYM_A, SYM_B),
                values=base_values
            )
            # print('Basic a: {}'.format(basic_a))
            # print('Basic b:  {}'.format(basic_b))
            # add distance between estimates and real values
            basic_acc = accuracy.avg_euclidean_dst(
                np.array(((PRECISE_A), (PRECISE_B))),
                np.array(((basic_a), (basic_b))))
            basic_accs[std_i, std_j] += basic_acc

            ##############
            # LSE search #
            ##############
            # use basic estimates as init estimates for LSE
            lse_a, lse_b = methods.search_lse(
                    expression=SYM_EXPR,
                    parameters=(SYM_A, SYM_B),
                    values={SYM_X: measured_vals_x},
                    result_values={SYM_Y: measured_vals_y},
                    init_estimates={SYM_A: basic_a, SYM_B: basic_b},
                    num_iter=LSE_NUM_ITER)
            # print('LSE({}) a: {}'.format(LSE_NUM_ITER, lse_a))
            # print('LSE({}) b:  {}'.format(LSE_NUM_ITER, lse_b))
            lse_acc = accuracy.avg_euclidean_dst(
                np.array(((PRECISE_A), (PRECISE_B))),
                np.array(((lse_a), (lse_b))))
            lse_accs[std_i, std_j] += lse_acc

            ##############
            # Mrt search #
            ##############
            # find params with mrt method
            mrt_a, mrt_b = methods.search_mrt(
                delta_expression=SYM_EXPR_DELTA,
                parameters=(SYM_A, SYM_B),
                values={SYM_X: measured_vals_x, SYM_Y: measured_vals_y},
                err_stds={SYM_X: err_std_x, SYM_Y: err_std_y}
            )
            # print('MRT a:    {}'.format(mrt_a))
            # print('MRT b:     {}'.format(mrt_b))
            mrt_acc = accuracy.avg_euclidean_dst(
                np.array(((PRECISE_A), (PRECISE_B))),
                np.array(((mrt_a), (mrt_b))))
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
