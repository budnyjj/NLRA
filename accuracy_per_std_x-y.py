#!/usr/bin/env python3

import os.path
import argparse
import numpy as np
import sympy as sp

from numpy.linalg.linalg import LinAlgError

import stats.accuracy as accuracy
import stats.estimators as estimators
import stats.methods as methods
import stats.utils as utils

################
# Declarations #
################

DESCRIPTION = 'Use this script to determine estimates accuracy'

SYM_PARAMS = sp.symbols('a b c')
PRECISE_PARAMS = (0, 0.07, 0.01)
SYM_X, SYM_Y = sp.symbols('x y')

# SYM_EXPR = sp.sympify('a * exp(-a*x)')
# SYM_EXPR_DELTA = sp.sympify('y - a * exp(-a*x)')

# linear function
# SYM_EXPR = sp.sympify('a + b*x')
# SYM_EXPR_DELTA = sp.sympify('y - a - b*x')

# quadratic function
# SYM_EXPR = sp.sympify('a + b*x + c*(x**2)')
# SYM_EXPR_DELTA = sp.sympify('y - a - b*x - c*(x**2)')

# inverse function
SYM_EXPR = sp.sympify('a + 1/(b + c*x)')
SYM_EXPR_DELTA = sp.sympify('y - (a + 1/(b + c*x))')

# exponential function
# SYM_EXPR = sp.sympify('10 + exp(a + b*x)')
# SYM_EXPR_DELTA = sp.sympify('y - 10 - exp(a + b*x)')

# logarithmic function
# SYM_EXPR = sp.sympify('b + c*ln(x+10)')
# SYM_EXPR_DELTA = sp.sympify('y - (b + c*ln(x+10))')

# logistic function
# SYM_EXPR = sp.sympify('1/(1+exp(-b*x))')
# SYM_EXPR_DELTA = sp.sympify('y - 1/(1+exp(-b*x))')

# sinusoidal function
# SYM_EXPR = sp.sympify('a + b*sin(0.2*x)')
# SYM_EXPR_DELTA = sp.sympify('y - (a + b*sin(0.2*x))')


MIN_X = 0
MAX_X = 10
NUM_VALS = 100          # number of source values

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
print('Symbolic parameters:      {}'.format(SYM_PARAMS))
print('Precise parameter values: {}'.format(PRECISE_PARAMS))
print('Real X:                   {}..{}'.format(MIN_X, MAX_X))
print('STD X:                    {}..{}'.format(ERR_MIN_STD_X, ERR_MAX_STD_X))
print('STD X step:               {}'.format(ERR_STEP_STD_X))
print('STD Y:                    {}..{}'.format(ERR_MIN_STD_Y, ERR_MAX_STD_Y))
print('STD Y step:               {}'.format(ERR_STEP_STD_Y))
print('Number of iterations:     {}'.format(NUM_ITER))
print('Output path:              {}'.format(output_path))

# get precise param values
precise_params = np.vstack(PRECISE_PARAMS)
# build precise values
precise_expr = sp.lambdify(
    SYM_X,
    SYM_EXPR.subs(zip(SYM_PARAMS, PRECISE_PARAMS)),
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

            # set base values as max distant values
            base_values = utils.base_values_avg(
                SYM_X, SYM_Y,
                measured_vals_x, measured_vals_y,
                len(SYM_PARAMS))

            ################
            # Basic search #
            ################
            # find params with basic method
            basic_params = methods.search_basic(
                delta_expression=SYM_EXPR_DELTA,
                parameters=SYM_PARAMS,
                values=base_values
            )
            # print('Basic params: {}'.format(basic_params))
            # add distance between estimates and real values
            basic_accs[std_i, std_j] += accuracy.avg_euclidean_dst(
                precise_params,
                np.vstack(basic_params))

            ##############
            # LSE search #
            ##############
            while True:
                # use basic estimates as init estimates for LSE
                try:
                    lse_params = methods.search_lse(
                        expression=SYM_EXPR,
                        parameters=SYM_PARAMS,
                        values={SYM_X: measured_vals_x},
                        result_values={SYM_Y: measured_vals_y},
                        init_estimates=dict(zip(SYM_PARAMS, basic_params)),
                        num_iter=LSE_NUM_ITER)
                    # print('LSE params: {}'.format(lse_params))
                    lse_accs[std_i, std_j] += accuracy.avg_euclidean_dst(
                        precise_params,
                        np.vstack(lse_params))
                    break
                except LinAlgError:
                    print('LSE: singular matrix')

            ##############
            # MRT search #
            ##############
            # find params with mrt method
            while True:
                try:
                    mrt_params = methods.search_mrt(
                        delta_expression=SYM_EXPR_DELTA,
                        parameters=SYM_PARAMS,
                        values={SYM_X: measured_vals_x, SYM_Y: measured_vals_y},
                        err_stds={SYM_X: err_std_x, SYM_Y: err_std_y}
                    )
                    # print('MRT params:    {}'.format(mrt_params))
                    mrt_accs[std_i, std_j] += accuracy.avg_euclidean_dst(
                        precise_params,
                        np.vstack(mrt_params))
                    break
                except LinAlgError:
                    print('MRT: singular matrix')


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
