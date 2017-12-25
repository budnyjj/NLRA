import unittest

import random
import sympy as sp
import numpy as np


import sys
import os
sys.path.append('.')

import stats.methods as methods
from stats.utils import *


class TestBasicMNK(unittest.TestCase):

    def setUp(self):
        self.num_vals = 20          # number of source values

    def test_linear_k(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_k = sp.symbols('k')
        sym_expr = sp.sympify('k*x')
        sym_expr_delta = sp.sympify('y - k*x')

        min_x = 1
        max_x = 20

        real_k = 2             # real 'k' value of source distribution

        err_y_avg = 0          # average of Y error values
        err_y_std = 0.01       # std of Y error values

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals, dtype=np.float)

        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs(
                    {sym_k: real_k}
                ),
                'numpy'
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        # get base values as averages of two half-length subgroups
        base_values_avg = {
            sym_x: (avg(x),),
            sym_y: (avg(y),)
        }

        # find params with basic method
        basic_k = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=(sym_k,),
            values=base_values_avg
        )

        # use basic estimates as init estimates for LSE
        for i, (lse_k,) in methods.search_lse2(
                expression=sym_expr,
                parameters=(sym_k,),
                values={sym_x: x},
                result_values={sym_y: y},
                init_estimates={sym_k: basic_k},
                num_iter=5
        ):
            lse_y = np.vectorize(
                sp.lambdify(
                    sym_x,
                    sym_expr.subs({sym_k: lse_k}),
                    'numpy'
                )
            )(x)

        self.assertAlmostEqual(real_k, lse_k[0], places=1)

    def test_linear_b(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_b = sp.symbols('b')
        sym_expr = sp.sympify('b')
        sym_expr_delta = sp.sympify('y - b')

        min_x = 1
        max_x = 20

        real_b = 2             # real 'b' value of source distribution

        err_y_avg = 0          # average of Y error values
        err_y_std = 0.01       # std of Y error values

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals, dtype=np.float)

        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs(
                    {sym_b: real_b}
                ),
                'numpy'
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        # get base values as averages of two half-length subgroups
        base_values_avg = {
            sym_x: (avg(x),),
            sym_y: (avg(y),)
        }

        # find params with basic method
        basic_b = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=(sym_b,),
            values=base_values_avg
        )

        # use basic estimates as init estimates for LSE
        for i, (lse_b,) in methods.search_lse2(
                expression=sym_expr,
                parameters=(sym_b,),
                values={sym_x: x},
                result_values={sym_y: y},
                init_estimates={sym_b: basic_b},
                num_iter=5
        ):
            lse_y = np.vectorize(
                sp.lambdify(
                    sym_x,
                    sym_expr.subs({sym_b: lse_b}),
                    'numpy'
                )
            )(x)

        self.assertAlmostEqual(real_b, lse_b[0], places=1)

    def test_exponential(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a = sp.symbols('a')
        sym_expr = sp.sympify('a*exp(x)')
        sym_expr_delta = sp.sympify('y - a*exp(x)')

        min_x = 1
        max_x = 20

        real_a = 10            # real 'a' value of source distribution

        err_y_avg = 0          # average of Y error values
        err_y_std = 0.01       # std of Y error values

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals, dtype=np.float)

        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs(
                    {sym_a: real_a}
                ),
                'numpy'
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        # get base values as averages of two half-length subgroups
        base_values_avg = {
            sym_x: (avg(x),),
            sym_y: (avg(y),)
        }

        # find params with basic method
        basic_a = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=(sym_a,),
            values=base_values_avg
        )

        # use basic estimates as init estimates for LSE
        for i, (lse_a,) in methods.search_lse2(
                expression=sym_expr,
                parameters=(sym_a,),
                values={sym_x: x},
                result_values={sym_y: y},
                init_estimates={sym_a: basic_a},
                num_iter=5
        ):
            lse_y = np.vectorize(
                sp.lambdify(
                    sym_x,
                    sym_expr.subs({sym_a: lse_a}),
                    'numpy'
                )
            )(x)

        self.assertAlmostEqual(real_a, lse_a[0], places=1)

    def test_sinusoidal(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a = sp.symbols('a')
        sym_expr = sp.sympify('a*sin(x)')
        sym_expr_delta = sp.sympify('y - a*sin(x)')

        min_x = 1
        max_x = 20

        real_a = 2             # real 'a' value of source distribution
        real_t = 0.5           # real 't' value of source distiribution

        err_y_avg = 0          # average of Y error values
        err_y_std = 0.01       # std of Y error values

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals, dtype=np.float)

        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs(
                    {sym_a: real_a}
                ),
                'numpy'
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        # get base values as averages of two half-length subgroups
        base_values_avg = {
            sym_x: (avg(x),),
            sym_y: (avg(y),)
        }

        # find params with basic method
        basic_a = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=(sym_a,),
            values=base_values_avg
        )

        # use basic estimates as init estimates for LSE
        for i, (lse_a,) in methods.search_lse2(
                expression=sym_expr,
                parameters=(sym_a,),
                values={sym_x: x},
                result_values={sym_y: y},
                init_estimates={sym_a: basic_a},
                num_iter=5
        ):
            lse_y = np.vectorize(
                sp.lambdify(
                    sym_x,
                    sym_expr.subs({sym_a: lse_a}),
                    'numpy'
                )
            )(x)

        self.assertAlmostEqual(real_a, lse_a[0], places=1)
