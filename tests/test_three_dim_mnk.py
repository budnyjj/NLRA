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

    def test_quadratic(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a, sym_b, sym_c = sp.symbols('a b c')
        sym_expr = sp.sympify('a*(x**2) + b*x + c')
        sym_expr_delta = sp.sympify('y - (a*(x**2) + b*x + c)')

        min_x = 1
        max_x = 20

        real_a = 2             # real 'a' value of source distribution
        real_b = 3             # real 'b' value of source distiribution
        real_c = 5             # real 'c' value of source distiribution

        err_y_avg = 0          # average of Y error values
        err_y_std = 0.01       # std of Y error values

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals, dtype=np.float)

        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_a: real_a,
                     sym_b: real_b,
                     sym_c: real_c}
                ),
                'numpy'
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        third_len = self.num_vals / 3

        # get base values as averages of two half-length subgroups
        base_values_avg = {
            sym_x: [avg(x[:third_len]),
                    avg(x[third_len:third_len * 2]),
                    avg(x[third_len * 2:])],
            sym_y: [avg(y[:third_len]),
                    avg(y[third_len:third_len * 2]),
                    avg(y[third_len * 2:])]
        }

        # find params with basic method
        basic_a, basic_b, basic_c = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=(sym_a, sym_b, sym_c),
            values=base_values_avg
        )

        # use basic estimates as init estimates for MNK
        for i, (mnk_a, mnk_b, mnk_c) in methods.search_mnk(
                expression=sym_expr,
                parameters=(sym_a, sym_b, sym_c),
                values={sym_x: x},
                result_values={sym_y: y},
                init_estimates={
                    sym_a: basic_a,
                    sym_b: basic_b,
                    sym_c: basic_c
                },
                num_iter=5
        ):
            mnk_y = np.vectorize(
                sp.lambdify(
                    sym_x,
                    sym_expr.subs({sym_a: mnk_a,
                                   sym_b: mnk_b,
                                   sym_c: mnk_c}),
                    'numpy'
                )
            )(x)

        self.assertAlmostEqual(real_a, mnk_a, places=1)
        self.assertAlmostEqual(real_b, mnk_b, places=1)
        self.assertAlmostEqual(real_c, mnk_c, places=1)
