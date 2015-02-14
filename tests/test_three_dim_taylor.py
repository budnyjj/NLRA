import unittest

import random
import sympy as sp
import numpy as np


import sys
import os
sys.path.append('.')

import stats.methods as methods
from stats.utils import *

class TestBasicTaylor(unittest.TestCase):
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
        real_c = 3             # real 'c' value of source distiribution

        err_y_avg = 0          # average of Y error values
        err_y_std = 0.01       # std of Y error values
        
        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals ,dtype=np.float)
        
        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_a: real_a,
                     sym_b: real_b,
                     sym_c: real_c}
                )
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        # find params with mrt method
        mrt_a, mrt_b, mrt_c = methods.search_mrt(
            delta_expression=sym_expr_delta,
            parameters=(sym_a, sym_b, sym_c),
            values={sym_x: x, sym_y: y},
            err_stds={sym_x: 0, sym_y: err_y_std}
        )

        mrt_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs({sym_a: mrt_a,
                               sym_b: mrt_b,
                               sym_c: mrt_c})
            )
        )(x)

        self.assertAlmostEqual(real_a, mrt_a, places=1)
        self.assertAlmostEqual(real_b, mrt_b, places=1)
        self.assertAlmostEqual(real_c, mrt_c, places=1)
