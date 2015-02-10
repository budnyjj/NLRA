import unittest
import sympy as sp
import numpy as np

import sys
import os
sys.path.append('.')

import stats.methods as methods
from stats.utils import *

class TestBasicSearch(unittest.TestCase):
    def setUp(self):
        self.num_vals = 20          # number of source values

    def test_quadratic(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a, sym_b, sym_c = sp.symbols('a b c')
        sym_expr = sp.sympify('a*(x**2) + b*x + c')
        sym_expr_delta = sp.sympify('y - (a*(x**2) + b*x + c)')

        min_x = 1
        max_x = 20 

        real_a = 2            # real 'a' value of source distribution
        real_b = 3            # real 'b' value of source distiribution
        real_c = 5            # real 'c' value of source distiribution

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals ,dtype=np.float)
        
        # real Y values without errors
        y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_a: real_a,
                     sym_b: real_b,
                     sym_c: real_c}
                )
            )
        )(x)

        third_len = self.num_vals / 3
        
        # get base values as half-distant pairs of values
        base_values_dist = {
            sym_x: [x[0], x[third_len], x[third_len*2]],
            sym_y: [y[0], y[third_len], y[third_len*2]]
        }

        # find params with basic method
        basic_a, basic_b, basic_c = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=(sym_a, sym_b, sym_c),
            values=base_values_dist
        )        

        self.assertAlmostEqual(real_a, basic_a, places=5)
        self.assertAlmostEqual(real_b, basic_b, places=5)
        self.assertAlmostEqual(real_c, basic_c, places=5)
