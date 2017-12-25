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

    def test_linear(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_k, sym_b = sp.symbols('k b')
        sym_expr = sp.sympify('k*x + b')
        sym_expr_delta = sp.sympify('y - (k*x + b)')

        min_x = 1
        max_x = 20

        real_k = 2             # real 'k' value of source distribution
        real_b = 10            # real 'b' value of source distiribution

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals, dtype=np.float)

        # real Y values without errors
        y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_k: real_k,
                     sym_b: real_b}
                ),
                'numpy'
            )
        )(x)

        half_len = int(self.num_vals / 2)

        # get base values as half-distant pairs of values
        base_values_dist = {
            sym_x: [x[0], x[half_len]],
            sym_y: [y[0], y[half_len]]
        }

        # find params with basic method
        basic_k, basic_b = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=(sym_k, sym_b),
            values=base_values_dist
        )

        self.assertAlmostEqual(real_k, basic_k, places=5)
        self.assertAlmostEqual(real_b, basic_b, places=5)

    def test_exponential(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a, sym_alpha = sp.symbols('a alpha')
        sym_expr = sp.sympify('a + alpha*exp(x)')
        sym_expr_delta = sp.sympify('y - (a + alpha*exp(x))')

        min_x = 1
        max_x = 20

        real_a = 10            # real 'a' value of source distribution
        real_alpha = 0.01      # real 'alpha' value of source distiribution

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals, dtype=np.float)

        # real Y values without errors
        y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_a: real_a,
                     sym_alpha: real_alpha}
                ),
                'numpy'
            )
        )(x)

        half_len = int(self.num_vals / 2)

        # get base values as half-distant pairs of values
        base_values_dist = {
            sym_x: [x[0], x[half_len]],
            sym_y: [y[0], y[half_len]]
        }

        # find params with basic method
        basic_a, basic_alpha = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=(sym_a, sym_alpha),
            values=base_values_dist
        )

        self.assertAlmostEqual(real_a, basic_a, places=5)
        self.assertAlmostEqual(real_alpha, basic_alpha, places=5)

    def test_sinusoidal(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a, sym_t = sp.symbols('a t')
        sym_expr = sp.sympify('a + t*sin(x)')
        sym_expr_delta = sp.sympify('y - (a + t*sin(x))')

        min_x = 1
        max_x = 20

        real_a = 2             # real 'a' value of source distribution
        real_t = 0.5           # real 't' value of source distiribution

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals, dtype=np.float)

        # real Y values without errors
        y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_a: real_a,
                     sym_t: real_t}
                ),
                'numpy'
            )
        )(x)

        half_len = int(self.num_vals / 2)

        # get base values as half-distant pairs of values
        base_values_dist = {
            sym_x: [x[0], x[half_len]],
            sym_y: [y[0], y[half_len]]
        }

        # find params with basic method
        basic_a, basic_t = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=(sym_a, sym_t),
            values=base_values_dist
        )

        self.assertAlmostEqual(real_a, basic_a, places=5)
        self.assertAlmostEqual(real_t, basic_t, places=5)
