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

    def test_linear(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_k, sym_b = sp.symbols('k b')
        sym_expr = sp.sympify('k*x + b')
        sym_expr_delta = sp.sympify('y - (k*x + b)')

        min_x = 1
        max_x = 20 

        real_k = 2             # real 'k' value of source distribution
        real_b = 10            # real 'b' value of source distiribution

        err_y_avg = 0          # average of Y error values
        err_y_std = 0.01       # std of Y error values
        
        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals ,dtype=np.float)
        
        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_k: real_k,
                     sym_b: real_b}
                )
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        # find params with mrt method
        mrt_k, mrt_b = methods.search_mrt(
            delta_expression=sym_expr_delta,
            parameters=(sym_k, sym_b),
            values={sym_x: x, sym_y: y},
            err_stds={sym_x: 0, sym_y: err_y_std}
        )

        mrt_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs({sym_k: mrt_k,
                               sym_b: mrt_b})
            )
        )(x)

        self.assertAlmostEqual(real_k, mrt_k, places=1)
        self.assertAlmostEqual(real_b, mrt_b, places=1)


    def test_exponential(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a, sym_alpha = sp.symbols('a alpha')
        sym_expr = sp.sympify('a + alpha*exp(x)')
        sym_expr_delta = sp.sympify('y - (a + alpha*exp(x))')

        min_x = 1
        max_x = 20 

        real_a = 10            # real 'a' value of source distribution
        real_alpha = 0.01      # real 'alpha' value of source distiribution

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
                     sym_alpha: real_alpha}
                )
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)
                
        # find params with mrt method
        mrt_a, mrt_alpha = methods.search_mrt(
            delta_expression=sym_expr_delta,
            parameters=(sym_a, sym_alpha),
            values={sym_x: x, sym_y: y},
            err_stds={sym_x: 0, sym_y: err_y_std}
        )

        mrt_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs({sym_a: mrt_a,
                               sym_alpha: mrt_alpha})
            )
        )(x)

        self.assertAlmostEqual(real_a, mrt_a, places=1)
        self.assertAlmostEqual(real_alpha, mrt_alpha, places=1)

        
    def test_sinusoidal(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a, sym_t = sp.symbols('a t')
        sym_expr = sp.sympify('a + t*sin(x)')
        sym_expr_delta = sp.sympify('y - (a + t*sin(x))')

        min_x = 1
        max_x = 20 

        real_a = 2             # real 'a' value of source distribution
        real_t = 0.5           # real 't' value of source distiribution

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
                     sym_t: real_t}
                )
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)
        
        # find params with mrt method
        mrt_a, mrt_t = methods.search_mrt(
            delta_expression=sym_expr_delta,
            parameters=(sym_a, sym_t),
            values={sym_x: x, sym_y: y},
            err_stds={sym_x: 0, sym_y: err_y_std}
        )

        mrt_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs({sym_a: mrt_a,
                               sym_t: mrt_t})
            )
        )(x)

        self.assertAlmostEqual(real_a, mrt_a, places=1)
        self.assertAlmostEqual(real_t, mrt_t, places=1)
