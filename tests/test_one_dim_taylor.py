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
                        self.num_vals ,dtype=np.float)

        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_k: real_k}
                )
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        # find params with taylor method
        taylor_k = methods.search_taylor(
            delta_expression=sym_expr_delta,
            parameters=[sym_k],
            values={sym_x: x, sym_y: y},
            err_stds={sym_x: 0, sym_y: err_y_std}
        )

        taylor_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs({sym_k: taylor_k})
            )
        )(x)

        self.assertAlmostEqual(real_k, taylor_k, places=1)


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
                        self.num_vals ,dtype=np.float)

        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_b: real_b}
                )
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        # find params with taylor method
        taylor_b = methods.search_taylor(
            delta_expression=sym_expr_delta,
            parameters=[sym_b],
            values={sym_x: x, sym_y: y},
            err_stds={sym_x: 0, sym_y: err_y_std}
        )

        taylor_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs({sym_b: taylor_b})
            )
        )(x)

        self.assertAlmostEqual(real_b, taylor_b, places=1)

        
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
                        self.num_vals ,dtype=np.float)

        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_a: real_a}
                )
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        # find params with taylor method
        taylor_a = methods.search_taylor(
            delta_expression=sym_expr_delta,
            parameters=[sym_a],
            values={sym_x: x, sym_y: y},
            err_stds={sym_x: 0, sym_y: err_y_std}
        )

        taylor_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs({sym_a: taylor_a})
            )
        )(x)

        self.assertAlmostEqual(real_a, taylor_a, places=1)

        
    def test_sinusoidal(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a = sp.symbols('a')
        sym_expr = sp.sympify('a*sin(x)')
        sym_expr_delta = sp.sympify('y - a*sin(x)')

        min_x = 1
        max_x = 20 

        real_a = 2             # real 'a' value of source distribution

        err_y_avg = 0          # average of Y error values
        err_y_std = 0.01       # std of Y error values

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals ,dtype=np.float)

        # real Y values without errors
        real_y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_a: real_a}
                )
            )
        )(x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(err_y_avg, err_y_std)
        )(real_y)

        # find params with taylor method
        taylor_a = methods.search_taylor(
            delta_expression=sym_expr_delta,
            parameters=[sym_a],
            values={sym_x: x, sym_y: y},
            err_stds={sym_x: 0, sym_y: err_y_std}
        )

        taylor_y = np.vectorize(
            sp.lambdify(
                sym_x,
                sym_expr.subs({sym_a: taylor_a})
            )
        )(x)

        self.assertAlmostEqual(real_a, taylor_a, places=1)
