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

        
    def test_linear_k(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_k = sp.symbols('k')
        sym_expr = sp.sympify('k*x')
        sym_expr_delta = sp.sympify('y - k*x')

        min_x = 1
        max_x = 20 

        real_k = 2             # real 'k' value of source distribution

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals ,dtype=np.float)
        
        # real Y values without errors
        y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_k: real_k}
                )
            )
        )(x)
        
        # get base values as half-distant pairs of values
        base_values_dist = {
            sym_x: [x[0]],
            sym_y: [y[0]]
        }

        # find params with basic method
        basic_k = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=[sym_k],
            values=base_values_dist
        )        
        
        self.assertAlmostEqual(real_k, basic_k, places=5)

        
    def test_linear_b(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_b = sp.symbols('b')
        sym_expr = sp.sympify('b')
        sym_expr_delta = sp.sympify('y - b')

        min_x = 1
        max_x = 20 

        real_b = 2             # real 'b' value of source distribution

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals ,dtype=np.float)
        
        # real Y values without errors
        y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_b: real_b}
                )
            )
        )(x)
        
        # get base values as half-distant pairs of values
        base_values_dist = {
            sym_x: [x[0]],
            sym_y: [y[0]]
        }

        # find params with basic method
        basic_b = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=[sym_b],
            values=base_values_dist
        )        
        
        self.assertAlmostEqual(real_b, basic_b, places=5)

    def test_exponential(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a = sp.symbols('a')
        sym_expr = sp.sympify('a*exp(x)')
        sym_expr_delta = sp.sympify('y - a*exp(x)')

        min_x = 1
        max_x = 20 

        real_a = 2             # real 'k' value of source distribution

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals ,dtype=np.float)
        
        # real Y values without errors
        y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_a: real_a}
                )
            )
        )(x)
        
        # get base values as half-distant pairs of values
        base_values_dist = {
            sym_x: [x[0]],
            sym_y: [y[0]]
        }

        # find params with basic method
        basic_a = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=[sym_a],
            values=base_values_dist
        )        
        
        self.assertAlmostEqual(real_a, basic_a, places=5)

        
    def test_sinusoidal(self):
        sym_x, sym_y = sp.symbols('x y')
        sym_a = sp.symbols('a')
        sym_expr = sp.sympify('a*sin(x)')
        sym_expr_delta = sp.sympify('y - a*sin(x)')

        min_x = 1
        max_x = 20 

        real_a = 2             # real 'k' value of source distribution

        # real X values without errors
        x = np.linspace(min_x, max_x,
                        self.num_vals ,dtype=np.float)
        
        # real Y values without errors
        y = np.vectorize(
            sp.lambdify(
                sym_x, sym_expr.subs(
                    {sym_a: real_a}
                )
            )
        )(x)
        
        # get base values as half-distant pairs of values
        base_values_dist = {
            sym_x: [x[0]],
            sym_y: [y[0]]
        }

        # find params with basic method
        basic_a = methods.search_basic(
            delta_expression=sym_expr_delta,
            parameters=[sym_a],
            values=base_values_dist
        )        
        
        self.assertAlmostEqual(real_a, basic_a, places=5)

