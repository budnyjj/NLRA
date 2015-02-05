#!/usr/bin/env python

import math
import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import methods

#######################
# Statistical helpers #
#######################    
def avg(vals):
    return sum(vals)/len(vals)

def disp(y_1, y_2):
    '''Calculate sum of differences between f[] and y[] values'''
    total = 0
    for i in range(min(len(y_1),
                       len(y_2))):
        total += math.pow(y_1[i] - y_2[i], 2)

    return total

def std(y_1, y_2):
    '''Return sqrt of dispersion'''
    return math.sqrt(disp(y_1, y_2))

#############
# Functions #
#############
sym_x, sym_y = sym_values = sp.symbols('x y')
sym_a, sym_alpha = sym_params = sp.symbols('a alpha')

sym_expr = sp.sympify('a * exp(-alpha*x)')
sym_expr_delta = sp.sympify('y - a * exp(-alpha*x)')

if __name__ == '__main__':
    NUM_VALS = 20              # number of source values
    
    REAL_A = 50                # real 'a' value of source distribution
    REAL_ALPHA = 0.005         # real 'alpha' value of source distiribution
    
    ERR_X_AVG = 0              # average of X error values
    ERR_X_MIN_STD = 0.1        # minimal std of X error values
    ERR_X_MAX_STD = 1.8        # maximal std of X error values

    ERR_Y_AVG = 0              # average of Y error values
    ERR_Y_MIN_STD = 0.1        # minimal std of Y error values
    ERR_Y_MAX_STD = 1.8        # maximal std of Y error values

    ERR_NUM_STD_ITER = 10      # number of stds iterations  

    # real X values without errors
    real_x = np.linspace(0, 500, NUM_VALS ,dtype=np.float)
    
    # real Y values without errors
    real_y = np.vectorize(
        sp.lambdify(sym_x,
                    sym_expr.subs({sym_a: REAL_A, sym_alpha: REAL_ALPHA})
        )
    )(real_x)
    
    # generate array of X error stds
    err_x_stds = np.linspace(ERR_X_MIN_STD, ERR_X_MAX_STD, ERR_NUM_STD_ITER)

    # generate array of Y error stds
    err_y_stds = np.linspace(ERR_Y_MIN_STD, ERR_Y_MAX_STD, ERR_NUM_STD_ITER)
        
    # collect dispersions of estimates
    basic_stds = []
    mnk_stds = []
    taylor_stds = []
    
    # iterate by error standart derivation values
    for err_x_std, err_y_std in zip(err_x_stds, err_y_stds):
        print('Error X std:   {}'.format(err_x_std))
        print('Error Y std:   {}\n'.format(err_y_std))

        # add X errors with current normal distribution
        x = np.vectorize(
            lambda v: v + random.gauss(ERR_X_AVG, err_x_std)
        )(real_x)

        # add Y errors with current normal distribution
        y = np.vectorize(
            lambda v: v + random.gauss(ERR_Y_AVG, err_y_std)
        )(real_y)

        # plot real values
        # plt.plot(x, y,
        #          color='b', linestyle=' ',
        #          marker='.', markersize=10, mfc='r')

        # get base values as first pairs of values
        base_values_first = ([x[0], x[1]],
                             [y[0], y[1]]
        )

        half_len = len(x) / 2
        
        # get base values as half-distant pairs of values
        base_values_first = ([x[0], x[half_len]],
                             [y[0], y[half_len]]
        )

        # get base values as averages of two half-length subgroups
        base_values_dist = ([avg(x[:half_len]), avg(x[half_len:])],
                            [avg(y[:half_len]), avg(y[half_len:])]
        )
        
        ################
        # Basic search #
        ################
        
        # find params with basic method
        basic_a, basic_alpha = methods.search_basic(sym_expr_delta,
                                                    (sym_a, sym_alpha),
                                                    (sym_x, sym_y),
                                                    base_values_first,
                                                    (1, 0))
        basic_y = np.vectorize(
            sp.lambdify(sym_x,
                        sym_expr.subs({sym_a: basic_a, sym_alpha: basic_alpha})
                    )
        )(real_x)
        basic_disp = disp(basic_y, y)
        basic_std = std(basic_y, y)
        basic_stds.append(basic_std)
        
        print('Basic a:       {}'.format(basic_a))
        print('Basic alpha:   {}'.format(basic_alpha))
        print('Dispersion:    {}'.format(basic_disp))
        print('Std:           {}\n'.format(basic_std))
        # plt.plot(x, basic_y,
        #          color='g', linestyle='-',
        #          marker='.', markersize=5, mfc='g')
        
        ##############
        # MNK search #
        ##############
        for i, (mnk_a, mnk_alpha) in methods.search_mnk(
                sym_expr,
                (sym_a, sym_alpha),
                (sym_x),
                (real_x),
                sym_y,
                real_y,
                (basic_a, basic_alpha)):
            mnk_y = np.vectorize(
                sp.lambdify(sym_x,
                            sym_expr.subs({sym_a: mnk_a,
                                           sym_alpha: mnk_alpha})
                        )
            )(real_x)
            mnk_disp = disp(mnk_y, y)
            mnk_std = std(mnk_y, y)
            mnk_stds.append(mnk_std) # only if one iteration

            print('MNK({}) a:      {}'.format(i, mnk_a))
            print('MNK({}) alpha:  {}'.format(i, mnk_alpha))
            print('Dispersion:    {}'.format(mnk_disp))
            print('Std:           {}\n'.format(mnk_std))
        
            # plt.plot(x, mnk_y,
            #      color='b', linestyle='-',
            #      marker='.', markersize=5, mfc='b')

        # find params with taylor method
        taylor_err_cnt, (taylor_a, taylor_alpha) = methods.search_taylor(
            sym_expr_delta,
            (sym_a, sym_alpha),
            (sym_x, sym_y),
            (real_x, real_y),
            (err_x_std, err_y_std)
        )
        
        taylor_y = np.vectorize(
                sp.lambdify(sym_x,
                            sym_expr.subs({sym_a: taylor_a,
                                           sym_alpha: taylor_alpha})
                        )
        )(real_x)
        taylor_disp = disp(taylor_y, y)
        taylor_std = std(taylor_y, y)
        taylor_stds.append(taylor_std)

        print('Taylor errors:  {}'.format(taylor_err_cnt))
        print('Taylor a:       {}'.format(taylor_a))
        print('Taylor alpha:   {}'.format(taylor_alpha))
        print('Dispersion:    {}'.format(taylor_disp))
        print('Std:           {}\n'.format(taylor_std))
        
        # plt.plot(x, taylor_y,
        #          color='r', linestyle='-',
        #          marker='.', markersize=5, mfc='r')

    plt.plot(err_x_stds, basic_stds,
             color='g', linestyle='-',
             marker='.', markersize=5, mfc='g')
    plt.plot(err_x_stds, mnk_stds,
             color='b', linestyle='-',
             marker='.', markersize=5, mfc='b')
    plt.plot(err_x_stds, taylor_stds,
             color='r', linestyle='-',
             marker='.', markersize=5, mfc='r')
        
    plt.xlabel('D(Y)')
    plt.ylabel('ro')
    # plt.grid(True)

    # plt.xlabel('x')
    # plt.ylabel('y')

    plt.grid(True)
    plt.show()
