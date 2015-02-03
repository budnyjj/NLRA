#!/usr/bin/env python2

import math
import random
import numpy as np
import matplotlib.pyplot as plt

#####################
# Utility functions #
#####################
def print_delimiter(width=40):
    print "-"*width, "\n"
    
def chunks_distant(lst):
    '''Yield pair of lst values with halflength distance.
    For example, (lst[0] and lst[len(lst)/2]), 
    (lst[1] and lst[len(lst)/2 + 1]) ...'''
    half_len = len(lst)/2
    for i in range(0, half_len):
        yield lst[i], lst[i+half_len]

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
def f(x, a, alpha):
    return a*math.exp(-alpha*x)

def diff_f_by_a(x, a, alpha):
    return math.exp(-alpha * x)

def diff_f_by_alpha(x, a, alpha):
    return -a*x*math.exp(-alpha * x)

#######################
# Statistical methods #
#######################
def f_search_basic(x, y):
    '''Search basic estimates by solving system of equations'''
    half = len(x) / 2
    
    x1 = avg(x[:half])
    x2 = avg(x[half:])

    y1 = avg(y[:half])
    y2 = avg(y[half:])
    
    alpha = math.log(y1/y2) / (x2 - x1)
    a = y1 / math.exp(-alpha*x1)
    return np.array([a, alpha])

def search_mnk(x, y, diff_funcs, init_values, num_iter=1):    
    '''Search estimates of ditribution parameters with MNK method.
    Function arguments:
        x, y        --- raw input values: y = f(x)
        diff_funcs  --- iterable of function f differentials
        init_values --- approximate values of distribution estimates
        num_iter    --- number of method iterations
    Yield values:
        cur_num_iter, estimates --- range of tuples contains target estimates
                                    by number of iteration
    '''

    y_t = np.array(y).T

    # init effective estimates with basic values
    cur_vals = init_values

    for i in range(num_iter):
        y_appr_t = np.array( map(lambda x: f(x, *cur_vals), x) ).T

        # construct Q from rows
        q_rows = []
        for diff_func in diff_funcs:
            diff_func_with_args = lambda x: diff_func(x, *cur_vals)
            q_rows.append(
                np.array(map(diff_func_with_args, x))
            )
        Q_t = np.vstack(q_rows)
        Q = Q_t.T
        
        # calculate addition = ((Q_t*Q)^-1)*Q_t*(y_t - appr_f_t)
        add = np.linalg.inv(np.dot(Q_t, Q))
        add = np.dot(add, Q_t)
        add = np.dot(add, y_t-y_appr_t)

        cur_vals += add
        yield i+1, cur_vals

def search_taylor(x, y, err_x_std, err_y_std):
    '''Search estimates of distribution parameters with Tayloar method.
    Arguments:
        x, y          --- raw input values with errors: y = f(x)
        err_x_std     --- standart deviation of X errors
        err_y_std     --- standart deviation of Y errors

    Return:
        (err_cnt,     --- number of errors due to incorrect input data,
                          singular matrix
         np.array)    --- array of estimates
    '''
    def alpha(x1, x2, y1, y2):
        return math.log(y1/y2) / (x2 - x1)

    def a(x1, x2, y1, y2):
        return y1 * math.exp(alpha(x1, x2, y1, y2) * x1)
    
    def diff_alpha_by_x1(x1, x2, y1, y2):
        return -math.log(y1/y2) / math.pow(x2 - x1, 2)
    
    def diff_alpha_by_x2(x1, x2, y1, y2):
        return math.log(y1/y2) / math.pow(x2 - x1, 2)

    def diff_alpha_by_y1(x1, x2, y1, y2):
        return 1 / (y1 * (x2 - x1))
    
    def diff_alpha_by_y2(x1, x2, y1, y2):
        return -1 / (y2 * (x2 - x1))
    
    def diff_a_by_x1(x1, x2, y1, y2):
        res = y1
        res *= math.exp(alpha(x1, x2, y1, y2) * x1)
        res *= alpha(x1, x2, y1, y2) + diff_alpha_by_x1(x1, x2, y1, y2) * x1
        return res

    def diff_a_by_x2(x1, x2, y1, y2):
        res = y2
        res *= math.exp(alpha(x1, x2, y1, y2) * x2)
        res *= alpha(x1, x2, y1, y2) + diff_alpha_by_x2(x1, x2, y1, y2) * x2
        return res

    def diff_a_by_y1(x1, x2, y1, y2):
        res = math.exp(alpha(x1, x2, y1, y2) * x1)
        res *= 1 + y1 * x1 * diff_alpha_by_y1(x1, x2, y1, y2)
        return res

    def diff_a_by_y2(x1, x2, y1, y2):
        res = math.exp(alpha(x1, x2, y1, y2) * x2)
        res *= 1 + y2 * x2 * diff_alpha_by_y2(x1, x2, y1, y2)
        return res    

    err_cnt = 0
    sum_Rinv = np.array([[0, 0],
                         [0, 0]])
    sum_Rinv_theta = np.array([[0, 0],
                               [0, 0]])

    for (x1, x2), (y1, y2) in zip(chunks_distant(x), chunks_distant(y)):
        # print "x1:", x1
        # print "x2:", x2
        # print "y1:", y1
        # print "y2:", y2
                
        try:
            G = np.array([
                [diff_a_by_x1(x1, x2, y1, y2),     diff_a_by_y1(x1, x2, y1, y2),     diff_a_by_x2(x1, x2, y1, y2),     diff_a_by_y2(x1, x2, y1, y2)],
                [diff_alpha_by_x1(x1, x2, y1, y2), diff_alpha_by_y1(x1, x2, y1, y2), diff_alpha_by_x2(x1, x2, y1, y2), diff_alpha_by_y2(x1, x2, y1, y2)]
            ])
            Dx = math.pow(err_x_std, 2)
            Dy = math.pow(err_y_std, 2)
            Rx = np.array([
                [Dx, 0, 0, 0],
                [0, Dy, 0, 0],
                [0, 0, Dx, 0],
                [0, 0, 0, Dy]
            ])            
            R = np.dot(np.dot(G, Rx), G.T)
            
            Rinv = np.linalg.inv(R)
            sum_Rinv = sum_Rinv + Rinv

            theta = np.array([[a(x1, x2, y1, y2)],
                              [alpha(x1, x2, y1, y2)]])
            Rinv_theta = np.dot(Rinv, theta)
            sum_Rinv_theta = sum_Rinv_theta + Rinv_theta

        except (ValueError, np.linalg.linalg.LinAlgError): # singular matrix, skip it
            err_cnt += 1
            continue
        # else:
        #     print "G:", G
        #     print "R:", R
        #     print "Theta:", theta
        #     print "Rinv:", Rinv
        #     print "Rinv_theta:", Rinv        
        #     print "sum_Rinv:", sum_Rinv
        #     print "sum_Rinv_theta", sum_Rinv_theta
        #     print

    res = np.dot(np.linalg.inv(sum_Rinv), sum_Rinv_theta)
    return (err_cnt, (res[0][0], res[1][0]))
    
if __name__ == '__main__':
    NUM_VALS = 100             # number of source values
    
    REAL_A = 30                # real "a" value of source distribution
    REAL_ALPHA = 0.005         # real "alpha" value of source distiribution
    
    ERR_X_AVG = 0              # average of X error values
    ERR_X_MIN_STD = 0.1        # minimal std of X error values
    ERR_X_MAX_STD = 2          # maximal std of X error values

    ERR_Y_AVG = 0              # average of Y error values
    ERR_Y_MIN_STD = 0.1        # minimal std of Y error values
    ERR_Y_MAX_STD = 2          # maximal std of Y error values

    ERR_NUM_STD_ITER = 10     # number of stds iterations  

    # real X values without errors
    real_x = np.linspace(0, 500, NUM_VALS ,dtype=np.float)

    # real Y values without errors
    real_y = np.array( map(lambda x: f(x, REAL_A, REAL_ALPHA), real_x) )
    
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
        print "Error X std:   {}".format(err_x_std)
        print "Error Y std:   {}".format(err_y_std)

        # add X errors with normal distribution
        x = np.array( map(lambda x: x + random.gauss(ERR_X_AVG, err_x_std), real_x) )
        
        # add Y errors with normal distribution
        y = np.array( map(lambda y: y + random.gauss(ERR_Y_AVG, err_y_std), real_y) )

        # plt.plot(x, y,
        #          color="b", linestyle=" ",
        #          marker=".", markersize=10, mfc="r")
        
        # find params with basic method
        basic_a, basic_alpha = f_search_basic(x, y)
        basic_y = map(lambda x: f(x, basic_a, basic_alpha), x) 
        basic_disp = disp(basic_y, y)
        basic_std = std(basic_y, y)
        basic_stds.append(basic_std)
        
        print "Basic a:       {}".format(basic_a)
        print "Basic alpha:   {}".format(basic_alpha)
        print "Dispersion:    {}".format(basic_disp)
        print "Std:           {}\n".format(basic_std)
        # plt.plot(x, basic_y,
        #          color="g", linestyle="-",
        #          marker=".", markersize=5, mfc="g")
        
        for i, (mnk_a, mnk_alpha) in search_mnk(x, y,
                                                diff_funcs=(diff_f_by_a,
                                                            diff_f_by_alpha),
                                                init_values=(basic_a,
                                                             basic_alpha),
                                                num_iter=1):
            print "MNK({}) a:      {}".format(i, mnk_a)
            print "MNK({}) alpha:  {}".format(i, mnk_alpha)
        
            mnk_y = map(lambda x: f(x, mnk_a, mnk_alpha), x)
            mnk_disp = disp(mnk_y, y)
            mnk_std = std(mnk_y, y)
            mnk_stds.append(mnk_std) # only if one iteration
            
            print "Dispersion:    {}".format(mnk_disp)
            print "Std:           {}\n".format(mnk_std)
        
            # plt.plot(x, mnk_y,
            #      color="b", linestyle="-",
            #      marker=".", markersize=5, mfc="b")

        # find params with taylor method
        taylor_err_cnt, (taylor_a, taylor_alpha) = search_taylor(x, y, err_x_std, err_y_std)
        taylor_y = map(lambda x: f(x, taylor_a, taylor_alpha), x) 
        taylor_disp = disp(taylor_y, y)
        taylor_std = std(taylor_y, y)
        taylor_stds.append(taylor_std)

        print "Taylor errors:  {}".format(taylor_err_cnt)
        print "Taylor a:       {}".format(taylor_a)
        print "Taylor alpha:   {}".format(taylor_alpha)
        print "Dispersion:    {}".format(taylor_disp)
        print "Std:           {}\n".format(taylor_std)
        
        # plt.plot(x, taylor_y,
        #          color="r", linestyle="-",
        #          marker=".", markersize=5, mfc="r")

            
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.grid(True)
    
    plt.plot(err_x_stds, basic_stds,
             color="g", linestyle="-",
             marker=".", markersize=5, mfc="g")
    plt.plot(err_x_stds, mnk_stds,
             color="b", linestyle="-",
             marker=".", markersize=5, mfc="b")
    plt.plot(err_x_stds, taylor_stds,
             color="r", linestyle="-",
             marker=".", markersize=5, mfc="r")
        
    plt.xlabel("D(Y)")
    plt.ylabel("ro")
    plt.grid(True)
        
    plt.show()
