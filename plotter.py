#!/usr/bin/env python

import os.path
import argparse
import numpy as np

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


################
# Constants    #
################

DESCRIPTION = 'Plots estimate accuracies'

################
# Program code #
################

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    '-i', '--input', metavar='PATH',
    type=str, required=True,
    help='base path to read data')
parser.add_argument(
    '-o', '--output', metavar='PATH',
    type=str, help='base path to write plots')
parser.add_argument(
    '-s', '--show',
    dest='show', action='store_true',
    help='show plots')
parser.set_defaults(show=False)
args = parser.parse_args()
input_path, _ = os.path.splitext(args.input)
output_path, output_ext = None, None
if args.output:
    output_path, output_ext = os.path.splitext(args.output)

print('Input path:       {}'.format(input_path))
print('Output path:      {}'.format(output_path))

# load data
err_stds_x = np.load(
    '{}_err-stds-x.npy'.format(input_path))
err_stds_y = np.load(
    '{}_err-stds-y.npy'.format(input_path))
lse_param_accs = np.load(
    '{}_lse-accs.npy'.format(input_path))
mrt_param_accs = np.load(
    '{}_mrt-accs.npy'.format(input_path))

# compute differences between accuracies
print('Avg(d_LSE):       {}'.format(np.average(lse_param_accs)))
print('Avg(d_MRT):       {}'.format(np.average(mrt_param_accs)))

param_accs_diff = lse_param_accs - mrt_param_accs
print('Avg(d_LSE-d_MRT): {}'.format(np.average(param_accs_diff)))

plt.figure(0)
contour_param = plt.contour(
    err_stds_x, err_stds_y, param_accs_diff,
    colors='black')
# plt.title('$ d_{param_{LSE}} - d_{param_{MRT}} $')
plt.clabel(contour_param, inline=True, fontsize=10)
plt.xlabel('$ \sigma_{\epsilon_x} $')
plt.ylabel('$ \sigma_{\epsilon_y} $')

if args.output:
    plt.savefig(
        '{}{}'.format(output_path, output_ext),
        dpi=200)

if args.show:
    plt.show()
