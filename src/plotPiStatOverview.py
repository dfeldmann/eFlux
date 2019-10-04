#!/usr/bin/env python3

# Purpose:  read one-point statistics from ascii files and plot 
# Usage:    python statisticsPlot.py 
# Authors:  Daniel Feldmann
# Date:     12th July 2018
# Modified: 04th October 2019

import sys
import os.path
import timeit
import math
import numpy as np
import h5py

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# pdf output file name
fnamout = 'plotPiStatOverview.pdf'

# read velocity statistics from ascii file
fnam = '../../onePointStatistics/statistics00570000to01675000nt0222.dat'
print('Read velocity statistics from file', fnam, 'with')
r      = np.loadtxt(fnam)[:, 0] # radial coordinate
uzMean = np.loadtxt(fnam)[:, 3] # u_z mean
uzRms  = np.loadtxt(fnam)[:, 7] # u_z rms 

# read eflux statistics from ascii file
fnam = 'piStatGauss2d_pipe0002_00570000to01675000nt0222.dat'
print('Read eFlux statistics from file', fnam)
piMean  = np.loadtxt(fnam)[:, 1]

# grid size
nr  = len(r)
print(nr, 'radial (r) points')

# wall distance in plus units
yp = (1.0-r) * Re_tau

# plot data as graph, (1) interactive, (2) pdf
plot = 2
if plot not in [1, 2]: sys.exit("\nError: Set plot mode to 1 (interactiv) or 2 (pdf).") # skip everything below
print('Creating plot (using LaTeX)...')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
r"\usepackage[utf8]{inputenc}",
r"\usepackage[T1]{fontenc}",
r"\usepackage[detect-all]{siunitx}",
r'\usepackage{amsmath, amstext, amssymb}',
r'\usepackage{mathptmx}',
#r'\usepackage{lmodern, palatino, eulervm}',
r'\usepackage{xfrac}']
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'font.size': 9})

# create figure suitable for A4 format
def mm2inch(*tupl):
 inch = 25.4
 if isinstance(tupl[0], tuple):
  return tuple(i/inch for i in tupl[0])
 else:
  return tuple(i/inch for i in tupl)
fig = plt.figure(num=None, figsize=mm2inch(86.0, 65.0), dpi=150)

# line colours appropriate for colour-blind
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Grey          = '#999999'
Black         = '#000000'

# normalise
fuz  = np.amax(uzMean)
fpi  = np.amax(piMean)
frms = np.amax(uzRms)

# define zero line
zero = np.zeros(len(yp[:-1]))

# plot mean and rms in arbitrary units
ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
ax1.set_xlabel(r"Viscous wall distance")
ax1.set_xscale('log')
ax1.set_xlim(left=6.0e-1, right=Re_tau)
ax1.set_ylabel(r"Arbitrary unit", labelpad=-5.5)
ax1.set_ylim(bottom=-0.19, top=1.04)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax1.fill_between(yp[:-1], piMean[:-1]/fpi, zero, where=piMean[:-1]/fpi<=0.15, color=Vermillion, label=r"$\langle \Pi\rangle <0$")
ax1.fill_between(yp[:-1], piMean[:-1]/fpi, zero, where=piMean[:-1]/fpi>=-0.05, color=Blue, label=r"$\langle \Pi\rangle >0$")
ax1.plot(yp[:-1], uzMean[:-1]/fuz, color=Black, linestyle='-',  label=r"$\langle u_{z}\rangle$")
ax1.plot(yp[:-1], uzRms[:-1]/frms, color=Black, linestyle='--',  label=r"$\langle u_{z}^{\prime 2}\rangle$")
ax1.legend(bbox_to_anchor=(0.0, 0.0, 2.90e-1, 0.82), frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# modify axes and stuff
ax1.spines['left'].set_color('none')                               # remove left axis
ax1.spines['right'].set_position(('data', 1.0+2.30102999566398))   # modify position of right axis
ax1.spines['right'].set_position(('data', 1.0+2.25623653320592))   # modify position of right axis
ax1.yaxis.set_ticks_position('right')                              # switch ticks to right axis
ax1.yaxis.set_label_position("right")                              # switich label to right axis
ax1.spines['bottom'].set_position(('data', 0.0))
ax1.spines['top'].set_color('none')

# buffer layer annotation
ax1.axvspan(5.0, 30.0, ymin=0.049, alpha=1.0, color=Grey, zorder=0)
ax1.text(6.0, 0.18, r"Buffer layer")
ax1.annotate(s='', xy=(5.0, 0.15), xytext=(30.0, 0.15), arrowprops=dict(arrowstyle='<->'))

# plot mode interactive (1) or pdf (2)
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 #fnam = fnam.replace(".dat", "viscous.pdf")
 plt.savefig(fnamout)
 print('Written file:', fnamout)

fig.clf()
print("Done!")
