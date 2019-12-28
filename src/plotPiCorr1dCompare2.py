#!/usr/bin/env python3
# Purpose:  Read pre-computed 1d cross-correlations between energy flux and
#           sweep (Q4) events. Plot azimtuhal (theta) and axial (z) correlation
#           factors based on different filter kernels (Fourier, Gauss, box) to
#           compare the influence of the kernel. Output is interactive and as
#           pfd figure file.
# Usage:    python plotPiCorr1dQ4Compare.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 23rd August 2019

import sys
import os.path
import timeit
import math
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
print('Plot 1d cross-correlations between energy flux Q4 events based on different kernels.')
plot = int(input("Enter plot mode (0 = none, 1 = interactive, 2 = pdf file): "))

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# read axial correlation with sweep (Q4) events for Fourier filtered eFlux from ascii file
fnam = 'piCorrZQ4Q2Fourier2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading axial correlation for Fourier filtered eFlux', fnam)
z = np.loadtxt(fnam)[:, 0] # axial separation only once
czF = np.loadtxt(fnam)[:, 2]

# read axial correlation with sweep (Q4) events for Gauss filtered eFlux from ascii file
fnam = 'piCorrZQ4Q2Gauss2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading axial correlation for Gauss filtered eFlux', fnam)
czG = np.loadtxt(fnam)[:, 2]

# read axial correlation with sweep (Q4) events for Box filtered eFlux from ascii file
fnam = 'piCorrZQ4Q2Box2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading axial correlation for Box filtered eFlux', fnam)
czB = np.loadtxt(fnam)[:, 2]

# read azimuthal correlation with sweep (Q4) events for Fourier filtered eFlux from ascii file
fnam = 'piCorrThQ4Q2Fourier2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading axial correlation for Fourier filtered eFlux', fnam)
th = np.loadtxt(fnam)[:, 0] # azimuthal separation only once
cthF = np.loadtxt(fnam)[:, 2]

# read azimuthal correlation with sweep (Q4) events for Gauss filtered eFlux from ascii file
fnam = 'piCorrThQ4Q2Gauss2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading axial correlation for Gauss filtered eFlux', fnam)
cthG = np.loadtxt(fnam)[:, 2]

# read azimuthal correlation with sweep (Q4) events for Box filtered eFlux from ascii file
fnam = 'piCorrThQ4Q2Box2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading axial correlation for Box filtered eFlux', fnam)
cthB = np.loadtxt(fnam)[:, 2]
# if you change the last occurence of fnam, also change the string replace for pdf file name below

# grid size
nth = len(th)
nz  = len(z)
print('With', nth, 'azimuthal (theta) and', nz, 'axial (z) grid points. It is your responsibility to')
print('make sure, that all data sets are defined on the exact same grids.')

if plot not in [1, 2]: sys.exit() # skip everything below
print('Creating plot (using LaTeX)...')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
r"\usepackage[utf8x]{inputenc}",
r"\usepackage[T1]{fontenc}",
r"\usepackage[detect-all]{siunitx}",
r'\usepackage{amsmath, amstext, amssymb}',
r'\usepackage{xfrac}',
r'\usepackage{lmodern, palatino, eulervm}']
mpl.rcParams.update({'font.family': 'sans-serif'})
mpl.rcParams.update({'font.family': 'serif'})
mpl.rcParams.update({'font.size': 5})
mpl.rcParams.update({'lines.linewidth': 0.5})
mpl.rcParams.update({'axes.linewidth': 0.5})
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['xtick.minor.size'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['ytick.minor.size'] = 1
mpl.rcParams['ytick.minor.width'] = 0.5

# create figure suitable for A4 format
def mm2inch(*tupl):
 inch = 25.4
 if isinstance(tupl[0], tuple):
  return tuple(i/inch for i in tupl[0])
 else:
   return tuple(i/inch for i in tupl)
fig = plt.figure(num=None, figsize=mm2inch(110.0, 45.0), dpi=300)

# line colours appropriate for colour-blind
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Black         = '#000000'
Grey          = '#999999'

# convert spatial separation from outer to inner units
th = th * Re_tau
z  =  z * Re_tau

# plot first order statistics
ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
ax1.set_xlabel(r"$r^+\Delta\theta$")
#ax1.set_xlim(left=6.0e-1, right=Re_tau)
#ax1.set_xticks([])
ax1.set_ylabel(r"$C_{\Pi Q_4}$")
ax1.set_ylim(bottom=-0.33, top=0.11)
ax1.axhline(y=0.0, color=Grey)
ax1.axvline(x=0.0, color=Grey)
ax1.plot(th, cthF, color=Black,      linestyle='-', zorder=7, label=r"Fourier")
ax1.plot(th, cthG, color=Vermillion, linestyle='-', zorder=8, label=r"Gauss")
ax1.plot(th, cthB, color=Blue,       linestyle='-', zorder=9, label=r"Box")
#ax1.legend(loc='best', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot second order statistics
ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
ax2.set_xlabel(r"$\Delta z^+$")
ax2.set_xlim(left=-1000.0, right=1000.0)
#ax2.set_xticks([])
#ax2.set_ylabel(r"$\text{RMS}\left(\Pi^{\prime}\right)$ in $ $")
ax2.set_ylim(bottom=-0.33, top=0.11)
ax2.set_yticks([])
ax2.axhline(y=0.0, color=Grey)
ax2.axvline(x=0.0, color=Grey)
ax2.plot(z, czF, color=Black,      linestyle='-', zorder=7, label=r"Fourier")
ax2.plot(z, czG, color=Vermillion, linestyle='-', zorder=8, label=r"Gauss")
ax2.plot(z, czB, color=Blue,       linestyle='-', zorder=9, label=r"Box")
ax2.legend(loc='best', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot mode interactive or pdf
if plot == 1:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = str.replace(fnam, '.dat', '.pdf')
 fnam = str.replace(fnam, 'piCorrThQ4Q2Box2d', 'plotPiCorr1dCompare')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
