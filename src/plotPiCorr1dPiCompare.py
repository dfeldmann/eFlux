#!/usr/bin/env python3
# Purpose:  Read pre-computed 1d auto-correlations for streamwise streaks
#           (u'_z), streamwise vorticity (omega_z), and energy flux. Plot
#           azimtuhal (theta) and axial (z) corelation factors base on different
#           filter kernels (Fourier, Gauss, box) to compare the influence of the
#           kernel. Output is interactive and as pfd figure file.
# Usage:    python plotPiCorr1dPiCompare.py
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 14th October 2019

import timeit
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
print('Plot 1d auto-correlations based on different kernels.')
plot = int(input("Enter plot mode (0 = none, 1 = interactive, 2 = pdf file): "))

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# read azimuthal auto-correlations for Fourier filtered eFlux from ascii file
fnam = 'piCorrThStreaksFourier2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading azimuthal auto correlations for Fourier filtered eFlux', fnam)
th     = np.loadtxt(fnam)[:, 0] # 1st column: Azimuthal separation DeltaTh
cuzth  = np.loadtxt(fnam)[:, 1] # 2nd column: Auto-correlation u'_z    with u'_z
cuzthF = np.loadtxt(fnam)[:, 2] # 3rd column: Auto-correlation u'_zF   with u'_zF
cozth  = np.loadtxt(fnam)[:, 3] # 4th column: Auto-correlation omega_z with omega_z
cpithF = np.loadtxt(fnam)[:, 4] # 5th column: Auto-correlation Pi      with Pi

# read azimuthal auto-correlations for Gauss filtered eFlux from ascii file
fnam = 'piCorrThStreaksGauss2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading azimuthal auto correlations for Gauss filtered eFlux', fnam)
cuzthG = np.loadtxt(fnam)[:, 2] # 3rd column: Auto-correlation u'_zF with u'_zF
cpithG = np.loadtxt(fnam)[:, 4] # 5th column: Auto-correlation Pi    with Pi

# read azimuthal auto-correlations for box filtered eFlux from ascii file
fnam = 'piCorrThStreaksBox2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading azimuthal auto correlations for box filtered eFlux', fnam)
cuzthB = np.loadtxt(fnam)[:, 2] # 3rd column: Auto-correlation u'_zF with u'_zF
cpithB = np.loadtxt(fnam)[:, 4] # 5th column: Auto-correlation Pi    with Pi

# read axial auto correlations Fourier filtered eFlux from ascii file
fnam = 'piCorrZStreaksFourier2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading axial auto correlations for Fourier filtered eFlux', fnam)
z     = np.loadtxt(fnam)[:, 0] # 1st column: Axial separation DeltaZ
cuzz  = np.loadtxt(fnam)[:, 1] # 2nd column: Auto-correlation  u'_z    with u'_z
cuzzF = np.loadtxt(fnam)[:, 2] # 3rd column: Auto-correlation  u'_zF   with u'_zF
cozz  = np.loadtxt(fnam)[:, 3] # 4th column: Auto-correlation  omega_z with omega_z
cpizF = np.loadtxt(fnam)[:, 4] # 5th column: Auto-correlation  Pi      with Pi

# read axial auto correlations Gauss filtered eFlux from ascii file
fnam = 'piCorrZStreaksGauss2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading axial auto correlations for Gauss filtered eFlux', fnam)
cuzzG = np.loadtxt(fnam)[:, 2] # 3rd column: Auto-correlation  u'_zF with u'_zF
cpizG = np.loadtxt(fnam)[:, 4] # 5th column: Auto-correlation  Pi    with Pi

# read axial auto correlations box filtered eFlux from ascii file
fnam = 'piCorrZStreaksBox2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading axial auto correlations for box filtered eFlux', fnam)
cuzzB = np.loadtxt(fnam)[:, 2] # 3rd column: Auto-correlation  u'_zF with u'_zF
cpizB = np.loadtxt(fnam)[:, 4] # 5th column: Auto-correlation  Pi    with Pi

# grid size
nth = len(th)
nz  = len(z)
print('With', nth, 'azimuthal (theta) grid points')
print('With', nz, 'axial (z) grid points.')
print('It is your responsibility to make sure, that all data sets are defined on the exact same grid.')

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
mpl.rcParams.update({'font.size': 6})
mpl.rcParams.update({'lines.linewidth': 0.75})
mpl.rcParams.update({'axes.linewidth': 0.75})
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['xtick.major.width'] = 0.75
mpl.rcParams['xtick.minor.size'] = 1
mpl.rcParams['xtick.minor.width'] = 0.75
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['ytick.major.width'] = 0.75
mpl.rcParams['ytick.minor.size'] = 1
mpl.rcParams['ytick.minor.width'] = 0.75

# create figure suitable for A4 format
def mm2inch(*tupl):
 inch = 25.4
 if isinstance(tupl[0], tuple):
  return tuple(i/inch for i in tupl[0])
 else:
   return tuple(i/inch for i in tupl)
fig = plt.figure(num=None, figsize=mm2inch(110.0, 120.0), dpi=300)

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

# plot azimuthal auto-correlation
ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
ax1.set_xlim(left=-200.0, right=200.0)
ax1.set_xticks([-200.0, -100.0, 0.0, 100.0, 200.0])
ax1.set_xticks([])
ax1.set_ylabel(r"$C_{u^{\prime}_{z} u^{\prime}_{z}}$")
ax1.set_ylim(bottom=-0.3, top=1.0)
ax1.axhline(y=0.0, color=Grey)
ax1.axvline(x=0.0, color=Grey)
ax1.plot(th, cuzth,  color=Black,      linestyle='--', zorder=6, label=r"Unfiltered")
ax1.plot(th, cuzthF, color=Black,      linestyle='-',  zorder=7, label=r"Fourier")
ax1.plot(th, cuzthG, color=Vermillion, linestyle='-',  zorder=8, label=r"Gauss")
ax1.plot(th, cuzthB, color=Blue,       linestyle='-',  zorder=9, label=r"Box")

# plot axial auto-correlation
ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1)
ax2.set_xlim(left=-1000.0, right=1000.0)
ax2.set_xticks([-1000.0, -500.0, 0.0, 500.0, 1000.0])
ax2.set_xticklabels([])
ax2.set_ylim(bottom=-0.3, top=1.0)
ax2.set_yticklabels([])
ax2.axhline(y=0.0, color=Grey)
ax2.axvline(x=0.0, color=Grey)
ax2.plot(z, cuzz,  color=Black,      linestyle='--', zorder=6, label=r"Unfiltered")
ax2.plot(z, cuzzF, color=Black,      linestyle='-',  zorder=7, label=r"Fourier")
ax2.plot(z, cuzzG, color=Vermillion, linestyle='-',  zorder=8, label=r"Gauss")
ax2.plot(z, cuzzB, color=Blue,       linestyle='-',  zorder=9, label=r"Box")
ax2.legend(loc='lower left', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot azimuthal auto-correlation
ax3 = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1)
ax3.set_xlim(left=-200.0, right=200.0)
ax3.set_xticks([-200.0, -100.0, 0.0, 100.0, 200.0])
ax3.set_xticklabels([])
ax3.set_ylabel(r"$C_{\omega_{z} \omega_{z}}$")
ax3.set_ylim(bottom=-0.2, top=1.0)
ax3.axhline(y=0.0, color=Grey)
ax3.axvline(x=0.0, color=Grey)
ax3.plot(th, cozth,  color=Black,      linestyle='--', zorder=6, label=r"Unfiltered")

# plot axial auto-correlation
ax4 = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1)
ax4.set_xlim(left=-1000.0, right=1000.0)
ax4.set_xticks([-1000.0, -500.0, 0.0, 500.0, 1000.0])
ax4.set_xticklabels([])
ax4.set_ylim(bottom=-0.3, top=1.0)
ax4.set_yticklabels([])
ax4.axhline(y=0.0, color=Grey)
ax4.axvline(x=0.0, color=Grey)
ax4.plot(z, cozz,  color=Black,      linestyle='--', zorder=6, label=r"Unfiltered")
ax4.legend(loc='lower left', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot azimuthal auto-correlation
ax5 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1)
ax5.set_xlabel(r"$\Delta\theta r^+$")
ax5.set_xlim(left=-200.0, right=200.0)
ax5.set_xticks([-200.0, -100.0, 0.0, 100.0, 200.0])
ax5.set_ylabel(r"$C_{\Pi\Pi}$")
ax5.set_ylim(bottom=-0.2, top=1.0)
ax5.axhline(y=0.0, color=Grey)
ax5.axvline(x=0.0, color=Grey)
ax5.plot(th, cpithF, color=Black,      linestyle='-', zorder=7, label=r"Fourier")
ax5.plot(th, cpithG, color=Vermillion, linestyle='-', zorder=8, label=r"Gauss")
ax5.plot(th, cpithB, color=Blue,       linestyle='-', zorder=9, label=r"Box")

# plot axial auto-correlation
ax6 = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1)
ax6.set_xlabel(r"$\Delta z^+$")
ax6.set_xlim(left=-1000.0, right=1000.0)
ax6.set_xticks([-1000.0, -500.0, 0.0, 500.0, 1000.0])
ax6.set_ylim(bottom=-0.3, top=1.0)
ax6.set_yticklabels([])
ax6.axhline(y=0.0, color=Grey)
ax6.axvline(x=0.0, color=Grey)
ax6.plot(z, cpizF, color=Black,      linestyle='-', zorder=7, label=r"Fourier")
ax6.plot(z, cpizG, color=Vermillion, linestyle='-', zorder=8, label=r"Gauss")
ax6.plot(z, cpizB, color=Blue,       linestyle='-', zorder=9, label=r"Box")
ax6.legend(loc='lower left', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot mode interactive or pdf
if plot == 1:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = str.replace(fnam, '.dat', '.pdf')
 fnam = str.replace(fnam, 'piCorrZStreaksBox2d', 'plotPiCorr1dPiCompare')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
print('Done!')
