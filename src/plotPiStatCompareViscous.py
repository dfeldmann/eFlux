#!/usr/bin/env python3
# Purpose:  Read pre-computed one-point energy flux statistics from ascii files
#           based on different filter kernels (Fourier, Gauss, box). Also read
#           reference data from literature (Hartel et la. 1994 PoF). Plot radial
#           profiles of mean, RMS, skewness and flatness factors in viscous
#           (inner) units. Output is interactive or as pfd figure file.
# Usage:    python plotPiStatCompareViscous.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 25th August 2019

import timeit
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
print('Plot energy flux one-point statistics in viscous units')
plot = int(input("Enter plot mode (0 = none, 1 = interactive, 2 = pdf file): "))

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# read one-point statistics for eflux Fourier filtered from ascii file
fnam = 'piStatFourier2d_pipe0002_01675000to01675000nt0001.dat'
fnam = 'piStatFourier2d_pipe0002_00570000to01675000nt0222.dat' 
print('Reading eFlux statistics from', fnam)
eFMeanFourier = np.loadtxt(fnam)[:, 1]
eFRmsFourier  = np.loadtxt(fnam)[:, 2]
eFSkewFourier = np.loadtxt(fnam)[:, 3]
eFFlatFourier = np.loadtxt(fnam)[:, 4]

# read one-point statistics for eflux Gauss filtered from ascii file
fnam = 'piStatGauss2d_pipe0002_01675000to01675000nt0001.dat'
fnam = 'piStatGauss2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading eFlux statistics from', fnam)
eFMeanGauss = np.loadtxt(fnam)[:, 1]
eFRmsGauss  = np.loadtxt(fnam)[:, 2]
eFSkewGauss = np.loadtxt(fnam)[:, 3]
eFFlatGauss = np.loadtxt(fnam)[:, 4]

# read one-point statistics for eflux box filtered from ascii file
fnam = 'piStatBox2d_pipe0002_01675000to01675000nt0001.dat'
fnam = 'piStatBox2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading eFlux statistics from', fnam)
r = np.loadtxt(fnam)[:, 0] # radial grid only once
eFMeanBox = np.loadtxt(fnam)[:, 1]
eFRmsBox  = np.loadtxt(fnam)[:, 2]
eFSkewBox = np.loadtxt(fnam)[:, 3]
eFFlatBox = np.loadtxt(fnam)[:, 4]

# grid size
nr  = len(r)
print('With', nr, 'radial (r) grid points.')
print('It is your responsibility to make sure, that all data sets are defined on the exact same grid.')

# read reference data from ascii file 
rnam = 'haertel1994fig8b.dat' # Hartel et al. PoF 1994
print('Reading reference data from', rnam)
ypRef = np.loadtxt(rnam)[:, 0]
eFMeanRef = np.loadtxt(rnam)[:, 1]

# plotting
if plot not in [1, 2]: sys.exit() # skip everything below
print('Creating plot (using LaTeX)...')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#from mpl_toolkits.axes_grid1.colorbar import colorbar
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
r"\usepackage[utf8]{inputenc}",
r"\usepackage[T1]{fontenc}",
r'\usepackage{lmodern, palatino, eulervm}',
#r'\usepackage{mathptmx}',
r"\usepackage[detect-all]{siunitx}",
r'\usepackage{amsmath, amstext, amssymb}',
r'\usepackage{xfrac}']
#mpl.rcParams.update({'font.family': 'sans-serif'})
mpl.rcParams.update({'font.family' : 'serif'})
mpl.rcParams.update({'font.size' : 8})
mpl.rcParams.update({'lines.linewidth'   : 0.75})
mpl.rcParams.update({'axes.linewidth'    : 0.75})
mpl.rcParams.update({'xtick.major.size'  : 2.00})
mpl.rcParams.update({'xtick.major.width' : 0.75})
mpl.rcParams.update({'xtick.minor.size'  : 1.00})
mpl.rcParams.update({'xtick.minor.width' : 0.75})
mpl.rcParams.update({'ytick.major.size'  : 2.00})
mpl.rcParams.update({'ytick.major.width' : 0.75})
mpl.rcParams.update({'ytick.minor.size'  : 1.00})
mpl.rcParams.update({'ytick.minor.width' : 0.75})

# create figure suitable for A4 format
def mm2inch(*tupl):
 inch = 25.4
 if isinstance(tupl[0], tuple):
  return tuple(i/inch for i in tupl[0])
 else:
   return tuple(i/inch for i in tupl)
fig = plt.figure(num=None, figsize=mm2inch(180.0, 120.0), dpi=150)

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

# wall distance in viscous (plus) units
yp = (1.0-r) * Re_tau

# conversion factor to viscous units
#fu  =  Re_b/Re_tau
#fp  = (Re_b/Re_tau)**2.0
fpi = 1.0 # (Re_b/Re_tau)**1.0  # please double check all this!!!
#fpi = (Re_b/Re_tau)**3.0
#fpi = (1.0/Re_b)**3.0

# plot first order statistics
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax1.set_xscale('log')
ax1.set_xlim(left=6.0e-1, right=Re_tau)
ax1.set_xticks([])
ax1.set_ylabel(r"$\langle\Pi\rangle_{\theta, z, t}$ in $U^3_{\text{HP}}R^2$")
#ax1.set_ylim(bottom=-0.010, top=0.035)
ax1.axhline(y=0.0, color=Grey)
ax1.plot(yp[:-2], eFMeanFourier[:-2]*fpi, color=Black,      linestyle='-',  zorder=7, label=r"Fourier")
ax1.plot(yp[:-2],   eFMeanGauss[:-2]*fpi, color=Vermillion, linestyle='-',  zorder=9, label=r"Gauss")
ax1.plot(yp[:-2],     eFMeanBox[:-2]*fpi, color=Blue,       linestyle='-',  zorder=8, label=r"Box")
#ax1.plot(ypRef[:], eFMeanRef[:],          color=BluishGreen, linestyle='-',  zorder=6, label=r"H\"artel et al. 1994")
ax1.legend(loc='best', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot second order statistics
ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax2.set_xscale('log')
ax2.set_xlim(left=6.0e-1, right=Re_tau)
ax2.set_xticks([])
ax2.set_ylabel(r"$\sqrt{ \langle\Pi^{\prime 2} \rangle_{\theta, z, t} }$ in $U^3_{\text{HP}}R^2$")
#ax2.set_ylim(bottom=0.0, top=0.25)
ax2.plot(yp[:-2], eFRmsFourier[:-2]*fpi, color=Black,      linestyle='-',  zorder=7, label=r"Fourier")
ax2.plot(yp[:-2],   eFRmsGauss[:-2]*fpi, color=Vermillion, linestyle='-',  zorder=9, label=r"Gauss")
ax2.plot(yp[:-2],     eFRmsBox[:-2]*fpi, color=Blue,       linestyle='-',  zorder=8, label=r"Box")

# plot third order statistics
ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax3.set_xlabel(r"$y^{+}$ in $\sfrac{\nu}{u_{\tau}}$")
ax3.set_xscale('log')
ax3.set_xlim(left=6.0e-1, right=Re_tau)
ax3.set_ylabel(r"$\langle\Pi^{\prime 3} \rangle_{\theta, z, t}$ in $\langle\Pi^{\prime 2} \rangle^{\sfrac{3}{2}}_{\theta, z, t} $  ")
ax3.axhline(y=0.0, color=Grey)
ax3.plot(yp[:-2], eFSkewFourier[:-2]*fpi, color=Black,      linestyle='-',  zorder=7, label=r"Fourier")
ax3.plot(yp[:-2],   eFSkewGauss[:-2]*fpi, color=Vermillion, linestyle='-',  zorder=9, label=r"Gauss")
ax3.plot(yp[:-2],     eFSkewBox[:-2]*fpi, color=Blue,       linestyle='-',  zorder=8, label=r"Box")

# plot fourth order statistics
ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
ax4.set_xlabel(r"$y^{+}$ in $\sfrac{\nu}{u_{\tau}}$")
ax4.set_xscale('log')
ax4.set_xlim(left=6.0e-1, right=Re_tau)
ax4.set_ylabel(r"$\langle\Pi^{\prime 4} \rangle_{\theta, z, t}$ in $\langle\Pi^{\prime 2} \rangle^{\sfrac{4}{2}}_{\theta, z, t} $  ")
ax4.set_ylim(bottom=0.0, top=500.0)
ax4.plot(yp[:-2], eFFlatFourier[:-2]*fpi, color=Black,      linestyle='-',  zorder=7, label=r"Fourier")
ax4.plot(yp[:-2],   eFFlatGauss[:-2]*fpi, color=Vermillion, linestyle='-',  zorder=9, label=r"Gauss")
ax4.plot(yp[:-2],     eFFlatBox[:-2]*fpi, color=Blue,       linestyle='-',  zorder=8, label=r"Box")

# plot mode interactive or pdf
if plot == 1:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = str.replace(fnam, '.dat', '.pdf')
 fnam = str.replace(fnam, 'piStatBox2d', 'plotPiStatCompareViscous')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
