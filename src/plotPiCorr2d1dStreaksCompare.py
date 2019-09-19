#!/usr/bin/env python3
# Purpose:  Read pre-computed 1d and 2d-correlation maps in a wall-parallel
#           plane for the energy flux (Pi) and the axial velocity component
#           (u'_z) representing streamwise streaks. Correlation maps are read
#           from several ascii files and plotted for three different filters
#           (Fourer, Gauss, box) to compare the effect of the kernel. The 1d
#           correlations appear along the corresponding axes of the 2d maps.
#           Additional contour lines can be plotted on top of the 2d maps.
#           Output is interactive or as pfd figure file. TODO: find out how
#           to read number following a particular string when reading from file,
#           see manual hack below.
# Usage:    python plotPiCorr2d1dStreaksCompare.py
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 19th September 2019

import sys
import os.path
import timeit
import math
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
print('Plot 1d and 2d cross-correlations between energy flux and streaks based on different kernels.')
plot = int(input("Enter plot mode (0 = none, 1 = interactive, 2 = pdf file): "))

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# read 1d axial cross-correlation with streaks for Fourier filtered eFlux from ascii file
fnam = 'piCorrZStreaksFourier2d_pipe0002_00570000to01265000nt0140.dat'
print('Reading 1d cross-correlation from', fnam)
z1d  = np.loadtxt(fnam)[:, 0] # 1st column: axial separation, only once
puzF = np.loadtxt(fnam)[:, 1] # 2nd column: UzPi correlation

# read 1d azimuthal cross-correlation with streaks for Fourier filtered eFlux from ascii file
fnam = 'piCorrThStreaksFourier2d_pipe0002_00570000to01265000nt0140.dat'
print('Reading 1d cross-correlation from', fnam)
th1d  = np.loadtxt(fnam)[:, 0] # 1st column: azimuthal separation, only once
puthF = np.loadtxt(fnam)[:, 1] # 2nd column: UzPi correlaation

# read 1d axial cross-correlation with streaks for Gauss filtered eFlux from ascii file
fnam = 'piCorrZStreaksGauss2d_pipe0002_00570000to01265000nt0140.dat'
print('Reading 1d cross-correlation from', fnam)
puzG = np.loadtxt(fnam)[:, 1] # 2nd column: UzPi correlation

# read 1d azimuthal cross-correlation with streaks for Gauss filtered eFlux from ascii file
fnam = 'piCorrThStreaksGauss2d_pipe0002_00570000to01265000nt0140.dat'
print('Reading 1d cross-correlation from', fnam)
puthG = np.loadtxt(fnam)[:, 1] # 2nd column: UzPi correlaation

# read 1d axial cross-correlation with streaks for box filtered eFlux from ascii file
fnam = 'piCorrZStreaksBox2d_pipe0002_00570000to01265000nt0140.dat'
print('Reading 1d cross-correlation from', fnam)
puzB = np.loadtxt(fnam)[:, 1] # 2nd column: UzPi correlation

# read 1d azimuthal cross-correlation with streaks for box filtered eFlux from ascii file
fnam = 'piCorrThStreaksBox2d_pipe0002_00570000to01265000nt0140.dat'
print('Reading 1d cross-correlation from', fnam)
puthB = np.loadtxt(fnam)[:, 1] # 2nd column: UzPi correlaation

# read 2d cross-correlation between eFlux and streaks for Fourier filter from ascii file
fnam = 'piCorrThZStreaksFourier2d_pipe0002_01675000to01675000nt0001.dat'
fnam = 'piCorrThZStreaksFourier2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading 2d cross-correlations from', fnam)
Dt = np.loadtxt(fnam)[:, 0] # 1st column: Azimuthal displacement
Dz = np.loadtxt(fnam)[:, 1] # 2nd column: Axial displacement
f  = np.loadtxt(fnam)[:, 6] # 7th column: Cross-correlation for u'_z and Pi

# read 2d cross-correlation between eFlux and streaks for Gauss filter from ascii file
fnam = 'piCorrThZStreaksGauss2d_pipe0002_01675000to01675000nt0001.dat'
fnam = 'piCorrThZStreaksGauss2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading 2d cross-correlations from', fnam)
g = np.loadtxt(fnam)[:, 6]

# read 2d cross-correlation between eFlux and streaks for box filter from ascii file
fnam = 'piCorrThZStreaksBox2d_pipe0002_01675000to01675000nt0001.dat'
fnam = 'piCorrThZStreaksBox2d_pipe0002_00570000to01675000nt0222.dat'
print('Reading 2d cross-correlations from', fnam)
b = np.loadtxt(fnam)[:, 6]

# manual hack (TODO: read this from header info of piCorrThZ*.dat) 
nth = 385  # azimuthal points
nz  = 2305 # axial points

# re-cast cross-correlation data into 2d array for plotting
# (TODO: this is straight-forward fortran programming style and can maybe be done much more efficiently in Python...?)
print('Re-cast data into 2d arrays for plotting')
DeltaTh = np.zeros(nth)
DeltaZ  = np.zeros(nz)
ccF = np.zeros((nth, nz))
ccG = np.zeros((nth, nz))
ccB = np.zeros((nth, nz))
for i in range(nth): 
 for j in range(nz):
  DeltaTh[i] = Dt[i*(nz)+j]
  DeltaZ[j]  = Dz[i*(nz)+j]
  ccF[i, j] = f[i*(nz)+j]
  ccG[i, j] = g[i*(nz)+j]
  ccB[i, j] = b[i*(nz)+j]

# find absolute maxima of 2d data sets
amccF = np.max(np.abs(ccF))            # Fourier max
amccG = np.max(np.abs(ccG))            # Gauss max
amccB = np.max(np.abs(ccB))            # Box max
amcc  = np.max([amccF, amccG, amccB])  # all max
print("Absolute maximum correlation value:", amcc)
amcc  = 0.300                          # manual max
clm   = 0.100*amcc                     # set contour level threshold

# convert spatial separation from outer to inner units
th1d = th1d * Re_tau
z1d  =  z1d * Re_tau
DeltaTh = DeltaTh * Re_tau
DeltaZ  = DeltaZ  * Re_tau

# plotting
if plot not in [1, 2]: sys.exit() # skip everything below
print('Creating plot (using LaTeX)...')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
r"\usepackage[utf8]{inputenc}",
r"\usepackage[T1]{fontenc}",
r"\usepackage[detect-all]{siunitx}",
r'\usepackage{amsmath, amstext, amssymb}',
r'\usepackage{xfrac}',
#r'\usepackage{lmodern, palatino, eulervm}',
r'\usepackage{mathptmx}']
#mpl.rcParams.update({'font.family': 'sans-serif'})
mpl.rcParams.update({'font.family': 'serif'})
mpl.rcParams.update({'font.size': 8})
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
#fig = plt.figure(num=None, figsize=mm2inch(134.0, 150.0), dpi=300, constrained_layout=False) 
fig = plt.figure(num=None, dpi=100, constrained_layout=False) 

# conservative colour palette appropriate for colour-blind (http://mkweb.bcgsc.ca/colorblind/)
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Grey          = '#999999'
Black         = '#000000'
exec(open("./colourMaps.py").read()) # many thanks to github.com/nesanders/colorblind-colormap 
VermBlue = CBWcm['VeBu']             # from Vermillion (-) via White (0) to Blue (+)

# modify box for filter name annotation and sub figure label
filterBox = dict(boxstyle="square, pad=0.3", lw=0.5, fc='w', ec=Black)
labelBox  = dict(boxstyle="square, pad=0.3", lw=0.5, fc='w', ec='w')

# axes grid for multiple subplots
import matplotlib.gridspec as gridspec
gs = fig.add_gridspec(nrows=4, ncols=2, hspace=0.0, wspace=0.0, width_ratios=[1,0.25], height_ratios=[1,1,1,1])

# my data range
# define sub-set for plotting (Here in plus units)
xmin = -800.00
xmax =  800.00
ymin = -180.00
ymax =  180.00
cmin =   -0.35
cmax =    0.20

# plot 2d correlation map for Fourier eFlux
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(left=xmin, right=xmax)
ax1.set_ylabel(r"$\Delta\theta r^{+}$")
ax1.set_ylim(bottom=ymin, top=ymax)
ax1.set_yticks([-100, 0.0, 100])
ax1.tick_params(labelbottom=False, labelleft=True)
ax1.minorticks_on()
im1 = ax1.imshow(ccF, vmin=-amcc, vmax=+amcc, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
cl1n = ax1.contour(DeltaZ, DeltaTh, ccF, levels=[-clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl1p = ax1.contour(DeltaZ, DeltaTh, ccF, levels=[+clm], colors=Blue, linestyles='-', linewidths=0.5)
ax1.text(-750.0, -130.0, r"Fourier", ha="left", va="bottom", rotation=0, bbox=filterBox)
ax1.text(-765.0,  145.0, r"a)", ha="left", va="top", rotation=0, bbox=labelBox)

# plot 2d correlation map for Gauss eFlux
ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
ax2.set_xlim(left=xmin, right=xmax)
ax2.set_ylabel(r"$\Delta\theta r^{+}$")
ax2.set_ylim(bottom=ymin, top=ymax)
ax2.set_yticks([-100, 0.0, 100])
ax2.tick_params(labelbottom=False, labelleft=True)
ax2.minorticks_on()
im2 = ax2.imshow(ccG, vmin=-amcc, vmax=+amcc, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
cl2n = ax2.contour(DeltaZ, DeltaTh, ccG, levels=[-clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl2p = ax2.contour(DeltaZ, DeltaTh, ccG, levels=[+clm], colors=Blue, linestyles='-', linewidths=0.5)
ax2.text(-750.0, -130.0, r"Gauss", ha="left", va="bottom", rotation=0, bbox=filterBox)
ax2.text(-765.0,  145.0, r"b)", ha="left", va="top", rotation=0, bbox=labelBox)

# plot 2d correlation map for box eFlux
ax3 = fig.add_subplot(gs[2,0], sharex=ax1)
ax3.set_xlim(left=xmin, right=xmax)
ax3.set_ylabel(r"$\Delta\theta r^{+}$")
ax3.set_ylim(bottom=ymin, top=ymax)
ax3.set_yticks([-100, 0.0, 100])
ax3.tick_params(labelbottom=False, labelleft=True)
ax3.minorticks_on()
im3 = ax3.imshow(ccB, vmin=-amcc, vmax=+amcc, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
cl3n = ax3.contour(DeltaZ, DeltaTh, ccB, levels=[-clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl3p = ax3.contour(DeltaZ, DeltaTh, ccB, levels=[+clm], colors=Blue, linestyles='-', linewidths=0.5)
ax3.text(-750.0, -130.0, r"Box", ha="left", va="bottom", rotation=0, bbox=filterBox)
ax3.text(-765.0,  145.0, r"c)", ha="left", va="top", rotation=0, bbox=labelBox)

# plot 1d azimuthal cross-correlation on the right
ax4 = fig.add_subplot(gs[2,1], sharey = ax3)
ax4.set_xlabel(r"$C_{u^{\prime}_{z}\Pi}$")
ax4.set_xlim(left=-0.22, right=0.15)
ax4.set_xticks([-0.15, 0.0, 0.15])
ax4.set_ylim(bottom=ymin, top=ymax)
ax4.tick_params(labelbottom=True, labelleft=False)
ax4.axhline(y=0.0, color=Grey)
ax4.axvline(x=0.0, color=Grey)
ax4.plot(puthF, th1d, color=Black,      linestyle='-', zorder=7, label=r"Fourier")
ax4.plot(puthG, th1d, color=Vermillion, linestyle='-', zorder=9, label=r"Gauss")
ax4.plot(puthB, th1d, color=Blue,       linestyle='-', zorder=8, label=r"Box")
ax4.text(-0.19, 145.0, r"d)", ha="left", va="top", rotation=0, bbox=labelBox)

# plot 1d axial cross-correlation at the bottom
ax5 = fig.add_subplot(gs[3,0], sharex = ax1)
ax5.set_xlabel(r"$\Delta z^+$")
ax5.set_xlim(left=xmin, right=xmax)
ax5.set_xticks([-800, -400, 0.0, 400, 800])
ax5.set_ylabel(r"$C_{u^{\prime}_{z}\Pi}$")
ax5.set_ylim(bottom=-0.33, top=0.1)
ax5.set_yticks([-0.3, -0.2, -0.1, 0.0, 0.1])
ax5.axhline(y=0.0, color=Grey)
ax5.axvline(x=0.0, color=Grey)
ax5.plot(z1d, puzF, color=Black,      linestyle='-', zorder=7, label=r"Fourier")
ax5.plot(z1d, puzG, color=Vermillion, linestyle='-', zorder=9, label=r"Gauss")
ax5.plot(z1d, puzB, color=Blue,       linestyle='-', zorder=8, label=r"Box")
ax5.legend(bbox_to_anchor=(1.247, 0.6125), frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)
ax5.text(-765.0, 0.07, r"e)", ha="left", va="top", rotation=0, bbox=labelBox)

# add this for consistent representation of images in ax1 to ax3
# ax1.set_aspect('equal')
# ax2.set_aspect('equal')
# ax3.set_aspect('equal')
# hack: aspect=equal is default for imshow, but destroys my gridspec
ax1.set_aspect('auto')
ax2.set_aspect('auto')
ax3.set_aspect('auto')

# plot common colour bar
axc = plt.axes([0.76, 0.515, 0.035, 0.355])
fmt = FormatStrFormatter('%4.1f')
cb1 = plt.colorbar(im1, cax=axc, format=fmt, orientation="vertical")
cb1.ax.set_ylabel(r"$C_{u^{\prime}_{z}\Pi}$")
cb1.set_ticks([-amcc, 0.0, +amcc])
axc.axhline(y=-clm, color=Vermillion) # mark negative contour level in colourbar
axc.axhline(y=+clm, color=Blue)       # mark positive contour level in colourbar

# plot mode interactive or pdf
if plot != 2:
 #plt.tight_layout()
 plt.show()
else:
 #fig.tight_layout()
 fnam = str.replace(fnam, '.dat', '.pdf')
 fnam = str.replace(fnam, 'piCorrThZStreaksBox2d', 'plotPiCorr2d1dStreaksCompare')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
