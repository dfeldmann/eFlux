#!/usr/bin/env python3
# Purpose:  Read pre-computed 2d correlation maps in a wall-parallel plane for
#           the energy flux (Pi) and the axial velocity component (u'_z)
#           representing streamwise streaks. Correlation maps are read from
#           ascii and plotted for three different filters (Fourier, Gauss, box)
#           to compare the effect of the filter kernel. Additional contour
#           lines can be plotted on top. Output is interactive or as pfd
#           figure file. TODO: find out how to read number following a
#           particular string when reading from file, see manual hack below
# Usage:    python plotPiCorr2dStreaksCompare.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 16th September 2019

import sys
import os.path
import timeit
import math
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
print('Plot 2d cross-correlations between energy flux and streaks based on different kernels.')
plot = int(input("Enter plot mode (0 = none, 1 = interactive, 2 = pdf file): "))

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# read 2d cross-correlation between eFlux and streaks for Fourier filter from ascii file
fnam = 'piCorrThZStreaksFourier2d_pipe0002_01675000to01675000nt0001.dat'
print('Reading 2d cross-correlations from', fnam)
Dt = np.loadtxt(fnam)[:, 0] # 1st column: Azimuthal displacement
Dz = np.loadtxt(fnam)[:, 1] # 2nd column: Axial displacement
f  = np.loadtxt(fnam)[:, 6] # 7th column: Cross-correlation for u'_z and Pi

# manual hack (TODO) 
nth = 385  # azimuthal points
nz  = 2305 # axial points

# read 2d cross-correlation between eFlux and streaks for Gauss filter from ascii file
fnam = 'piCorrThZStreaksGauss2d_pipe0002_01675000to01675000nt0001.dat'
print('Reading 2d cross-correlations from', fnam)
g = np.loadtxt(fnam)[:, 6]

# read 2d cross-correlation between eFlux and streaks for box filter from ascii file
fnam = 'piCorrThZStreaksBox2d_pipe0002_01675000to01675000nt0001.dat'
print('Reading 2d cross-correlations from', fnam)
b = np.loadtxt(fnam)[:, 6]

# re-cast cross-correlation data into 2d array for plotting
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

# convert spatial coordinates from outer to inner units
DeltaTh = DeltaTh * Re_tau
DeltaZ  = DeltaZ  * Re_tau

# find absolute maxima of extracted 2d data
amccF = np.max(np.abs(ccF))           # Fourier max
amccG = np.max(np.abs(ccG))           # Gauss max
amccB = np.max(np.abs(ccB))           # Box max
amcc  = np.max([amccF, amccG, amccB]) # all max
print("Absolute maximum correlation value:", amcc)
amcc  = 0.30000                       # manual max
clm   = 0.10*amcc                     # manual contour level

# plotting
if plot not in [1, 2]: sys.exit() # skip everything below
print('Creating plot (using LaTeX)...')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
fig = plt.figure(num=None, figsize=mm2inch(75.0, 65.0), dpi=300)

# conservative colour palette appropriate for colour-blind (http://mkweb.bcgsc.ca/colorblind/)
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Black         = '#000000'
exec(open("./colourMaps.py").read()) # many thanks to github.com/nesanders/colorblind-colormap 
VermBlue = CBWcm['VeBu']             # from Vermillion (-) via White (0) to Blue (+)

# modify box for filter name annotation
filterBox = dict(boxstyle="square, pad=0.3", fc='w', ec=Black, lw=0.5)

# axes grid for multiple subplots with common colour bar
from mpl_toolkits.axes_grid1 import ImageGrid
ig = ImageGrid(fig, 111, nrows_ncols=(3, 1), axes_pad=0.04, cbar_size=0.10, cbar_mode='single', cbar_location='right', cbar_pad=0.05)
ax1 = ig[0]
ax2 = ig[1]
ax3 = ig[2]

# define sub-set for plotting (Here in plus units)
xmin =  -700.0
xmax =   700.0
ymin =  -250.0
ymax =   250.0

# plot Fourier eFlux
ax1.set_xlim(left=xmin, right=xmax)
ax1.set_ylabel(r"$\Delta\theta r^{+}$")
ax1.set_ylim(bottom=ymin, top=ymax)
im1 = ax1.imshow(ccF, vmin=-amcc, vmax=+amcc, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
cl1n = ax1.contour(DeltaZ, DeltaTh, ccF, levels=[-clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl1p = ax1.contour(DeltaZ, DeltaTh, ccF, levels=[+clm], colors=Blue, linestyles='-', linewidths=0.5)
ax1.set_aspect('equal')
t1 = ax1.text(-650.0, 200.0, r"Fourier", ha="left", va="top", rotation=0, bbox=filterBox)

# plot Gauss eFlux
ax2.set_xlim(left=xmin, right=xmax)
ax2.set_ylabel(r"$\Delta\theta r^{+}$")
ax2.set_ylim(bottom=ymin, top=ymax)
im2 = ax2.imshow(ccG, vmin=-amcc, vmax=+amcc, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
cl2n = ax2.contour(DeltaZ, DeltaTh, ccG, levels=[-clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl2p = ax2.contour(DeltaZ, DeltaTh, ccG, levels=[+clm], colors=Blue, linestyles='-', linewidths=0.5)
ax2.set_aspect('equal')
t2 = ax2.text(-650.0, 200.0, r"Gauss", ha="left", va="top", rotation=0, bbox=filterBox)

# plot box eFlux
ax3.set_xlabel(r"$\Delta z^{+}$")
ax3.set_xlim(left=xmin, right=xmax)
ax3.set_ylabel(r"$\Delta\theta r^{+}$")
ax3.set_ylim(bottom=ymin, top=ymax)
im3 = ax3.imshow(ccB, vmin=-amcc, vmax=+amcc, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
cl3n = ax3.contour(DeltaZ, DeltaTh, ccB, levels=[-clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl3p = ax3.contour(DeltaZ, DeltaTh, ccB, levels=[+clm], colors=Blue, linestyles='-', linewidths=0.5)
ax3.set_aspect('equal')
t3 = ax3.text(-650.0, 200.0, r"Box", ha="left", va="top", rotation=0, bbox=filterBox)

# plot common colour bar
fmt = FormatStrFormatter('%9.2f')
cb1 = ax1.cax.colorbar(im1, format=fmt)
cb1 = ig.cbar_axes[0].colorbar(im1)
cb1.ax.set_ylabel(r"$C_{u^{\prime}_{z}\Pi^{\prime}}$")
cb1.ax.set_yticks([-amcc, 0.0, +amcc])
cb1.ax.set_yticks([-amcc, clm, 0.0, -clm, +amcc]) # tweak colour bar ticks to show manual countour level

# plot mode interactive or pdf
if plot != 2:
 #plt.tight_layout()
 plt.show()
else:
 #fig.tight_layout()
 fnam = str.replace(fnam, '.dat', '.pdf')
 fnam = str.replace(fnam, 'piCorrThZStreaksBox2d', 'plotPiCorr2dStreaksCompare')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
