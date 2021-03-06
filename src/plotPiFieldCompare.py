#!/usr/bin/env python3
# Purpose:  Read pre-computed instantaneous 3d energy flux fields from HDF5 file
#           based on different filter kernels (Fourier, Gauss, box). Extract 2d
#           data sets in a wall-parallel plane and plot them to compare the
#           effect of the filter kernel. Additional contour lines and a box to
#           to highlight a zoom region can be plotted on top. Output is
#           interactive or as pfd figure file.
# Usage:    python plotPiFieldCompare.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 30th September 2019

import timeit
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
print('Plot instantaneous energy flux in a wall-parallel plane for different filter kernels')
plot = int(input("Enter plot mode (0 = none, 1 = interactive, 2 = pdf file): "))

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# read Fourier energy flux field from file
fnam = '../40x75/piFieldFourier2d_pipe0002_01675000.h5'
f = h5py.File(fnam, 'r')
print("Reading eflux from file", fnam)
r   = np.array(f['grid/r']) # read grid only once
z   = np.array(f['grid/z'])
th  = np.array(f['grid/th'])
piF = np.array(f['fields/pi/pi']).transpose(0,2,1)
f.close()

# read Gauss energy flux field from file
fnam = str.replace(fnam, 'Fourier', 'Gauss')
f = h5py.File(fnam, 'r')
print("Reading eflux from file", fnam)
piG = np.array(f['fields/pi/pi']).transpose(0,2,1)
f.close()

# read box energy flux field from file
fnam = str.replace(fnam, 'Gauss', 'Box')
f = h5py.File(fnam, 'r')
print("Reading eflux from file", fnam)
piB = np.array(f['fields/pi/pi']).transpose(0,2,1)
f.close()

# grid size
nr  = len(r)
nth = len(th)
nz  = len(z)
print(nr, 'radial (r) points')
print(nth, 'azimuthal (th) points')
print(nz, 'axial (z) points')

# extract streaks and eflux from a plane
k = 65
print ("Extracting 2d data in wall-parallel plane at y+ =", (1-r[k])*Re_tau)
piF = piF[k, :, :]  
piG = piG[k, :, :]  
piB = piB[k, :, :]  

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
#fig = plt.figure(num=None, figsize=mm2inch(135.0, 65.0), dpi=300)
fig = plt.figure(num=None, dpi=300)

# conservative colour palette appropriate for colour-blind (http://mkweb.bcgsc.ca/colorblind/)
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Black         = '#000000'
exec(open("./colourMaps.py").read()) # 
VermBlue = CBWcm['VeBu']             # Vermillion (-) White (0) Blue (+)

# convert spatial coordiantes from outer to inner units
#r = r * Re_tau
#z = z * Re_tau

# convert eFlux from outer to inner (viscous) units
fpi = Re_b**3.0 / Re_tau**4.0
print('Factor for eFlux in viscous units:', fpi)
piF = piF * fpi
piG = piG * fpi
piB = piB * fpi

# find absolute maxima of extracted 2d data
ampiF = np.max(np.abs(piF))           # Fourier max
ampiG = np.max(np.abs(piG))           # Gauss max
ampiB = np.max(np.abs(piB))           # Box max
ampi  = np.max([ampiF, ampiG, ampiB]) # all max
ampi  = 1.5000                        # manual max
#clm   =-0.0020                       # manual contour level

# modify box for filter name annotation
filterBox = dict(boxstyle="square, pad=0.3", fc='w', ec=Black, lw=0.5)

# axes grid for multiple subplots with common colour bar
from mpl_toolkits.axes_grid1 import ImageGrid
ig = ImageGrid(fig, 111, nrows_ncols=(3, 1), axes_pad=0.04, cbar_size=0.10, cbar_mode='single', cbar_location='right', cbar_pad=0.05)
ax1 = ig[0]
ax2 = ig[1]
ax3 = ig[2]

# plot Fourier eFlux
ax1.set_ylabel(r"$\theta r$ in $R$")
im1 = ax1.imshow(piF, vmin=-ampi, vmax=+ampi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
#cl1 = ax1.contour(z, th*r[k], piF, levels=[clm], colors=Black, linestyles='-', linewidths=0.1)
ax1.set_aspect('equal')
ax1.text(0.985, 0.90, r"Fourier", ha="right", va="top", transform=ax1.transAxes, rotation=0, bbox=filterBox)
ax1.text(0.008, 0.93, r"a)", ha="left", va="top", transform=ax1.transAxes)

# plot Gauss eFlux
ax2.set_ylabel(r"$\theta r$ in $R$")
im2 = ax2.imshow(piG, vmin=-ampi, vmax=+ampi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
#cl2 = ax2.contour(z, th*r[k], piG, levels=[clm], colors=Black, linestyles='-', linewidths=0.1)
ax2.set_aspect('equal')
ax2.text(0.985, 0.90, r"Gauss", ha="right", va="top", transform=ax2.transAxes, rotation=0, bbox=filterBox)
ax2.text(0.008, 0.93, r"b)", ha="left", va="top", transform=ax2.transAxes)

# plot box eFlux
ax3.set_xlabel(r"$z$ in $R$")
ax3.set_ylabel(r"$\theta r$ in $R$")
im3 = ax3.imshow(piB, vmin=-ampi, vmax=+ampi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
#cl3 = ax3.contour(z, th*r[k], piB, levels=[clm], colors=Black, linestyles='-', linewidths=0.1)
ax3.set_aspect('equal')
ax3.text(0.985, 0.90, r"Box", ha="right", va="top", transform=ax3.transAxes, rotation=0, bbox=filterBox)
ax3.text(0.008, 0.93, r"c)", ha="left", va="top", transform=ax3.transAxes)

# plot common colour bar
fmt = FormatStrFormatter('%g') # %6.3f
cb1 = ax1.cax.colorbar(im1, format=fmt)
cb1 = ig.cbar_axes[0].colorbar(im1)
#cb1.ax.set_ylabel(r"$\Pi$ in $U^{3}_{c}R^{2}$") # outer units
cb1.ax.set_ylabel(r"$\Pi$ in $\sfrac{u_{\tau}^{4}}{\nu}$", labelpad=-5.0)  # inner units
cb1.ax.set_yticks([-ampi, 0.000, +ampi])
#cb1.set_ticks([-ampi, clm, 0.000, -clm, +ampi]) # tweak colour bar ticks to show manual countour level

# optional rectangle to highlight inset or zoom area
from matplotlib.patches import Rectangle
ax1.add_patch(Rectangle((0.0/Re_tau, 0.0/Re_tau), 1800.0/Re_tau, 600.0/Re_tau, fill=False, color=Vermillion, alpha=1, linewidth=1.0, clip_on=False, zorder=100))
ax2.add_patch(Rectangle((0.0/Re_tau, 0.0/Re_tau), 1800.0/Re_tau, 600.0/Re_tau, fill=False, color=Vermillion, alpha=1, linewidth=1.0, clip_on=False, zorder=100))
ax3.add_patch(Rectangle((0.0/Re_tau, 0.0/Re_tau), 1800.0/Re_tau, 600.0/Re_tau, fill=False, color=Vermillion, alpha=1, linewidth=1.0, clip_on=False, zorder=100))

# plot mode interactive or pdf
if plot != 2:
 #plt.tight_layout()
 plt.show()
else:
 #fig.tight_layout()
 fnam = str.replace(fnam, '.h5', '.pdf')
 fnam = str.replace(fnam, 'piFieldBox2d', 'plotPiFieldCompare')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
