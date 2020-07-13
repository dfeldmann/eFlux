#!/usr/bin/env python3
# Purpose:  Read pre-computed filtered velocity fields from HDF5 file based on
#           box, Fourier and Gauss filter kernel. Read original velocity data
#           from the corresponding unfiltered HDF5 file. Read statistically
#           steady mean profile from ascii file to compute fluctuating velocity
#           fields to visualise streaks (u'_z). Extract 2d data sets in a
#           wall-parallel plane and compute the residual fluctuations to
#           visualise the scales removed by each filter. Output is interactive
#           or as pfd figure file.
# Usage:    python plotFilteredFieldCompare.py
# Authors:  Daniel Feldmann
# Date:     11th February 2020
# Modified: 13th February 2020

import sys
import timeit
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
print('Plot filtered/unfiltered velocity field in a wall-parallel plane for different kernels')
plot = int(input("Enter plot mode (0 = none, 1 = interactive, 2 = pdf file): "))

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# read velocity data from HDF5 file
fpath = '../../outFiles/'
fnam = fpath+'field_pipe0002_01675000.h5'
fnam = fpath+'field_pipe0002_04570000.h5'
f = h5py.File(fnam, 'r')
print("Reading axial velocity field from file", fnam)
r  = np.array(f['grid/r'])    # radial co-ordinate
th = np.array(f['grid/th'])   # azimuthal co-ordinate
z  = np.array(f['grid/z'])    # axial co-ordainte
uz = np.array(f['fields/velocity/u_z']).transpose(0,2,1)
f.close()

# read mean velocity profiles from ascii file
fnam = '../../onePointStatistics/statistics00570000to01675000nt0222.dat'
fnam = '../../onePointStatistics/statistics00570000to04070000nt0351.dat'
print('Reading mean velocity profiles from', fnam)
uzM = np.loadtxt(fnam)[:, 3]

# subtract mean velocity profiles (1d) from flow field (3d)
uz  = uz  - np.tile(uzM, (len(z), len(th), 1)).T

# read box filtered field from file
fnam = '../40x75/filteredFieldBox2d_pipe0002_01675000.h5'
fnam = '../40x75/filteredFieldBox2d_pipe0002_04570000.h5'
f = h5py.File(fnam, 'r')
print("Reading box filtered field from file", fnam)
uzB = np.array(f['fields/velocity/u_zF']).transpose(0,2,1)
f.close()

# read Fourier filtered field from file
fnam = '../40x75/filteredFieldFourier2d_pipe0002_01675000.h5'
fnam = '../40x75/filteredFieldFourier2d_pipe0002_04570000.h5'
f = h5py.File(fnam, 'r')
print("Reading Fourier filtered field from file", fnam)
uzF = np.array(f['fields/velocity/u_zF']).transpose(0,2,1)
f.close()

# read Gauss filtered field from file
fnam = '../40x75/filteredFieldGauss2d_pipe0002_01675000.h5'
fnam = '../40x75/filteredFieldGauss2d_pipe0002_04570000.h5'
f = h5py.File(fnam, 'r')
print("Reading Gauss filtered field from file", fnam)
uzG = np.array(f['fields/velocity/u_zF']).transpose(0,2,1)
f.close()

# grid size
nr  = len(r)
nth = len(th)
nz  = len(z)
print('With', nr, 'radial (r) points')
print('With', nth, 'azimuthal (th) points')
print('With', nz, 'axial (z) points')
print('It is your responsibility to make sure that both fields are defined on the exact same grid.')

# convert velocity from outer to inner (viscous) units
fu = Re_b**1.0 / Re_tau**1.0
print('Conversion factor for velocity in viscous units:', fu)
uz  = uz  * fu
uzB = uzB * fu
uzF = uzF * fu
uzG = uzG * fu

# report global maxima
print("Global max/min u'_z (unfiltered): ", np.max(uz),  np.min(uz))
print("Global max/min u'_z (Fourier):    ", np.max(uzF), np.min(uzF))
print("Global max/min u'_z (Gauss):      ", np.max(uzG), np.min(uzG))
print("Global max/min u'_z (Box):        ", np.max(uzB), np.min(uzB))

# extract 2d streaks and eflux data
k = 65
print ("Extracting 2d data in wall-parallel plane at y+ =", (1-r[k])*Re_tau)
uz2d  =  uz[k, :, :]
uzB2d = uzB[k, :, :]
uzF2d = uzF[k, :, :]
uzG2d = uzG[k, :, :]

# report plane global maxima
print("Plane max/min u'_z (unfiltered): ", np.max(uz2d),  np.min(uz2d))
print("Plane max/min u'_z (Fourier):    ", np.max(uzF2d), np.min(uzF2d))
print("Plane max/min u'_z (Gauss):      ", np.max(uzG2d), np.min(uzG2d))
print("Plane max/min u'_z (Box):        ", np.max(uzB2d), np.min(uzB2d))

# compute residual field
uzBres = uz2d - uzB2d
uzFres = uz2d - uzF2d
uzGres = uz2d - uzG2d

# report residual maxima
print("Residual max/min u'_z (Fourier): ", np.max(uzFres), np.min(uzFres))
print("Residual max/min u'_z (Gauss):   ", np.max(uzGres), np.min(uzGres))
print("Residual max/min u'_z (Box):     ", np.max(uzBres), np.min(uzBres))

# plotting
if plot not in [1, 2]: sys.exit() # skip everything below
print('Creating plot (using LaTeX)...')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
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
mpl.rcParams.update({'font.size' : 6})
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
fig = plt.figure(num=None, figsize=mm2inch(135.0, 70.0), dpi=300)
#fig = plt.figure(num=None, dpi=300)

# conservative colour palette appropriate for colour-blind (http://mkweb.bcgsc.ca/colorblind/)
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Black         = '#000000'
Grey          = '#999999'
exec(open("./colourMaps.py").read()) # 
VermBlue = CBWcm['VeBu']             # Vermillion (-) White (0) Blue (+)

# convert spatial coordiantes from outer to inner units
#r = r * Re_tau
#z = z * Re_tau

# manually set absolute maxima of extracted 2d data for plotting/scaling
amuz  = 7.000 # axial velocity fluctuations, unfiltered
amuzf = 8.000 # axial velocity fluctuations, filtered
amres = 5.000 # residual velocity fields, removed scales

# modify box for filter name annotation
filterBox = dict(boxstyle="square, pad=0.3", fc='w', ec=Black, lw=0.5)
labelBox  = dict(boxstyle="square, pad=0.2", fc='w', ec='w')

# axes grid for multiple subplots with common colour bar
from mpl_toolkits.axes_grid1 import ImageGrid
igTop = ImageGrid(fig, 311, nrows_ncols=(1, 1), axes_pad=0.05, share_all=True, cbar_size=0.08, cbar_mode='single', cbar_location='right', cbar_pad=0.05, label_mode="L", direction="row")
igMid = ImageGrid(fig, 312, nrows_ncols=(1, 3), axes_pad=0.05, share_all=True, cbar_size=0.08, cbar_mode='single', cbar_location='right', cbar_pad=0.05, label_mode="L", direction="row")
igBot = ImageGrid(fig, 313, nrows_ncols=(1, 3), axes_pad=0.05, share_all=True, cbar_size=0.08, cbar_mode='single', cbar_location='right', cbar_pad=0.05, label_mode="L", direction="row")
ax1 = igTop[0]
ax2 = igMid[0]
ax3 = igMid[1]
ax4 = igMid[2]
ax5 = igBot[0]
ax6 = igBot[1]
ax7 = igBot[2]

# zoom box dimensions
zbdx0 = 14.15 
zbdx1 = zbdx0 + 42.0/3.05 
zbdy0 = np.min(th*r[k]) # 0.0
zbdy1 = np.max(th*r[k]) # zbdy0 + 6.28318530717959

# plot unfiltered field
ax1.set_xlim(left=np.min(z), right=np.max(z))
ax1.set_ylabel(r"$\theta r$ in $R$")
im1 = ax1.imshow(uz2d, vmin=-amuz, vmax=+amuz, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
ax1.set_aspect('equal')
ax1.text(0.988, 0.92, r"Unfiltered", ha="right", va="top", transform=ax1.transAxes, bbox=filterBox)
ax1.text(0.012, 0.92, r"a)", ha="left", va="top", transform=ax1.transAxes, bbox=labelBox)

# optional rectangle to highlight inset or zoom area
from matplotlib.patches import Rectangle
ax1.add_patch(Rectangle((zbdx0, zbdy0), (zbdx1-zbdx0), (zbdy1-zbdy0), fill=False, color=Vermillion, alpha=1, linewidth=1.0, clip_on=False, zorder=100))

# plot Fourier filtered field
ax2.set_xlim(left=zbdx0, right=zbdx1)
ax2.set_ylabel(r"$\theta r$ in $R$")
ax2.set_ylim(bottom=zbdy0, top=zbdy1)
im2 = ax2.imshow(uzF2d, vmin=-amuzf, vmax=+amuzf, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
ax2.set_aspect('equal')
ax2.text(0.957, 0.92, r"Fourier", ha="right", va="top", transform=ax2.transAxes, bbox=filterBox)
ax2.text(0.042, 0.92, r"b)", ha="left", va="top", transform=ax2.transAxes, bbox=labelBox)

# plot Gauss filtered field
ax3.set_xlim(left=zbdx0, right=zbdx1)
ax3.set_ylim(bottom=zbdy0, top=zbdy1)
im3 = ax3.imshow(uzG2d, vmin=-amuzf, vmax=+amuzf, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
ax3.set_aspect('equal')
ax3.text(0.957, 0.92, r"Gauss", ha="right", va="top", transform=ax3.transAxes, bbox=filterBox)
ax3.text(0.042, 0.92, r"c)", ha="left", va="top", transform=ax3.transAxes, bbox=labelBox)

# plot box filtered field
ax4.set_xlim(left=zbdx0, right=zbdx1)
ax4.set_ylim(bottom=zbdy0, top=zbdy1)
im4 = ax4.imshow(uzB2d, vmin=-amuzf, vmax=+amuzf, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
ax4.set_aspect('equal')
ax4.text(0.957, 0.92, r"Box", ha="right", va="top", transform=ax4.transAxes, bbox=filterBox)
ax4.text(0.042, 0.92, r"d)", ha="left", va="top", transform=ax4.transAxes, bbox=labelBox)

# plot Fourier residual field
ax5.set_xlabel(r"$z$ in $R$")
ax5.set_xlim(left=zbdx0, right=zbdx1)
ax5.set_ylabel(r"$\theta r$ in $R$")
ax5.set_ylim(bottom=zbdy0, top=zbdy1)
im5 = ax5.imshow(uzFres, vmin=-amres, vmax=+amres, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
ax5.set_aspect('equal')
ax5.text(0.957, 0.92, r"Fourier", ha="right", va="top", transform=ax5.transAxes, bbox=filterBox)
ax5.text(0.042, 0.92, r"e)", ha="left", va="top", transform=ax5.transAxes, bbox=labelBox)

# plot Gauss residual field
ax6.set_xlabel(r"$z$ in $R$")
ax6.set_xlim(left=zbdx0, right=zbdx1)
ax6.set_ylim(bottom=zbdy0, top=zbdy1)
im6 = ax6.imshow(uzGres, vmin=-amres, vmax=+amres, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
ax6.set_aspect('equal')
ax6.text(0.957, 0.92, r"Gauss", ha="right", va="top", transform=ax6.transAxes, bbox=filterBox)
ax6.text(0.042, 0.92, r"f)", ha="left", va="top", transform=ax6.transAxes, bbox=labelBox)

# plot box residual field
ax7.set_xlabel(r"$z$ in $R$")
ax7.set_xlim(left=zbdx0, right=zbdx1)
ax7.set_ylim(bottom=zbdy0, top=zbdy1)
im7 = ax7.imshow(uzBres, vmin=-amres, vmax=+amres, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
ax7.set_aspect('equal')
ax7.text(0.957, 0.92, r"Box", ha="right", va="top", transform=ax7.transAxes, bbox=filterBox)
ax7.text(0.042, 0.92, r"g)", ha="left", va="top", transform=ax7.transAxes, bbox=labelBox)

# annotate filter
lambdaX = 24.0       # place x
lambdaY =  1.0       # place y
lambdaT = 40.0/180.4 # theta scale
lambdaZ = 75.0/180.4 # z scale
ax1.annotate(s='', xy=(lambdaX, lambdaY-lambdaT/2.0), xytext=(lambdaX, lambdaY+lambdaT/2.0), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax1.annotate(s='', xy=(lambdaX-lambdaZ/2.0, lambdaY), xytext=(lambdaX+lambdaZ/2.0, lambdaY), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax2.annotate(s='', xy=(lambdaX, lambdaY-lambdaT/2.0), xytext=(lambdaX, lambdaY+lambdaT/2.0), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax2.annotate(s='', xy=(lambdaX-lambdaZ/2.0, lambdaY), xytext=(lambdaX+lambdaZ/2.0, lambdaY), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax3.annotate(s='', xy=(lambdaX, lambdaY-lambdaT/2.0), xytext=(lambdaX, lambdaY+lambdaT/2.0), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax3.annotate(s='', xy=(lambdaX-lambdaZ/2.0, lambdaY), xytext=(lambdaX+lambdaZ/2.0, lambdaY), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax4.annotate(s='', xy=(lambdaX, lambdaY-lambdaT/2.0), xytext=(lambdaX, lambdaY+lambdaT/2.0), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax4.annotate(s='', xy=(lambdaX-lambdaZ/2.0, lambdaY), xytext=(lambdaX+lambdaZ/2.0, lambdaY), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax5.annotate(s='', xy=(lambdaX, lambdaY-lambdaT/2.0), xytext=(lambdaX, lambdaY+lambdaT/2.0), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax5.annotate(s='', xy=(lambdaX-lambdaZ/2.0, lambdaY), xytext=(lambdaX+lambdaZ/2.0, lambdaY), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax6.annotate(s='', xy=(lambdaX, lambdaY-lambdaT/2.0), xytext=(lambdaX, lambdaY+lambdaT/2.0), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax6.annotate(s='', xy=(lambdaX-lambdaZ/2.0, lambdaY), xytext=(lambdaX+lambdaZ/2.0, lambdaY), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax7.annotate(s='', xy=(lambdaX, lambdaY-lambdaT/2.0), xytext=(lambdaX, lambdaY+lambdaT/2.0), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))
ax7.annotate(s='', xy=(lambdaX-lambdaZ/2.0, lambdaY), xytext=(lambdaX+lambdaZ/2.0, lambdaY), arrowprops=dict(arrowstyle='-', linewidth=0.5, shrinkA=0.0, shrinkB=0.0, edgecolor=Black))

# plot colour bars
fmt = FormatStrFormatter('%g') # '%6.1f'
ca1 = igTop.cbar_axes[0]
ca2 = igMid.cbar_axes[0]
ca3 = igBot.cbar_axes[0]
ca1.colorbar(im1, format=fmt)
ca2.colorbar(im2, format=fmt)
ca3.colorbar(im5, format=fmt)
ca1.set_ylabel(r"$u^{\prime}_{z}$ in $u_{\tau}$")
ca2.set_ylabel(r"$\overline{u^{\prime}_{z}}$ in $u_{\tau}$")
ca3.set_ylabel(r"$\widetilde{u^{\prime}_{z}}$ in $u_{\tau}$")
ca1.set_yticks([-amuz, 0.0, +amuz])
ca2.set_yticks([-amuzf, 0.0, +amuzf])
ca3.set_yticks([-amres, 0.0, +amres])

# plot mode interactive or pdf
if plot != 2:
 #plt.tight_layout()
 plt.show()
else:
 #fig.tight_layout()
 fnam = str.replace(fnam, '.h5', '.pdf')
 fnam = str.replace(fnam, 'filteredFieldGauss2d', 'plotFilteredFieldCompare')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
