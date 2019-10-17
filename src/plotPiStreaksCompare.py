#!/usr/bin/env python3
# Purpose:  Read pre-computed instantaneous 3d energy flux fields from HDF5 file
#           based on a Fourier filter kernel. Read axial velocity data from the
#           corresponding HDF5 file. Read statistically steady mean profile from
#           ascii file to compute fluctuating velocity field. Extract 2d data
#           sets in a wall-parallel plane and energy flux contours in top of the
#           axial velocity field to visualise the connection between eflux and
#           high-speed and low-speed streaks. Output is interactive or as pfd
#           figure file.
# Usage:    python plotPiStreaksComapre.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 30th September 2019

import timeit
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
print('Plot instantaneous energy flux on top of streaks in a wall-parallel plane for a Fourier filter')
plot = int(input("Enter plot mode (0 = none, 1 = interactive, 2 = pdf file): "))

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# read velocity data from HDF5 file
fpath = '../../outFiles/'
fnam = fpath+'fields_pipe0002_01675000.h5'
f = h5py.File(fnam, 'r')
print("Reading radial and axial velocity field from file", fnam)
r  = np.array(f['grid/r'])    # radial co-ordinate
th = np.array(f['grid/th'])   # azimuthal co-ordinate
z  = np.array(f['grid/z'])    # axial co-ordainte
ur = np.array(f['fields/velocity/u_r']).transpose(0,2,1)
uz = np.array(f['fields/velocity/u_z']).transpose(0,2,1)
f.close()

# read mean velocity profiles from ascii file
fnam = '../../onePointStatistics/statistics00570000to01675000nt0222.dat'
print('Reading mean velocity profiles from', fnam)
uzM = np.loadtxt(fnam)[:, 3]

# subtract mean velocity profiles (1d) from flow field (3d)
uz = uz - np.tile(uzM, (len(z), len(th), 1)).T

# read Fourier energy flux field from file
fnam = '../40x75/piFieldFourier2d_pipe0002_01675000.h5'
f = h5py.File(fnam, 'r')
print("Reading eflux from file", fnam)
piF = np.array(f['fields/pi/pi']).transpose(0,2,1)
f.close()

# read Gauss energy flux field from file
fnam = str.replace(fnam, 'Fourier', 'Gauss')
f = h5py.File(fnam, 'r')
print("Reading eflux from file", fnam)
piG = np.array(f['fields/pi/pi']).transpose(0,2,1)
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
ur = ur * fu
uz = uz * fu

# report global maxima
print("Global max/min eFlux (Fourier): ", np.max(piF), np.min(piF))
print("Global max/min eFlux (Gauss):   ", np.max(piG), np.min(piG))
print("Global max/min u'_r:            ", np.max(ur), np.min(ur))
print("Global max/min u'_z:            ", np.max(uz), np.min(uz))

# extract 2d streaks and eflux data
k = 65
print ("Extracting 2d data in wall-parallel plane at y+ =", (1-r[k])*Re_tau)
piF  = piF[k, :, :]  
piG  = piG[k, :, :]  
ur2d =  ur[k, :, :]  
uz2d =  uz[k, :, :]  

# detect and extract Q events from the 2d volocity sub-set
tqs = timeit.default_timer()
print("Extracting Q events from 2d volocity field...", end='', flush=True)
q1 = np.zeros(ur2d.shape)
q2 = np.zeros(ur2d.shape)
q3 = np.zeros(ur2d.shape)
q4 = np.zeros(ur2d.shape)
for i in range(nz):
 for j in range(nth):
  if (uz2d[j,i]>0) and (ur2d[j,i]<0): q1[j,i] = ur2d[j,i]*uz2d[j,i] # outward interaction: high-speed fluid away from wall
  if (uz2d[j,i]<0) and (ur2d[j,i]<0): q2[j,i] = ur2d[j,i]*uz2d[j,i] # ejection event:       low-speed fluid away from wall
  if (uz2d[j,i]<0) and (ur2d[j,i]>0): q3[j,i] = ur2d[j,i]*uz2d[j,i] # inward interaction:   low-speed fluid towards   wall
  if (uz2d[j,i]>0) and (ur2d[j,i]>0): q4[j,i] = ur2d[j,i]*uz2d[j,i] # sweep event:         high-speed fluid towards   wall
ioi = q1 - q3 # unify inward interactions (Q3 being negativ) and outward interactions (Q1 being positive) in one array
see = q2 - q4 # unify sweep events (Q4 being negativ) and ejection events (Q2 being positiv) in one array
print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tqs), 'seconds')

# report plane global maxima
print("Global max/min eFlux (Fourier): ", np.max(piF), np.min(piF))
print("Global max/min eFlux (Gauss):   ", np.max(piG), np.min(piG))
print("Plane max/min u'_r:             ", np.max(ur2d), np.min(ur2d))
print("Plane max/min u'_z:             ", np.max(uz2d), np.min(uz2d))
print("Plane max/min Q_1:              ", np.max(q1), np.min(q1))
print("Plane max/min Q_2:              ", np.max(q2), np.min(q2))
print("Plane max/min Q_3:              ", np.max(q3), np.min(q3))
print("Plane max/min Q_4:              ", np.max(q4), np.min(q4))

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
mpl.rcParams.update({'font.size' : 7})
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
#fig = plt.figure(num=None, figsize=mm2inch(110.0, 45.0), dpi=300)
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
Grey          = '#999999'
exec(open("./colourMaps.py").read()) # 
VermBlue = CBWcm['VeBu']             # Vermillion (-) White (0) Blue (+)

# convert spatial coordiantes from outer to inner units
r = r * Re_tau
z = z * Re_tau

# manually set absolute maxima of extracted 2d data for plotting/scaling
ampi = 0.0100      # energy flux 
amuz = 7.5000      # axial velocity fluctuation
amqs = 1.0000      # Q events
clm  = 0.1000*ampi # manual contour level

# modify box for filter name annotation
filterBox = dict(boxstyle="square, pad=0.3", fc='w', ec=Black, lw=0.5)
labelBox  = dict(boxstyle="square, pad=0.2", fc='w', ec='w')

# axes grid for multiple subplots with common colour bar
from mpl_toolkits.axes_grid1 import ImageGrid
ig = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1, cbar_size=0.08, cbar_mode='edge', cbar_location='top', cbar_pad=0.08, label_mode="L", direction="row")
ax1 = ig[0]
ax2 = ig[1]
ax3 = ig[2]
ax4 = ig[3]

# plot Fourier eFlux on top of streaks
ax1.set_xlim([0.0, 1800.0])
ax1.set_ylabel(r"$\theta r^+$ in $\sfrac{\nu}{u_{\tau}}$", labelpad=-8.0)
ax1.set_ylim([0.0,  600.0])
ax1.yaxis.set_major_locator(ticker.MultipleLocator(600.0))
ax1.minorticks_on()
#im1 = ax1.imshow(piF, vmin=-ampi, vmax=+ampi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
im1 = ax1.imshow(uz2d, vmin=-amuz, vmax=+amuz, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
cl1 = ax1.contour(z, th*r[k], piF, levels=[-clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl1 = ax1.contour(z, th*r[k], piF, levels=[+clm], colors=Blue, linestyles='-', linewidths=0.5)
ax1.set_aspect('equal')
ax1.text(1750.0, 550.0, r"Fourier", ha="right", va="top", rotation=0, bbox=filterBox)
ax1.text(0.06, 0.93, r"a)", ha="right", va="top", transform=ax1.transAxes, bbox=labelBox)

# plot Fourier eFlux on top of inward/outward interactions
#ax2.set_xlabel(r"$z^+$")
ax2.set_xlim([0.0, 1800.0])
ax2.set_ylim([0.0,  600.0])
ax2.yaxis.set_major_locator(ticker.MultipleLocator(600.0))
ax2.minorticks_on()
#im2 = ax1.imshow(pi, vmin=-ampi, vmax=+ampi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
im2 = ax2.imshow(ioi, vmin=-amqs, vmax=+amqs, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
cl2 = ax2.contour(z, th*r[k], piF, levels=[-clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl2 = ax2.contour(z, th*r[k], piF, levels=[+clm], colors=Blue, linestyles='-', linewidths=0.5)
ax2.set_aspect('equal')
ax2.text(1750.0, 550.0, r"Fourier", ha="right", va="top", rotation=0, bbox=filterBox)
ax2.text(0.06, 0.93, r"b)", ha="right", va="top", transform=ax2.transAxes, bbox=labelBox)

# plot Gauss eFlux on top of streaks
ax3.set_xlabel(r"$z^+$ in $\sfrac{\nu}{u_{\tau}}$")
ax3.set_xlim([0.0, 1800.0])
ax3.xaxis.set_major_locator(ticker.MultipleLocator(500.0))
ax3.set_ylabel(r"$\theta r^+$ in $\sfrac{\nu}{u_{\tau}}$", labelpad=-8.0)
ax3.set_ylim([0.0,  600.0])
ax3.yaxis.set_major_locator(ticker.MultipleLocator(600.0))
ax3.minorticks_on()
#im3 = ax3.imshow(pi, vmin=-ampi, vmax=+ampi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
im3 = ax3.imshow(uz2d, vmin=-amuz, vmax=+amuz, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
cl3 = ax3.contour(z, th*r[k], piG, levels=[-clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl3 = ax3.contour(z, th*r[k], piG, levels=[+clm], colors=Blue, linestyles='-', linewidths=0.5)
ax3.set_aspect('equal')
ax3.text(1750.0, 550.0, r"Gauss", ha="right", va="top", rotation=0, bbox=filterBox)
ax3.text(0.06, 0.93, r"c)", ha="right", va="top", transform=ax3.transAxes, bbox=labelBox)

# plot Gauss eFlux on top of inward/outward interactions
ax4.set_xlabel(r"$z^+$ in $\sfrac{\nu}{u_{\tau}}$")
ax4.set_xlim([0.0, 1800.0])
ax4.xaxis.set_major_locator(ticker.MultipleLocator(500.0))
ax4.set_ylim([0.0,  600.0])
ax4.yaxis.set_major_locator(ticker.MultipleLocator(600.0))
ax4.minorticks_on()
#im4 = ax4.imshow(pi, vmin=-ampi, vmax=+ampi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
im4 = ax4.imshow(ioi, vmin=-amqs, vmax=+amqs, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
cl4 = ax4.contour(z, th*r[k], piG, levels=[-clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl4 = ax4.contour(z, th*r[k], piG, levels=[+clm], colors=Blue, linestyles='-', linewidths=0.5)
ax4.set_aspect('equal')
ax4.text(1750.0, 550.0, r"Gauss", ha="right", va="top", rotation=0, bbox=filterBox)
ax4.text(0.06, 0.93, r"d)", ha="right", va="top", transform=ax4.transAxes, bbox=labelBox)

# plot colour bars
fmt = FormatStrFormatter('%g') # '%6.1f'
ca1 = ig.cbar_axes[0]
ca2 = ig.cbar_axes[1]
ca1.colorbar(im1, format=fmt)
ca2.colorbar(im2, format=fmt)
ca1.toggle_label(True)
ca2.toggle_label(True)
#ca1.set_xlabel(r"$u^{\prime}_{z}$ in $U_{c}$", labelpad=-6.0) # outer units
#ca1.set_xlabel(r"$u^{\prime}_{z}$ in $u_{\tau}$", labelpad=-6.0) # inner units
ca2.set_xlabel(r"In- \& Outward interactions", labelpad=-6.0)
ca1.set_xticks([-6.0, -3.0, 0.0, 3.0, 6.0])
ca2.set_xticks([-amqs, +amqs])
ca1.set_xticklabels([-6, -3, r"$u^{\prime}_{z}$ in $u_{\tau}$", 3, 6])
ca2.set_xticklabels([r"$Q_3$", r"$Q_1$"])

# plot mode interactive or pdf
if plot != 2:
 #plt.tight_layout()
 plt.show()
else:
 #fig.tight_layout()
 fnam = str.replace(fnam, '.h5', '.pdf')
 fnam = str.replace(fnam, 'piFieldGauss2d', 'plotPiStreaksCompare')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
