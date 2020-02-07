#!/usr/bin/env python3
# Purpose:  Read pre-computed instantaneous 3d energy flux fields from HDF5 file
#           based on a Gauss filter kernel. Read radial and axial velocity
#           data from the corresponding HDF5 file. Read statistically steady
#           mean profile from ascii file to compute the fluctuating axial
#           velocity field. Extract 2d data sub-sets in a wall-parallel plane
#           and detect inward (Q3) and outward (Q1) interacations in the
#           sub-set. Plot energy flux contours on top of the inward and outward
#           interactions to visualise the connection between them. Output is
#           interactive or as pfd figure file.
# Usage:    python plotPiQ3Q1Gauss.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 30th September 2019

import timeit
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
print('Plot instantaneous energy flux on top of inward (Q3) and outward (Q1) interactions in a wall-parallel plane for a Gauss filter')
plot = int(input("Enter plot mode (0 = none, 1 = interactive, 2 = pdf file): "))

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# read velocity data from HDF5 file
fpath = '../../outFiles/'
fnam = fpath+'field_pipe0002_01675000.h5'
f = h5py.File(fnam, 'r')
print("Reading radial and axial velocity field from file", fnam)
r  = np.array(f['grid/r']) # read grid only once
z  = np.array(f['grid/z'])
th = np.array(f['grid/th'])
ur = np.array(f['fields/velocity/u_r']).transpose(0,2,1)
uz = np.array(f['fields/velocity/u_z']).transpose(0,2,1)
f.close()

# read mean velocity profiles from ascii file
fnam = '../../onePointStatistics/statistics00570000to04070000nt0351.dat'
print('Reading mean velocity profiles from', fnam)
uzM = np.loadtxt(fnam)[:, 3]

# subtract mean velocity profiles (1d) from flow field (3d)
uz = uz - np.tile(uzM, (len(z), len(th), 1)).T

# read Gauss energy flux field from file
fnam = 'piFieldGauss2d_pipe0002_01675000.h5'
f = h5py.File(fnam, 'r')
print("Reading eflux from file", fnam)
pi = np.array(f['fields/pi/pi']).transpose(0,2,1)
f.close()

# grid size
nr  = len(r)
nth = len(th)
nz  = len(z)
print('With', nr, 'radial (r) points')
print('With', nth, 'azimuthal (th) points')
print('With', nz, 'axial (z) points')
print('It is your responsibility to make sure that both fields are defined on the exact same grid.')

# report global maxima
print("Global max/min eFlux:", np.max(pi), np.min(pi))
print("Global max/min u'_r: ", np.max(ur), np.min(ur))
print("Global max/min u'_z: ", np.max(uz), np.min(uz))

# extract 2d streaks and eflux data
k = 65
print ("Extracting 2d data in wall-parallel plane at y+ =", (1-r[k])*Re_tau)
pi = pi[k, :, :]  
ur = ur[k, :, :]  
uz = uz[k, :, :]  

# detect inward (Q3) and outward (Q1) interactions within the sub-set
print("Detecting inward (Q3) and outward (Q1) interactions...")
q1 = np.zeros(uz.shape)
q3 = np.zeros(uz.shape)
for i in range(nz):
 for j in range(nth):
  if (uz[j, i] > 0) and (ur[j, i] < 0): q1[j, i] = uz[j, i] * ur[j, i] # Q1 (outward) interaction: high-speed fluid away from wall
  if (uz[j, i] < 0) and (ur[j, i] > 0): q3[j, i] = uz[j, i] * ur[j, i] # Q3 (inward) interaction:   low-speed fluid toward the wall
io = q1 - q3 # unify inward interactions (Q3 being negativ) and outward interactions (Q1 being positive) in one array

# report plane global maxima
print("Plane max/min eFlux:", np.max(pi), np.min(pi))
print("Plane max/min u'_r: ", np.max(ur), np.min(ur))
print("Plane max/min u'_z: ", np.max(uz), np.min(uz))
print("Plane max/min Q3 i: ", np.max(q3), np.min(q3))
print("Plane max/min Q1 o: ", np.max(q1), np.min(q1))

# plotting
if plot not in [1, 2]: sys.exit() # skip everything below
print('Creating plot (using LaTeX)...')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
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
exec(open("./colourMaps.py").read()) # many thanks to github.com/nesanders/colorblind-colormap 
VermBlue = CBWcm['VeBu']             # Vermillion (-) White (0) Blue (+)

# find absolute maxima of extracted 2d data for plotting/scaling
ampi = np.max(np.abs(pi)) # eFlux
ampi = 0.0100             # manual max
amq3 = np.max(np.abs(q3)) # inward interaaction
amq1 = np.max(np.abs(q1)) # outward interaction
amio = np.max(np.abs(io)) # unifyied in-out interactions
amio = 0.3*amio # 0.0020  # manual max
amur = np.max(np.abs(ur)) # radial velocity fluctuation
amuz = np.max(np.abs(uz)) # axial velocity fluctuation
clm  =-0.0020             # manual contour level

# convert spatial coordiantes from outer to inner units
r = r * Re_tau
z = z * Re_tau

# modify box for filter name annotation
filterBox = dict(boxstyle="square, pad=0.3", fc='w', ec=Black, lw=0.5)

# plot Gauss eFlux
ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
ax1.set_xlabel(r"$z^+$")
ax1.set_xlim([0.0, 1800.0])
ax1.set_ylabel(r"$\theta r^+$")
ax1.set_ylim([0.0,  600.0])
#im1 = ax1.imshow(pi, vmin=-ampi, vmax=+ampi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
im1 = ax1.imshow(io, vmin=-amio, vmax=+amio, cmap=VermBlue, interpolation='bilinear', extent=[np.min(z), np.max(z), np.min(th*r[k]), np.max(th*r[k])], origin='lower')
cl1 = ax1.contour(z, th*r[k], pi, levels=[clm], colors=Vermillion, linestyles='-', linewidths=0.5)
cl1 = ax1.contour(z, th*r[k], pi, levels=[-clm], colors=Blue, linestyles='-', linewidths=0.5)
ax1.set_aspect('equal')
t1 = ax1.text(1750.0, 550.0, r"Gauss", ha="right", va="top", rotation=0, bbox=filterBox)

# plot colour bar
axd = make_axes_locatable(ax1) # divider
axc = axd.append_axes("right", size=0.10, pad=0.05) # add an axes right of the main axes
cb1 = plt.colorbar(im1, cax=axc, orientation='vertical') 
cb1.set_label(r"In- \& Outward interactions")
cb1.set_ticks([-amio, +amio])
cb1.set_ticklabels([r"$Q_3$", r"$Q_1$"])

# plot mode interactive or pdf
if plot != 2:
 #plt.tight_layout()
 plt.show()
else:
 #fig.tight_layout()
 fnam = str.replace(fnam, '.h5', '.pdf')
 fnam = str.replace(fnam, 'piFieldGauss2d', 'plotPiQ3Q1Gauss')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
