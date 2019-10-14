#!/usr/bin/env python3
# Purpose:  Read pre-computed instantaneous 3d energy flux field from HDF5 file.
#           Locate global maxima and extract a 2d data set in a cross-sectional
#           plane. Plot data as colour map with optional contour lines. Output
#           is interactive or as pfd figure file.
# Usage:    python plotPiFieldCross.py 
# Authors:  Daniel Feldmann
# Date:     28th March 2019
# Modified: 01st October 2019

import sys
import timeit
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
print('Plot instantaneous energy flux in a cross-secional plane')
plot = int(input("Enter plot mode (0 = none, 1 = interactive, 2 = pdf file): "))

# some case parameters
Re_b   = 5300.0 # Bulk Reynolds number  Re_b   = u_b   * D / nu = u_cHP * R / nu 
Re_tau =  180.4 # Shear Reynolds number Re_tau = u_tau * R / nu

# path to data files (do modify)
fpath = '../40x75/'

# read energy flux field from file
fnam = fpath+'piFieldGauss2d_pipe0002_01675000.h5'
print("Reading eflux from file", fnam, "with:")
f  = h5py.File(fnam, 'r')                         # open hdf5 file for read only
r  = np.array(f['grid/r'])                        # radial co-ordinate
th = np.array(f['grid/th'])                       # azimuthal co-ordinate
z  = np.array(f['grid/z'])                        # axial co-ordainte
pi = np.array(f['fields/pi/pi']).transpose(0,2,1) # energy flux
f.close()                                         # close hdf5 file

# report grid size
nr  = len(r)
nth = len(th)
nz  = len(z)
print(nr, 'radial (r) points')
print(nth, 'azimuthal (th) points')
print(nz, 'axial (z) points')

# find global maxima of full 3d data set
pimin = np.min(pi)
index = np.where(pi == pimin)
print("Global minimum pi =", '{: 9.7f}'.format(pimin), "at (k,l,m) = (",index[0][0],',', index[1][0],',', index[2][0],')')
pimax = np.max(pi)
index = np.where(pi == pimax)
print("Global maximum pi =", '{: 9.7f}'.format(pimax), "at (k,l,m) = (",index[0][0],',', index[1][0],',', index[2][0],')')
piabs = np.max([np.abs(pimax), np.abs(pimin)])

# extract 2d data sub-set
m = 7 # 2102
print("Extract 2d data in cross sectional plane at z = ", '{:5.3f}'.format(z[m]), "R, z+ =", '{:7.1f}'.format(z[m]*Re_tau))
pi = pi[:, :, m]  

# find maxima within extracted 2d data sub-set
pimin = np.min(pi)
index = np.where(pi == pimin)
print("Local minimum pi =", '{: 9.7f}'.format(pimin), "at (k,l) = (",index[0][0],',', index[1][0],')')
pimax = np.max(pi)
index = np.where(pi == pimax)
print("Local maximum pi =", '{: 9.7f}'.format(pimax), "at (k,l) = (",index[0][0],',', index[1][0],')')
piabs = np.max([np.abs(pimax), np.abs(pimin)])

# contour levels for plotting
cln = -0.1*piabs
clp = +0.1*piabs

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
#r'\usepackage{lmodern, palatino, eulervm}',
r'\usepackage{mathptmx}',
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
fig = plt.figure(num=None, figsize=mm2inch(72.0, 48.0), dpi=300)
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

# buffer layer
b0 = np.ones(len(th))*(1.0- 5.0/180.0)
b1 = np.ones(len(th))*(1.0-30.0/180.0)

# rectangular plot of polar data
x, y = np.meshgrid(th, r)

# subplot layout
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1, 1, left=0.400, right=0.99, bottom=0.01, top=0.99)
ax1 = plt.subplot(gs[0], polar=True)
#ax2 = plt.subplot(gs[1], polar=True)

# plot eFlux in cross section
ax1.set_xticks([])
ax1.set_yticks([])
ax1.grid(False) # hide grid lines
pcm = ax1.pcolormesh(x, y, pi, vmin=-piabs, vmax=+piabs, cmap=VermBlue, shading ='gouraud')
ax1.plot(th, np.ones(len(th))*b0, color=Grey, zorder=2) # 15+
ax1.plot(th, np.ones(len(th))*b1, color=Grey, zorder=2) # 15+
ax1.contour(th, r, pi, levels=[cln], colors=Vermillion, linestyles='-', zorder=3) #, linewidths=1.1)
ax1.contour(th, r, pi, levels=[clp], colors=Blue, linestyles='-', zorder=3) # , linewidths=1.1)


# plot colour bar
axc = plt.axes([0.02, 0.10, 0.4, 0.08])
cb1 = plt.colorbar(pcm, cax=axc, orientation="horizontal", ticklocation='top')
cb1.set_ticks([-piabs, 0.0, +piabs])
cb1.set_ticklabels([r"$-$", r"$\Pi$", r"$+$"])
axc.axvline(x=0.5-0.05, color=Vermillion) # mark negative contour level in colourbar
axc.axvline(x=0.5+0.05, color=Blue)       # mark positive contour level in colourbar

# plot mode interactive or pdf
if plot != 2:
 #plt.tight_layout()
 plt.show()
else:
 #fig.tight_layout()
 fnam = str.replace(fnam, '.h5', '.pdf')
 fnam = str.replace(fnam, fpath, '')
 fnam = str.replace(fnam, 'piField', 'plotPiFieldCross')
 plt.savefig(fnam, transparent=True)
 print('Written file', fnam)
fig.clf()
