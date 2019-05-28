#!/usr/bin/env python3
# Purpose:  Computes the cross-correlation coefficient between the fluctuating velocity
#           components with the filtered interscale energy flux in axial direction.
#           Reads HDF5 files from given number of snapshots.
#           Computes the fluctuating field by subtracting the average from the statistics
#           file. Computes the filtered interscale energy flux using 2D Spectral filter.
#           Plots and prints the output in ascii format.
# Usage:    python piCorr1dGaussZ.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 28th March 2019

import sys
import os.path
import timeit
import math
import numpy as np
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

# snapshot to read flow field data from
iFile =   875000
print('Plot snapshot:', iFile)

# read grid and unfiltered flow field from hdf5 file
fnam = 'outFiles/fields_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
print('Reading from', fnam)
f  = h5py.File(fnam, 'r') # open hdf5 file for read only
r  = np.array(f['grid/r'])
z  = np.array(f['grid/z'])
th = np.array(f['grid/th'])
u_z  = np.array(f['fields/velocity/u_z']).transpose(0,2,1) 
f.close() # close hdf5 file

# read filtered flow field from hdf5 file
fnam = 'filteredFields/gauss/filteredfieldGauss_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
print('Reading from', fnam)
f  = h5py.File(fnam, 'r') # open hdf5 file for read only
u_zF = np.array(f['fields/velocity/u_zF']).transpose(0,2,1) 
f.close() # close hdf5 file

# report grid size
nr  = len(r)
nth = len(th)
nz  = len(z)
print(nr, 'radial (r) points')
print(nth, 'azimuthal (th) points')
print(nz, 'axial (z) points')

# read mean velocity profiles from ascii file
fnam = 'statistics00570000to00875000nt0062.dat'
print('Reading mean velocity profile from', fnam)
u_zM = np.loadtxt(fnam)[:, 3]

# subtract mean velocity profile (1d) from flow field (3d)
#u_z  = u_z - np.tile(u_zM, (len(z), len(th), 1)).T

# extract 2d subset 
k = 300
uz  =  u_z[:, :, k]  # wall-parallel slice at radial index k
uzF = u_zF[:, :, k]  # wall-parallel slice at radial index k
print ("Extract cross plane at z =", z[k])

# extract 1d subsets
k1 = 63 # 15+
uzk1  =  u_z[k1, :, k]  # line at radial index k1
uzFk1 = u_zF[k1, :, k]  # line at radial index k2
print ("Extract line at r =", r[k1], " (r+ = ", (1-r[k1])*180.4, ")")
k2 = 33 # 100+
uzk2  =  u_z[k2, :, k]  # line at radial index k1
uzFk2 = u_zF[k2, :, k]  # line at radial index k2
print ("Extract line at r =", r[k2], " (r+ = ", (1-r[k2])*180.4, ")")

# plot data as 2d contour plot, (1) interactive, (2) pdf
plot = 1
if plot not in [1, 2]: sys.exit() # skip everything below
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
r"\usepackage[utf8x]{inputenc}",
r"\usepackage[T1]{fontenc}",
r"\usepackage[detect-all]{siunitx}",
r'\usepackage{amsmath, amstext, amssymb}',
r'\usepackage{xfrac}',
r'\usepackage{lmodern, palatino, eulervm}']
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'font.size': 9})

# create figure suitable for A4 format
def mm2inch(*tupl):
 inch = 25.4
 if isinstance(tupl[0], tuple):
  return tuple(i/inch for i in tupl[0])
 else:
  return tuple(i/inch for i in tupl)
fig = plt.figure(num=None, figsize=mm2inch(160.0, 55.0), dpi=200)

# line colours appropriate for colour-blind
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Black         = '#000000'

# absolute maximum in this slice/sub-set
uzam = np.max(np.abs(uz))

# colour-coding
comap = 'RdGy' # colourmap
ncl   = 100   # number of countur levels

# plot unfiltered cross-section 
ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1, projection='polar')
#ax1.set_xticklabels([])
#ax1.set_yticklabels([])
cl1 = np.linspace(0, +uzam, ncl) # set contour levels manually
cf1 = ax1.contourf(th, r, uz, cl1, cmap=comap) # , extend='both')
ax1.plot(th, np.ones(len(th))*r[k1], color=Blue) # 15+
ax1.plot(th, np.ones(len(th))*r[k2], color=Vermillion) # 100+
ax1.grid(False)

# plot filtered cross-section 
#ax2 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, projection='polar')
#cl2 = np.linspace(-uzam, +uzam, ncl) # set contour levels manually
#cf2 = ax2.contourf(th, r, uzF, cl2, cmap=comap) # , extend='both')
#ax2.grid(False)

# line plot at two wall-distances
ax3 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=2)
ax3.set_xlabel(r"$r\theta$ in $R$")
ax3.set_xlim([0.0, np.max(r[k1]*th)])
ax3.set_ylabel(r"$u_{z}$ in $u_{\text{c, HP}}$")
ax3.set_ylim([0.15, 0.80])
x0 = 1.0
lambdaTh=39.0/180.4
#ax3.axvspan(x0, x0+lambdaTh, ymin=0, ymax=1, facecolor='gray')
ax3.plot(r[k1]*th, np.ones(len(th))*u_zM[k1], color=Black, linestyle='--', label=r"$\langle u_z \rangle$")
ax3.plot(r[k2]*th, np.ones(len(th))*u_zM[k2], color=Black, linestyle='--', label=r"")
ax3.plot(r[k1]*th, uzFk1+u_zM[k1], color=Black, linestyle='-', label=r"$\overline{u}_z$")
ax3.plot(r[k2]*th, uzFk2+u_zM[k2], color=Black, linestyle='-', label=r"")
ax3.plot(r[k1]*th, uzk1, color=Blue, linestyle='-', label=r"$r^{+}=15$")
ax3.plot(r[k2]*th, uzk2, color=Vermillion, linestyle='-', label=r"$r^{+}=100$")

ax3.legend(loc='best', ncol=2)



#ax1.set_aspect('equal') # makes axis ratio natural
#dvr = make_axes_locatable(ax1) # devider
#cbx = dvr.append_axes("right", size="5%", pad=0.1) # make colorbar axis
#cb1 = plt.colorbar(cf1, cax=cbx, ticks=[-uzam, 0, +uzam]) # set colorbar scale
#cb1.set_label(r"$u^{\prime}_{z}$ in $U_{\text{c, HP}}$")

# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = 'plotCrossLine_pipe0002_'+'{:08d}'.format(iFile)+'.pdf'
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()





