#!/usr/bin/env python3
#====================================================================================
# Purpose:  Computes the cross-correlation coefficient between the fluctuating velocity
#           components with the filtered interscale energy flux in axial direction.
#           Reads HDF5 files from given number of snapshots.
#           Computes the fluctuating field by subtracting the average from the statistics
#           file. Computes the filtered interscale energy flux using 2D Spectral filter.
#           Plots and prints the output in ascii format.
# ----------------------------------------------------------------------------------
# IMPORTANT:Make sure the statistics file should correspond to the given number
#           of snapshots 
# ----------------------------------------------------------------------------------
# Usage:    python piCorr1dFourierZ.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 28th March 2019
# ===================================================================================
import sys
import os.path
import timeit
import math
import numpy as np
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

# state file to read from
iFile =  570000

# read grid from first hdf5 file
fnam = '../../outFiles/fields_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
print('Reading grid from', fnam, 'with:')
f  = h5py.File(fnam, 'r') # open hdf5 file for read only
r  = np.array(f['grid/r'])
th = np.array(f['grid/th'])
z  = np.array(f['grid/z'])
f.close() # close hdf5 file

# grid size
nr  = len(r)
nth = len(th)
nz  = len(z)
print(nr, 'radial (r) points')
print(nth, 'azimuthal (th) points')
print(nz, 'axial (z) points')

# define filter width in units of length, for each direction seperately
lambdaRp  =  20   # wall-normal radial direction (r)
lambdaThp =  40   # cross-stream azimuthal direction (theta)
lambdaZp  =  75   # streamwise axial direction (z)
ReTau     = 180.4 # shear Reynolds number for Re=5300 acc. to Blasius
lambdaR   = lambdaRp/ReTau
lambdaTh  = lambdaThp/ReTau
lambdaZ   = lambdaZp/ReTau
print('Filter width in r:  lambdaR+  =', '{:6.1f}'.format(lambdaRp),  'viscous units, lambdaR  =', '{:7.5f}'.format(lambdaR),  'R')
print('Filter width in th: lambdaTh+ =', '{:6.1f}'.format(lambdaThp), 'viscous units, lambdaTh =', '{:7.5f}'.format(lambdaTh), 'R')
print('Filter width in z:  lambdaZ+  =', '{:6.1f}'.format(lambdaZp),  'viscous units, lambdaZ  =', '{:7.5f}'.format(lambdaZ),  'R')

print(r[0], r[-1])

# the actual cut-off wave-length in plus units
lambdaThCp = np.zeros(len(r))

for k in range(0, len(r)):
 rPos = r[k]
 # print(rPos)

 # sample spacing in each direction in unit length (pipe radii R, gap width d, etc)
 deltaTh = (th[1] - th[0]) * rPos # equidistant but r dependent

 # set-up wavenumber vector in units of cycles per unit distance (1/R)
 kTh = np.fft.fftfreq(len(th), d=deltaTh) # homogeneous direction theta

 # compute cut-off wave number in units of radian per unit length as defined in Pope, p. 
 kappaThC = np.pi/lambdaTh # azimuthal direction theta

 # construct 2d rectangular Fourier filter kernel
 gTh   = np.heaviside(kappaThC - abs(2.0*np.pi*kTh), 1)

 idx1 = np.where(gTh>0)
 gTh1 = gTh[idx1]
 idxc = len(gTh1)
 lambdaThCp[k] = 1.0/kTh[idxc]*180.4
 print('k=', k, 'r=', r[k], 'Cut-off index', idxc, lambdaThCp[k])


# plot data as graph, (0) none, (1) interactive, (2) pdf
plot = 1
if plot not in [1, 2]: sys.exit() # skip everything below
import matplotlib.pyplot as plt
import matplotlib as mpl
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
fig = plt.figure(num=None, figsize=mm2inch(240.0, 160.0), dpi=150)

# line colours appropriate for colour-blind
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Black         = '#000000'

# plot
ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
ax1.set_xlabel(r"$r$ in $R$")
ax1.set_xlim([0,1])
ax1.set_ylabel(r"$\lambda_{\theta}^{+}$ in $\sfrac{\nu}{u_{\tau}}$")
ax1.set_ylim([0,80])
ax1.plot(r, lambdaThp*np.ones(len(r)), label=r"Nominal")
ax1.plot(r, lambdaThCp, linestyle='-', marker="o", label=r"Actual")

# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = 'filterCutOff.pdf'
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
                             


