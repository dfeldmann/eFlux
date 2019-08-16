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

# just pic a plane
k = 63
rPos = r[k]
print ("Wall-normal plane at y+ =", (1-rPos)*180.4)

# sample spacing in each direction in unit length (pipe radii R, gap width d, etc)
deltaTh = (th[1] - th[0]) * rPos # equidistant but r dependent
deltaZ  = ( z[1] -  z[0])        # equidistant

# set-up wavenumber vector in units of cycles per unit distance (1/R)
kTh = np.fft.fftfreq(len(th), d=deltaTh) # homogeneous direction theta
kZ  = np.fft.fftfreq(len(z),  d=deltaZ)  # homogeneous direction z

idm = np.round(len(th)/2).astype(int)
print(kTh[0], kTh[idm-1], kTh[idm], kTh[idm+1], kTh[-1])

# compute cut-off wave number in units of radian per unit length as defined in Pope, p. 
kappaThC = np.pi/lambdaTh # azimuthal direction theta
kappaZC  = np.pi/lambdaZ  # axial direction z

# construct 2d rectangular Fourier filter kernel
gTh   = np.heaviside(kappaThC - abs(2.0*np.pi*kTh), 1) # 1d step function in theta direction
gZ    = np.heaviside(kappaZC  - abs(2.0*np.pi*kZ),  1) # 1d step function in z direction
g2dFr = np.outer(gTh, gZ)                # 2d rectangular step function in theta-z-plane

# construct 2d elliptical Fourier filter kernel
g2dFe = np.zeros((len(kTh), len(kZ)))
print(len(kTh), len(kZ))
print(g2dFe.shape)
for j in range(0, len(kTh)):
 for i in range(0, len(kZ)):
  ellipse = (2.0*np.pi*kTh[j]/kappaThC)**2.0 + (2.0*np.pi*kZ[i]/kappaZC)**2.0
  if ellipse <= 1:
   g2dFe[j,i] = 1.0

# construct 2d elliptical Gauss filter kernel
gTh   = np.exp(-((2.0*np.pi*kTh * lambdaTh)**2.0/24.0)) # Gauss exp kernel in theta direction
gZ    = np.exp(-((2.0*np.pi*kZ  * lambdaZ )**2.0/24.0)) # Gauss exp kernel in axial direction
g2dG  = np.outer(gTh, gZ)

# construct 2d elliptic box filter kernel
gTh   = np.sinc(kTh*np.pi*lambdaTh) # Box sinc kernel in theta direction
gZ    = np.sinc(kZ*np.pi*lambdaZ)   # Box sinc kernel in axial direction
g2dBe = np.outer(gTh, gZ)           # 2d filter kernel in theta-z plane

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

# colour-coding
comap = 'RdGy_r' # colourmap
ncl   = 128    # number of countur levels

# roll the wave number vectors only for plotting
rollTh = np.round(len(th)/2).astype(int)
rollZ  = np.round(len(z)/2).astype(int)
x = np.roll(kZ, rollZ, axis=0)
y = np.roll(kTh, rollTh, axis=0)
print(y[0], y[idm-1], y[idm], y[idm+1], y[-1])

# plot elliptical Fourier kernel
ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
ax1.set_xlabel(r"$k_{z}$ in $\sfrac{1}{R}$")
#ax1.set_xlim([-4, 4])
ax1.set_ylabel(r"$k_{\theta}$ in $\sfrac{1}{R}$")
#ax1.set_ylim([-6, 6])
am = np.max(np.abs(g2dFe)) # absolute maximum in this data-set
cl1 = np.linspace(-am, +am, ncl) # set contour levels manually
z = np.roll(np.roll(g2dFe, rollZ, axis=1), rollTh, axis=0) # roll the kernel only for plotting
#cf1 = ax1.contourf(x, y, z, cl1, cmap=comap)
cf1 = ax1.imshow(z, cmap=comap, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], vmax=am, vmin=-am)
ax1.annotate(r"$\sfrac{1}{\lambda_{z}}$", xy=(0.0, 0.0), xycoords='data', xytext=(1.0/lambdaZ, 0.0), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"), horizontalalignment='left', verticalalignment='center')
ax1.annotate(r"$\sfrac{1}{\lambda_{\theta}}$", xy=(0.0, 0.0), xycoords='data', xytext=(0.0, 1.0/lambdaTh), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"), horizontalalignment='center', verticalalignment='bottom')
ax1.set_aspect('equal') # makes axis ratio natural
dvr = make_axes_locatable(ax1) # devider
cbx = dvr.append_axes("top", size="5%", pad=0.5) # make colorbar axis
cb1 = plt.colorbar(cf1, cax=cbx, ticks=[-am, 0, +am], orientation="horizontal") # set colorbar scale
cb1.set_label(r"Fourier (elliptical)")

# plot elliptical Gauss kernel
ax2 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1, sharex=ax1, sharey=ax1)
ax2.set_xlabel(r"$k_{z}$ in $\sfrac{1}{R}$")
am = np.max(np.abs(g2dG)) # absolute maximum in this data-set
cl2 = np.linspace(-am, +am, ncl) # set contour levels manually
z = np.roll(np.roll(g2dG, rollZ, axis=1), rollTh, axis=0) # roll the kernel only for plotting
#cf2 = ax2.contourf(x, y, z, cl2, cmap=comap)
cf2 = ax2.imshow(z, cmap=comap, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], vmax=am, vmin=-am)
ax2.annotate(r"$\sfrac{1}{\lambda_{z}}$", xy=(0.0, 0.0), xycoords='data', xytext=(1.0/lambdaZ, 0.0), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"), horizontalalignment='left', verticalalignment='center')
ax2.annotate(r"$\sfrac{1}{\lambda_{\theta}}$", xy=(0.0, 0.0), xycoords='data', xytext=(0.0, 1.0/lambdaTh), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"), horizontalalignment='center', verticalalignment='bottom')
ax2.set_aspect('equal') # makes axis ratio natural
dvr = make_axes_locatable(ax2) # devider
cbx = dvr.append_axes("top", size="5%", pad=0.5) # make colorbar axis
cb2 = plt.colorbar(cf2, cax=cbx, ticks=[-am, 0, +am], orientation="horizontal") # set colorbar scale
cb2.set_label(r"Gauss (elliptical)")

# plot elliptical Box kernel
ax3 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1, sharex=ax1, sharey=ax1)
ax3.set_xlabel(r"$k_{z}$ in $\sfrac{1}{R}$")
am = np.max(np.abs(g2dBe)) # absolute maximum in this data-set
cl3 = np.linspace(-am, +am, ncl) # set contour levels manually
z = np.roll(np.roll(g2dBe, rollZ, axis=1), rollTh, axis=0) # roll the kernel only for plotting
cf3 = ax3.imshow(z, cmap=comap, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], vmax=am, vmin=-am)
ax3.annotate(r"$\sfrac{1}{\lambda_{z}}$", xy=(0.0, 0.0), xycoords='data', xytext=(1.0/lambdaZ, 0.0), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"), horizontalalignment='left', verticalalignment='center')
ax3.annotate(r"$\sfrac{1}{\lambda_{\theta}}$", xy=(0.0, 0.0), xycoords='data', xytext=(0.0, 1.0/lambdaTh), textcoords='data',
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"), horizontalalignment='center', verticalalignment='bottom')
ax3.set_aspect('equal') # makes axis ratio natural
dvr = make_axes_locatable(ax3) # devider
cbx = dvr.append_axes("top", size="5%", pad=0.5) # make colorbar axis
cb3 = plt.colorbar(cf3, cax=cbx, ticks=[-am, 0, +am], orientation="horizontal") # set colorbar scale
cb3.set_label(r"Box (elliptical)")

# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = 'filterKernel'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.pdf'
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
                             


