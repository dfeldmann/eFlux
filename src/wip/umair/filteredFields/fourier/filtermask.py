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

# compute cut-off wave number in units of radian per unit length as defined in Pope, p. 
kappaThC = np.pi/lambdaTh # azimuthal direction theta
kappaZC  = np.pi/lambdaZ  # axial direction z

# construct 2d rectangular Fourier filter kernel
gTh   = np.heaviside(kappaThC - abs(2.0*np.pi*kTh), 1) # 1d step function in theta direction
gZ    = np.heaviside(kappaZC  - abs(2.0*np.pi*kZ),  1) # 1d step function in z direction
g2dFr = np.outer(gTh, gZ)                # 2d rectangular step function in theta-z-plane

# construct 2d elliptical Fourier filter kernel
g2dFe = np.ones((len(gTh), len(gZ))) # * np.random.random_sample()          # 2d rectangular step function in theta-z-plane


def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask




mask = sector_mask(g2dFe.shape,(0,0),20,(0,359))
g2dFe[~mask] = 0


#for j in range (len(kTh)-1):
#  for i in range (len(kZ)-1):
#    a = 2
#    b = 1
#    ellipse = (kTh[j]/a)**2 + (kZ[i]/b)**2
#    if ellipse <= 2:
#      g2dFe[j,i] = 1
#    else:
#      g2dFe[j,i] = 0
#print (g2dFe[0,0], g2dFr[0,0])


# construct 2d elliptical Gauss filter kernel
gTh   = np.exp(-((2.0*np.pi*kTh)**2.0*lambdaTh**2/24.0)) # Gauss exp kernel in theta direction
gZ    = np.exp(-((2.0*np.pi*kZ )**2.0*lambdaZ**2 /24.0)) # Gauss exp kernel in axial direction
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

# plot elliptical Fourier kernel
ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=1)
ax1.set_ylabel(r"$k_{\theta}$ in $\sfrac{1}{R}$")
am = np.max(np.abs(g2dFe)) # absolute maximum in this data-set
cl1 = np.linspace(-am, +am, ncl) # set contour levels manually
cf1 = ax1.contourf(kZ, kTh, g2dFe, cl1, cmap=comap) # , extend='both')
ax1.set_aspect('equal') # makes axis ratio natural
dvr = make_axes_locatable(ax1) # devider
cbx = dvr.append_axes("top", size="5%", pad=0.5) # make colorbar axis
cb1 = plt.colorbar(cf1, cax=cbx, ticks=[-am, 0, +am], orientation="horizontal") # set colorbar scale
cb1.set_label(r"Fourier (elliptical)")

# plot Gaus kernel
ax2 = plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=1, sharex=ax1, sharey=ax1)
am = np.max(np.abs(g2dG)) # absolute maximum in this data-set
cl2 = np.linspace(-am, +am, ncl) # set contour levels manually
cf2 = ax2.contourf(kZ, kTh, g2dG, cl2, cmap=comap) # , extend='both')
ax2.set_aspect('equal') # makes axis ratio natural
dvr = make_axes_locatable(ax2) # devider
cbx = dvr.append_axes("top", size="5%", pad=0.5) # make colorbar axis
cb2 = plt.colorbar(cf2, cax=cbx, ticks=[-am, 0, +am], orientation="horizontal") # set colorbar scale
cb2.set_label(r"Gauss (elliptical)")

# plot elliptical box kernel
ax3 = plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1, sharex=ax1, sharey=ax1)
am = np.max(np.abs(g2dBe)) # absolute maximum in this data-set
ax3.set_ylabel(r"$k_{\theta}$ in $\sfrac{1}{R}$")
am = np.max(np.abs(g2dFr)) # absolute maximum in this data-set
cl3 = np.linspace(-am, +am, ncl) # set contour levels manually
cf3 = ax3.contourf(kZ, kTh, g2dBe, cl3, cmap=comap) # , extend='both')
ax3.set_aspect('equal') # makes axis ratio natural
dvr = make_axes_locatable(ax3) # devider
cbx = dvr.append_axes("top", size="5%", pad=0.5) # make colorbar axis
cb3 = plt.colorbar(cf3, cax=cbx, ticks=[-am, 0, +am], orientation="horizontal") # set colorbar scale
cb3.set_label(r"Box (elliptici)")

# plot rectangular Fourier kernel
ax4 = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=1, sharex=ax1, sharey=ax1)
ax4.set_xlabel(r"$k_{z}$ in $\sfrac{1}{R}$")
ax4.set_ylabel(r"$k_{\theta}$ in $\sfrac{1}{R}$")
am = np.max(np.abs(g2dFr)) # absolute maximum in this data-set
cl4 = np.linspace(-am, +am, ncl) # set contour levels manually
cf4 = ax4.contourf(kZ, kTh, g2dFr, cl4, cmap=comap) # , extend='both')
#ax4.xhline(y=0.0, xmin=0.0, xmax=lambdaZ)
ax4.set_aspect('equal') # makes axis ratio natural
dvr = make_axes_locatable(ax4) # devider
cbx = dvr.append_axes("top", size="5%", pad=0.5) # make colorbar axis
cb4 = plt.colorbar(cf4, cax=cbx, ticks=[-am, 0, +am], orientation="horizontal") # set colorbar scale
cb4.set_label(r"Fourier (rectangular)")


# plot elliptical kernel

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
                             


