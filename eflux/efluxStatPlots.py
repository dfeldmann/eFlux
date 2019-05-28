#!/usr/bin/env python3
#

import sys
import os.path
import timeit
import math
import numpy as np
import h5py


# range of state files to read flow field data from
iFirst =   570000
iLast  =   1005000
iStep  =   5000
nt = int((iLast -iFirst)/iStep + 1.0)
iFiles = range(iFirst, iLast+iStep, iStep)
print('eFlux Statistical data from', len(iFiles), 'snapshots:', iFiles[0], 'to', iFiles[-1])

# read mean velocity profiles from ascii file
box = 'piStatBox2d00570000to01005000nt0088.dat'
print('Reading eFlux statistics from', box)
r = np.loadtxt(box)[:, 0]
eFMeanBox  = np.loadtxt(box)[:, 1]
eFRmsBox   = np.loadtxt(box)[:, 2]
eFSkewBox  = np.loadtxt(box)[:, 3]
eFFlatBox  = np.loadtxt(box)[:, 4]

gauss = 'piStatGauss2d00570000to01005000nt0088.dat'
print('Reading eFlux statistics from', gauss)
r = np.loadtxt(gauss)[:, 0]
eFMeanGauss = np.loadtxt(gauss)[:, 1]
eFRmsGauss  = np.loadtxt(gauss)[:, 2]
eFSkewGauss = np.loadtxt(gauss)[:, 3]
eFFlatGauss = np.loadtxt(gauss)[:, 4]

fourier = 'piStatFourier2d00570000to01005000nt0088.dat'
print('Reading eFlux statistics from', fourier)
r = np.loadtxt(fourier)[:, 0]
eFMeanFourier = np.loadtxt(fourier)[:, 1]
eFRmsFourier  = np.loadtxt(fourier)[:, 2]
eFSkewFourier = np.loadtxt(fourier)[:, 3]
eFFlatFourier = np.loadtxt(fourier)[:, 4]


# plot data as graph, (0) none, (1) interactive, (2) pdf
plot =2
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
#fig = plt.figure(num=None, figsize=mm2inch(210.0, 297.0), dpi=600)
fig = plt.figure(num=None, figsize=mm2inch(160.0, 140.0), dpi=150)
#fig = plt.figure(num=None, figsize=mm2inch(90.0, 70.0), dpi=150)

# line colours appropriate for colour-blind
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Black         = '#000000'

# plot first order statistics
rp=(1.0-r)*180.4
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax1.set_xlabel(r"$r$ in $R$")
ax1.set_ylabel(r"$\langle \Pi \rangle_{\theta, z, t}$")
ax1.plot(rp, eFMeanFourier, color=BluishGreen, linestyle='-',  label=r"Fourier")
ax1.plot(rp, eFMeanGauss,   color=Vermillion,  linestyle='-',  label=r"$Gauss$")
ax1.plot(rp, eFMeanBox,     color=Blue,        linestyle='-',  label=r"$Box$")
#ax1.legend(loc='best', ncol=4)
ax1.legend(loc='best')

# plot second order statistics
ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax2.set_xlabel(r"$r$ in $R$")
ax2.set_ylabel(r"$\text{RMS}\left(\Pi^{\lambda}\right)$")
ax2.plot(rp, eFRmsFourier, color=BluishGreen, linestyle='-',  label=r"Fourier")
ax2.plot(rp, eFRmsGauss,   color=Vermillion,  linestyle='-',  label=r"$Gauss$")
ax2.plot(rp, eFRmsBox,     color=Blue,        linestyle='-',  label=r"$Box$")
#ax2.legend(loc='best', ncol=4)
ax2.legend(loc='best')

# plot third order statistics
ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax3.set_xlabel(r"$r$ in $R$")
ax3.set_ylabel(r"$\text{Skewness}\left(\Pi^{\lambda}\right)$ in $\text{RMS}^{3}$")
ax3.plot(rp, eFSkewFourier, color=BluishGreen, linestyle='-',  label=r"Fourier")
ax3.plot(rp, eFSkewGauss,   color=Vermillion,  linestyle='-',  label=r"$Gauss$")
ax3.plot(rp, eFSkewBox,     color=Blue,        linestyle='-',  label=r"$Box$")
#ax3.legend(loc='best', ncol=4)
ax3.legend(loc='best')

# plot fourth order statistics
ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
ax4.set_xlabel(r"$r$ in $R$")
ax4.set_ylabel(r"$\text{Flatness}\left(\Pi^{\lambda}\right)$ in $\text{RMS}^{4}$")
ax4.plot(rp, eFFlatFourier, color=BluishGreen, linestyle='-',  label=r"Fourier")
ax4.plot(rp, eFFlatGauss,   color=Vermillion,  linestyle='-',  label=r"$Gauss$")
ax4.plot(rp, eFFlatBox,     color=Blue,        linestyle='-',  label=r"$Box$")
#ax4.legend(loc='best', ncol=4)
ax4.legend(loc='best')


# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = 'eFluxStats'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.pdf'
 plt.savefig(fnam)
 print('Written file', fnam)

fig.clf()
