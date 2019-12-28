#!/usr/bin/env python3
# Purpose:  Read full 3d pipe flow velocity fields from one or more HDF5 files.
#           Read the computational grid in cylindrical co-ordinates from one of
#           these files. Read a statistically converged mean velocity profile
#           from a separate ascii file and subtract it from the each snapshot to
#           obtain the fluctuating velocity field. Define filter widths and
#           compute the inter-scale turbulent kinetic energy flux field (Pi) for
#           each individual snapshot based on a two-dimensional spatial box
#           filter operation in wall-parallel planes for each radial location.
#           Compute one point statistics (mean, rms, skewness, flatness) for Pi
#           based on temporal (ensemble) averaging over individual snapshots and
#           additional averaging in the two homogeneous directions (theta, z).
#           Write the resulting 1d (radial) profiles to ascii file and,
#           optionally, plot profiles interactively or as pfd figure file.
# Usage:    python piStatBox2d.py
# Authors:  Daniel Feldmann, Jan Chen
# Date:     12th July 2018
# Modified: 25th September 2019

import timeit
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
plot = 0

# range of state files to read from flow field data
iFirst =  570000
iLast  = 1675000
iStep  =    5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('Compute energy flux one-point statistics from', len(iFiles), 'snapshot(s):', iFiles[0], 'to', iFiles[-1])

# path to data files (do modify)
fpath = '../../outFiles/'

# read grid from first HDF5 file
fnam = fpath+'fields_pipe0002_'+'{:08d}'.format(iFirst)+'.h5'
print('Reading grid from', fnam, 'with:')
f  = h5py.File(fnam, 'r')    # open hdf5 file for read only
r  = np.array(f['grid/r'])   # radial co-ordinate
th = np.array(f['grid/th'])  # azimuthal co-ordinate
z  = np.array(f['grid/z'])   # axial co-ordainte
f.close()                    # close hdf5 file

# report grid size
nr  = len(r)
nth = len(th)
nz  = len(z)
print(nr, 'radial (r) points')
print(nth, 'azimuthal (th) points')
print(nz, 'axial (z) points')

# read mean velocity profiles from ascii file (do modify)
fnam = '../../onePointStatistics/statistics00570000to01675000nt0222.dat'
print('Reading mean velocity profile from', fnam)
rM   = np.loadtxt(fnam)[:, 0] # 1st column r
u_zM = np.loadtxt(fnam)[:, 3] # 4th column <u_z>

# define filter width for each direction seperately
lambdaThp =  40   # cross-stream azimuthal direction (theta)
lambdaZp  =  75   # streamwise axial direction (z)
ReTau     = 180.4 # shear Reynolds number for Re=5300 acc. to Blasius
lambdaTh  = lambdaThp/ReTau
lambdaZ   = lambdaZp/ReTau
print('Filter width in th: lambdaTh+ =', '{:6.1f}'.format(lambdaThp), 'viscous units, lambdaTh =', '{:7.5f}'.format(lambdaTh), 'R')
print('Filter width in z:  lambdaZ+  =', '{:6.1f}'.format(lambdaZp),  'viscous units, lambdaZ  =', '{:7.5f}'.format(lambdaZ),  'R')

# parallel stuff
import multiprocessing
from joblib import Parallel, delayed
print("Running on", multiprocessing.cpu_count(), "cores")

# prepare variables for eFlux statisitcs
pi1 = np.zeros(nr) # Pi first moment
pi2 = np.zeros(nr) # Pi second moment
pi3 = np.zeros(nr) # Pi third moment
pi4 = np.zeros(nr) # Pi fourth moment
nt  = len(iFiles)  # ensemble counter

# reset wall-clock time
t0 = timeit.default_timer()

# statistics loop over all state files
for iFile in iFiles:

    # read flow field data from next HDF5 file
    fnam = fpath+'fields_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    f = h5py.File(fnam, 'r')
    print("Reading velocity field from file", fnam, end='', flush=True)
    u_r  = np.array(f['fields/velocity/u_r']).transpose(0,2,1)  # do modify transpose to match data structures
    u_th = np.array(f['fields/velocity/u_th']).transpose(0,2,1) # filter functions were made for u[r, th, z]
    u_z  = np.array(f['fields/velocity/u_z']).transpose(0,2,1)
    step=f['grid'].attrs.__getitem__('step')
    timeIn=f['grid'].attrs.__getitem__('time')
    Re=f['setup'].attrs.__getitem__('Re')
    f.close()
    print(' with data structure u',u_z.shape)

    # subtract mean velocity profile (1d) to obtain full (3d) fluctuating velocity field
    u_z = u_z - np.tile(u_zM, (nz, nth, 1)).T

    # filter velocity field
    print('Filtering velocity components and mixed terms... ', end='', flush=True)
    tfilter = timeit.default_timer()
    import filter2d as f2
    u_rF    = f2.box2d(u_r,       lambdaTh, lambdaZ, r, th, z)
    u_thF   = f2.box2d(u_th,      lambdaTh, lambdaZ, r, th, z)
    u_zF    = f2.box2d(u_z,       lambdaTh, lambdaZ, r, th, z)
    u_rRF   = f2.box2d(u_r*u_r,   lambdaTh, lambdaZ, r, th, z)
    u_rThF  = f2.box2d(u_r*u_th,  lambdaTh, lambdaZ, r, th, z)
    u_rZF   = f2.box2d(u_r*u_z,   lambdaTh, lambdaZ, r, th, z)
    u_thThF = f2.box2d(u_th*u_th, lambdaTh, lambdaZ, r, th, z)
    u_thZF  = f2.box2d(u_th*u_z,  lambdaTh, lambdaZ, r, th, z)
    u_zZF   = f2.box2d(u_z*u_z,   lambdaTh, lambdaZ, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tfilter), 'seconds')

    # compute instantaneous energy flux
    tflux = timeit.default_timer()
    print('Computing energy flux... ', end='', flush=True)
    import eFlux
    pi = eFlux.eFlux(u_rF, u_thF, u_zF, u_rRF, u_rThF, u_rZF, u_thThF, u_thZF, u_zZF, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tflux), 'seconds')

    # sum up statistics in time and homogeneous directions (theta, z)
    tstat = timeit.default_timer()
    print('Spatial averaging in homogeneous directions (theta, z)... ', end='', flush=True)
    pi1 = pi1 + np.sum(np.sum(pi,    axis=1), axis=1)
    pi2 = pi2 + np.sum(np.sum(pi**2, axis=1), axis=1)
    pi3 = pi3 + np.sum(np.sum(pi**3, axis=1), axis=1)
    pi4 = pi4 + np.sum(np.sum(pi**4, axis=1), axis=1)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tstat), 'seconds')

# divide by total number of spatio-temporal samples
pi1 = pi1/(nth*nz*nt)
pi2 = pi2/(nth*nz*nt)
pi3 = pi3/(nth*nz*nt)
pi4 = pi4/(nth*nz*nt)

# compute mean, RMS, skewness and flatness factors
piMean = pi1
piRms  = np.sqrt(pi2 - pi1**2)
piSkew = (pi3 - 3.0*pi2*pi1 + 2.0*pi1**3) / piRms**3
piFlat = (pi4 - 4.0*pi3*pi1 + 6.0*pi2*pi1**2 - 3.0*pi1**4) / piRms**4

print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')

# write 1d one-point statistics to ascii file
fnam = 'piStatBox2d_pipe0002_'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
f = open(fnam, 'w')
f.write("# One-dimensional one-point statistics (mean, RMS, skewness and flatness factors)\n")
f.write("# For the inter-scale energy flux (Pi) across scale lambda\n")
f.write("# Flux based on filtered quantities using a 2d box kernel with:\n")
f.write("# + Azimuthal filter scale: lambdaTh+ = %f viscous units, lambdaTh = %f R\n" % (lambdaThp, lambdaTh))
f.write("# + Axial filter scale:     lambdaZ+  = %f viscous units, lambdaZ  = %f R\n" % (lambdaZp,  lambdaZ))
f.write("# Python post-processing on data set nsPipe/pipe0002 generated in a DNS using nsPipe\n")
f.write("# Temporal (ensemble) averaging over %d sample(s)\n" % (nt))
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# Additional spatial averaging in azimuthal (theta) direction over %d points\n" % (nth))
f.write("# Additional spatial averaging in axial (z) direction over %d points\n" % (nz))
f.write("# R: pipe radius\n")
f.write("# U: Hagen-Poiseuille center line velocity\n")
f.write("# 01st column: Radial co-ordinate r in units of R, nr = %d points\n" % nr)
f.write("# 02nd column: Mean energy flux <Pi> in units of U^3 R^2,\n")
f.write("# 03rd column: RMS of Pi in units of U^3 R^2\n")
f.write("# 04th column: Skewness of Pi in units RMS^3\n")
f.write("# 05th column: Flatness of Pi in units RMS^4\n")
for i in range(nr):
 f.write("%23.16e %23.16e %23.16e %23.16e %23.16e\n" % (r[i], piMean[i], piRms[i], piSkew[i], piFlat[i]))
f.close()
print('Written energy flux one-point statistics to file', fnam)

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
#fig = plt.figure(num=None, figsize=mm2inch(134.0, 70.0), dpi=300) # , constrained_layout=False)
fig = plt.figure(num=None, dpi=100) # , constrained_layout=False)

# conservative colour palette appropriate for colour-blind (http://mkweb.bcgsc.ca/colorblind/)
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Grey          = '#999999'
Black         = '#000000'

# plot mean flux profile
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax1.set_xlabel(r"$r$ in $R$")
ax1.set_xlim(left=0.0, right=1.0)
ax1.set_ylabel(r"$\langle\Pi\rangle_{\theta, z, t}$ in $U^3_{\text{HP}}R^2$")
ax1.axhline(y=0.0, color=Grey)
ax1.plot(r, piMean, color=Vermillion, linestyle='-',  label=r"Mean")
ax1.legend(loc='best', ncol=1, frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot rms flux profile
ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax2.set_xlabel(r"$r$ in $R$")
ax2.set_xlim(left=0.0, right=1.0)
ax2.set_ylabel(r"$\sqrt{ \langle\Pi^{\prime 2} \rangle_{\theta, z, t} }$ in $U^3_{\text{HP}}R^2$")
ax2.plot(r, piRms, color=Vermillion, linestyle='-', label=r"RMS")
ax2.legend(loc='best', ncol=1, frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot skewness flux profile
ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax3.set_xlabel(r"$r$ in $R$")
ax3.set_xlim(left=0.0, right=1.0)
ax3.set_ylabel(r"$\langle\Pi^{\prime 3} \rangle_{\theta, z, t}$ in $\langle\Pi^{\prime 2} \rangle^{\sfrac{3}{2}}_{\theta, z, t} $  ")
ax3.axhline(y=0.0, color=Grey)
ax3.plot(r, piSkew, color=Vermillion, linestyle='-', label=r"Skewness")
ax3.legend(loc='best', ncol=1, frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot flatness flux profile
ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
ax4.set_xlabel(r"$r$ in $R$")
ax4.set_xlim(left=0.0, right=1.0)
ax4.set_ylabel(r"$\langle\Pi^{\prime 4} \rangle_{\theta, z, t}$ in $\langle\Pi^{\prime 2} \rangle^{\sfrac{4}{2}}_{\theta, z, t} $  ")
ax4.plot(r, piFlat, color=Vermillion, linestyle='-',  label=r"Flatness")
ax4.legend(loc='best', ncol=1, frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = str.replace(fnam, '.dat', '.pdf')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()

print('Done!')
