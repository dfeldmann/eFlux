#!/usr/bin/env python3
#====================================================================================
# Purpose:  compute interscale energy flux from HDF5 data sets, subtract mean profile, 
#           based on 2d Gauss space cut-off filter, subtract mean velocity profile,
#           filter, compute flux, compute statistics (mean, rms, skewness, flatness)
#           plot and print to file
# Usage:    python piStatGauss2d.py 
# Authors:  Daniel Feldmann, Jan Chen
# Date:     12th July 2018
# Modified: 08th March 2019
# ===================================================================================
import sys
import os.path
import timeit
import math
import numpy as np
import h5py
#--------------------------------------------------
# range of state files to read from flow field data
iFirst =  570000
iLast  =  1540000
iStep  =  5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('=============================================================================================')
print('Computing energy flux statistics from', len(iFiles), 'snapshots:', iFiles[0], 'to', iFiles[-1])
print('=============================================================================================')

#--------------------------------
# read grid from first hdf5 file
fnam = '../outFiles/fields_pipe0002_'+'{:08d}'.format(iFirst)+'.h5'
print('Reading grid from', fnam, 'with:')
f  = h5py.File(fnam, 'r') # open hdf5 file for read only
r  = np.array(f['grid/r'])
z  = np.array(f['grid/z'])
th = np.array(f['grid/th'])
f.close() # close hdf5 file

#------------------------------
# grid size
nr  = len(r)
nth = len(th)
nz  = len(z)
print(nr, 'radial (r) points')
print(nth, 'azimuthal (th) points')
print(nz, 'axial (z) points')

#-------------------------------------------
# read mean velocity profiles from ascii file
fnam = '../statistics00570000to01540000nt0195.dat'
print('-----------------------------------------------------------------')
print('Reading mean velocity profiles from', fnam)
u_zM = np.loadtxt(fnam)[:, 3]

#--------------------------------------------------
print('=============================================================================================')
# define filter width for each direction seperately
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

#------------------------------------------------
# parallel stuff
import multiprocessing
from joblib import Parallel, delayed
print('=============================================================================================')
print("Running on", multiprocessing.cpu_count(), "cores")
print('=============================================================================================')

#===================================================================================
# reset ensemble counter and statistical moments
nt = 0
pi1 = np.zeros(nr)
pi2 = np.zeros(nr)
pi3 = np.zeros(nr)
pi4 = np.zeros(nr)

# reset wall-clock time
t0 = timeit.default_timer()

# statistics loop over all state files
for iFile in iFiles:

    # read flow field data from next hdf5 file
    fnam = '../outFiles/fields_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    f = h5py.File(fnam, 'r')
    print("Reading velocity field from file", fnam, end='', flush=True)
    u_r  = np.array(f['fields/velocity/u_r']).transpose(0,2,1)  # do moidify transpose command to match data structures
    u_th = np.array(f['fields/velocity/u_th']).transpose(0,2,1) # openpipeflow u[r,th,z] and nsCouette/nsPipe u[r,th,z]
    u_z  = np.array(f['fields/velocity/u_z']).transpose(0,2,1)  # filter functions were made for u[r,th,z]
    f.close()
    print(' with data structure u', u_z.shape)

    # subtract mean velocity profiles (1d) from flow field (3d)
    u_z  = u_z - np.tile(u_zM, (len(z), len(th), 1)).T
   
    # filter velocity field, single components and mixed terms
    print('-------------------------------------------------------------------------')
    print('Filtering velocities... ', end='', flush=True)
    t1 = timeit.default_timer()
    import filters as f
    u_rF    = f.gauss2d(u_r,       lambdaTh, lambdaZ, r, th, z)
    u_thF   = f.gauss2d(u_th,      lambdaTh, lambdaZ, r, th, z)
    u_zF    = f.gauss2d(u_z,       lambdaTh, lambdaZ, r, th, z)
    u_rRF   = f.gauss2d(u_r*u_r,   lambdaTh, lambdaZ, r, th, z)
    u_rThF  = f.gauss2d(u_r*u_th,  lambdaTh, lambdaZ, r, th, z)
    u_rZF   = f.gauss2d(u_r*u_z,   lambdaTh, lambdaZ, r, th, z)
    u_thThF = f.gauss2d(u_th*u_th, lambdaTh, lambdaZ, r, th, z)
    u_thZF  = f.gauss2d(u_th*u_z,  lambdaTh, lambdaZ, r, th, z)
    u_zZF   = f.gauss2d(u_z*u_z,   lambdaTh, lambdaZ, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t1), 'seconds')
    

    # compute instantaneous energy flux
    t2 = timeit.default_timer()
    print('-------------------------------------------------------------------------')
    print('Computing energy flux... ', end='', flush=True)
    import eflux as ef
    pi = ef.eflux(u_rF, u_thF, u_zF, u_rRF, u_rThF, u_rZF, u_thThF, u_thZF, u_zZF, r, th, z)

    # sum up spatio-temporal statistics in time and (homogeneous) theta and z direction
    pi1 = pi1 + np.sum(np.sum(pi,    axis=1), axis=1)
    pi2 = pi2 + np.sum(np.sum(pi**2, axis=1), axis=1)
    pi3 = pi3 + np.sum(np.sum(pi**3, axis=1), axis=1)
    pi4 = pi4 + np.sum(np.sum(pi**4, axis=1), axis=1)
    nt = nt + 1 # increase temporal/ensemble counter
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t2), 'seconds')

# divide by total number of spatio-temporal samples
pi1 = pi1/(nth*nz*nt)
pi2 = pi2/(nth*nz*nt)
pi3 = pi3/(nth*nz*nt)
pi4 = pi4/(nth*nz*nt)

# compute mean, rms, skewness, flatness
piMean = pi1
piRms  = np.sqrt(pi2 - pi1**2)
piSkew = (pi3 - 3.0*pi2*pi1 + 2.0*pi1**3) / piRms**3
piFlat = (pi4 - 4.0*pi3*pi1 + 6.0*pi2*pi1**2 - 3.0*pi1**4) / piRms**4
print('-------------------------------------------------------------------------')
print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')
#=============================================================================================

# write energy flux statistics to ascii file
fnam = 'piStatGauss2d'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
f = open(fnam, 'w')
f.write('# First to fourth order one-point statistics (mean, rms, skewness, flatness) for the\n')
f.write('# Inter-scale energy flux Pi^l based on a 2d Gauss space cut-off filter)\n')
f.write('# Python post-processing on data set nsPipe/pipe0003 generated in a DNS using nsPipe\n')
f.write('# Temporal (ensemble) averaging over %d samples\n' % (nt))
f.write('# Filter width in r:  lambdaR+  =%6.1f viscous units, lambdaR  =%7.5f in R.}' % (lambdaRp, lambdaR))
f.write('# Filter width in th: lambdaTh+ =%6.1f viscous units, lambdaTh =%7.5f in R.}' % (lambdaThp,lambdaTh))
f.write('# Filter width in z:  lambdaZ+  =%6.1f viscous units, lambdaZ  =%7.5f in R.}' % (lambdaZp, lambdaZ))
#f.write('# First snapshot: %09d at t = %6.1f bulk time units (R/U)\n' % (iFirst, tFirst))
#f.write('# Last snapshot:  %09d at t = %6.1f bulk time units (R/U)\n' % (iLast, tLast))
# time information should be in the h5 file but it is not,  TODO: change in nsPipe/nsCouette
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# Additional averaging in theta and z directions over %d x %d points\n" % (nth, nz))
f.write("# 01st column: radial co-ordinate r in units of pipe radii (R)\n")
f.write("# 02nd column: mean energy flux <Pi> in units of \n")
f.write("# 03rd column: rms of <Pi> in units of \n")
f.write("# 04th column: skewness of <Pi> in units rms^3\n")
f.write("# 05th column: flatness of <Pi> in units rms^4\n")
for i in range(len(r)):
 f.write("%23.16e %23.16e %23.16e %23.16e %23.16e\n" % (r[i], piMean[i], piRms[i], piSkew[i], piFlat[i] ))
f.close()
print('-------------------------------------------------------------------------')
print('Written energy flux one-point statistics to file', fnam)

#=================================================================================
# plot data as graph, (0) none, (1) interactive, (2) pdf
plot = 0
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

# plot mean flux profile
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
#ax1.set_title(r"Mean energy flux, $Re=5300$, \texttt{pipe0003}")
ax1.set_xlabel(r"$r$ in $R$")
ax1.set_ylabel(r"$\langle\Pi^{\lambda}\rangle_{\theta, z, t}$ in $ $")
ax1.axhline(y=0, xmin=0, xmax=1, color=Black, linewidth=0.5)
ax1.plot(r, piMean, color=Vermillion, linestyle='-',  label=r"$\lambda_{\theta}\times\lambda_{z}=({}\times{})$") # .format(lambdaTh, lambdaZ))
ax1.legend(loc='best', ncol=4)

# plot rms flux profile
ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax2.set_xlabel(r"$r$ in $R$")
ax2.set_ylabel(r"$\text{RMS}\left(\Pi^{\lambda}\right)$ in $ $")
ax2.plot(r, piRms, color=Vermillion, linestyle='-',  label=r"$\lambda_{\theta}\times\lambda_{z}=(\times)$")
ax2.legend(loc='best', ncol=4)

# plot skewness flux profile
ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax3.set_xlabel(r"$r$ in $R$")
ax3.set_ylabel(r"$\text{Skewness}\left(\Pi^{\lambda}\right)$ in $\text{RMS}^{3}$")
ax3.plot(r, piSkew, color=Vermillion, linestyle='-',  label=r"$\lambda_{\theta}\times\lambda_{z}=(\times)$")
ax3.legend(loc='best', ncol=4)

# plot flatness flux profile
ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
ax4.set_xlabel(r"$r$ in $R$")
ax4.set_ylabel(r"$\text{Flatness}\left(\Pi^{\lambda}\right)$ in $\text{RMS}^{4}$")
ax4.plot(r, piFlat, color=Vermillion, linestyle='-',  label=r"$\lambda_{\theta}\times\lambda_{z}=(\times)$")
ax4.legend(loc='best', ncol=4)

# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = 'piStatGauss2d'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.pdf'
 plt.savefig(fnam)
 print('Written file', fnam)
print('=============================================================================================')
fig.clf()
