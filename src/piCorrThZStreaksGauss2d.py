#!/usr/bin/env python3
# Purpose:  Read full 3d pipe flow velocity fields from one or more HDF5 files.
#           Read the computational grid in cylindrical co-ordinates from one of
#           these files. Read a statistically converged mean velocity profile
#           from a separate ascii file and subtract it from each snapshot to
#           obtain the fluctuating velocity field. Define the filter widths and
#           compute the inter-scale turbulent kinetic energy flux field (Pi) for
#           each individual snapshot based on a two-dimensional spatial Gauss
#           filter operation in wall-parallel planes for each radial location.
#           Additionally, compute the vorticity vector field (omega) for each
#           snapshot. Finally, compute two-dimensional two-point correlation
#           maps in a selected wall-parallel (theta-z) plane; auto-correlations
#           for the original (u'_z) and filtered (u'_zF) streamwise velocity
#           component (representing streaks), for the axial vorticity component
#           (representing streamwise alligned vortices) and for the energy flux,
#           cross-correlations for all of the quantities with the energy flux.
#           Do statistics over all snapshots, and  write the resulting 2d
#           correlation maps to a single ascii file. TODO: At some point one
#           could compute correlation maps for ALL wall-parallel planes and save
#           the resulting 3d fields in one single h5 file...
# Usage:    python piCorrThZStreaksGauss2d.py
# Authors:  Daniel Feldmann, Mohammad Umair, Jan Chen
# Date:     28th March 2019
# Modified: 16th September 2019

import sys
import os.path
import timeit
import math
import numpy as np
import h5py

# range of state files to read flow field data
iFirst =  1675000
iLast  =  1675000
iStep  =     5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('Compute eFlux (Gauss) and 2d correlation maps for', len(iFiles), 'snapshot(s):', iFiles[0], 'to', iFiles[-1])

# path to data files (do modify)
fpath = '../../outFiles/'

# read grid from first HDF5 file
fnam = fpath+'fields_pipe0002_'+'{:08d}'.format(iFirst)+'.h5'
print('Reading grid from', fnam, 'with:')
f  = h5py.File(fnam, 'r') # open hdf5 file for read only
r  = np.array(f['grid/r'])
z  = np.array(f['grid/z'])
th = np.array(f['grid/th'])
f.close() # close hdf5 file

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

# prepare arrays for statistics
acUz       = np.zeros((nth, nz))  # initialise auto-correlation for u'_z
acUzF      = np.zeros((nth, nz))  # initialise auto-correlation for u'_z Gauss filtered
acOmegaZ   = np.zeros((nth, nz))  # initialise auto-correlation for omega_z
acPi       = np.zeros((nth, nz))  # initialise auto-correlation for Pi
ccUzPi     = np.zeros((nth, nz))  # initialise cross-correlation for u'_z and eFlux
ccUzFPi    = np.zeros((nth, nz))  # initialise cross-correlation for u'_z filtered and eFlux
ccOmegaZPi = np.zeros((nth, nz))  # initialise cross-correlation for omega_z nd eFlux
nt         = 0                    # reset ensemble counter

# reset wall-clock time
t0 = timeit.default_timer()

# loop over all state files
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

    # compute instantaneous vorticity vector field
    t1 = timeit.default_timer()
    print('Computing vorticity vector field... ', end='', flush=True)
    import vorticity as v
    omegaR, omegaTh, omegaZ = v.omegaCyl(u_r, u_th, u_z, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t1), 'seconds')
    
    # subtract mean velocity profile (1d) to obtain full (3d) fluctuating velocity field
    u_z = u_z - np.tile(u_zM, (len(z), len(th), 1)).T
    
    # filter velocity field
    print('Filtering velocity components and mixed terms... ', end='', flush=True)
    t2 = timeit.default_timer()
    import filter2d as f2
    u_rF    = f2.gauss2d(u_r,       lambdaTh, lambdaZ, r, th, z)
    u_thF   = f2.gauss2d(u_th,      lambdaTh, lambdaZ, r, th, z)
    u_zF    = f2.gauss2d(u_z,       lambdaTh, lambdaZ, r, th, z)
    u_rRF   = f2.gauss2d(u_r*u_r,   lambdaTh, lambdaZ, r, th, z)
    u_rThF  = f2.gauss2d(u_r*u_th,  lambdaTh, lambdaZ, r, th, z)
    u_rZF   = f2.gauss2d(u_r*u_z,   lambdaTh, lambdaZ, r, th, z)
    u_thThF = f2.gauss2d(u_th*u_th, lambdaTh, lambdaZ, r, th, z)
    u_thZF  = f2.gauss2d(u_th*u_z,  lambdaTh, lambdaZ, r, th, z)
    u_zZF   = f2.gauss2d(u_z*u_z,   lambdaTh, lambdaZ, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t2), 'seconds')

    # compute instantaneous energy flux
    t3 = timeit.default_timer()
    print('Computing energy flux... ', end='', flush=True)
    import eFlux
    pi = eFlux.eFlux(u_rF, u_thF, u_zF, u_rRF, u_rThF, u_rZF, u_thThF, u_thZF, u_zZF, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t3), 'seconds')

    # extract 2d data sub-sets in a wall parallel plane
    k = 65
    print("Extracting 2d data in wall-parallel plane at y+ =", (1-r[k])*ReTau)
    uz2d     =    u_z[k, :, :]
    uzF2d    =   u_zF[k, :, :]
    omegaZ2d = omegaZ[k, :, :]
    pi2d     =     pi[k, :, :]

    # compute correlations and sum up temporal (ensemble) statistics
    t4 = timeit.default_timer()
    print('Computing 2d correlations... ', end='', flush=True)
    import crossCorrelation as c 
    acUz       = acUz       + c.corr2d(uz2d,     uz2d)     # auto-correlations
    acUzF      = acUzF      + c.corr2d(uzF2d,    uzF2d)
    acOmegaZ   = acOmegaZ   + c.corr2d(omegaZ2d, omegaZ2d)
    acPi       = acPi       + c.corr2d(pi2d,     pi2d)
    ccUzPi     = ccUzPi     + c.corr2d(uz2d,     pi2d)     # cross-correlations
    ccUzFPi    = ccUzFPi    + c.corr2d(uzF2d,    pi2d)
    ccOmegaZPi = ccOmegaZPi + c.corr2d(omegaZ2d, pi2d)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t4), 'seconds')

    # increase temporal/ensemble counter
    nt = nt + 1

# divide by total number of temporal samples
acUz       = acUz       / nt
acUzF      = acUzF      / nt
acOmegaZ   = acOmegaZ   / nt
acPi       = acPi       / nt
ccUzPi     = ccUzPi     / nt
ccUzFPi    = ccUzFPi    / nt
ccOmegaZPi = ccOmegaZPi / nt
print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')

# compute centered azimuthal and axial separation/displacement (for nice plotting only)
DeltaTh = (th - (th[-1] - th[0]) / 2.0) * r[k]
DeltaZ  =   z - ( z[-1] -  z[0]) / 2.0

# write 2d correlation map to ascii file
fnam = 'piCorrThZStreaksGauss2d_pipe0002_'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
f = open(fnam, 'w')
f.write("# Two-dimensional two-point correlation maps in a theta-z plane\n")
f.write("# At wall-normal location r = %f -> y+ = %f\n" % (r[k], (1.0-r[k])*ReTau))
f.write("# For the following flow variables:\n")
f.write("# + Streamwise velocity u'_z (High-speed and low-speed streaks)\n")
f.write("# + Filtered streamwise velocity u'_zF (Smoothed streaks)\n")
f.write("# + Axial vorticity component omega_z (Streamwise aligned vortices)\n")
f.write("# + Inter-scale energy flux Pi across scale lambda\n")
f.write("# Flux and filtered quantities based on 2d Gauss kernel with:\n")
f.write("# + Azimuthal filter scale: lambdaTh+ = %f viscous units, lambdaTh = %f R\n" % (lambdaThp, lambdaTh))
f.write("# + Axial filter scale:     lambdaZ+  = %f viscous units, lambdaZ  = %f R\n" % (lambdaZp,  lambdaZ))
f.write("# Python post-processing on data set nsPipe/pipe0002 generated in a DNS using nsPipe\n")
f.write("# Temporal (ensemble) averaging over %d sample(s)\n" % (nt))
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# 01st column: Azimuthal separation DeltaTh in units of pipe radii (R), nth = %d points\n" % nth)
f.write("# 02nd column: Axial separation DeltaZ in units of pipe radii (R), nz = %d points\n" % nz)
f.write("# 03rd column: Auto-correlation  u'_z    with u'_z\n")
f.write("# 04th column: Auto-correlation  u'_zF   with u'_zF\n")
f.write("# 05th column: Auto-correlation  omega_z with omega_z\n")
f.write("# 06th column: Auto-correlation  Pi      with Pi\n")
f.write("# 07th column: Cross-correlation u'_z    with Pi\n")
f.write("# 08th column: Cross-correlation u'_zF   with Pi\n")
f.write("# 09th column: Cross-correlation omega_z with Pi\n")
for i in range(nth):
 for j in range(nz):
  f.write("%23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n" % (DeltaTh[i], DeltaZ[j], acUz[i,j], acUzF[i,j], acOmegaZ[i,j], acPi[i,j], ccUzPi[i,j], ccUzFPi[i,j], ccOmegaZPi[i,j]))
f.close()
print('Written 2d correlation maps to file', fnam)

print('Done!')
