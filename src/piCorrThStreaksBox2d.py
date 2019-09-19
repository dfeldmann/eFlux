#!/usr/bin/env python3
# Purpose:  Read full 3d pipe flow velocity fields from one or more HDF5 files.
#           Read the computational grid in cylindrical co-ordinates from one of
#           these files. Read a statistically converged mean velocity profile
#           from a separate ascii file and subtract it from each snapshot to
#           obtain the fluctuating velocity field. Define the filter widths and
#           compute the inter-scale turbulent kinetic energy flux field (Pi) for
#           each individual snapshot based on a two-dimensional spatial box
#           filter operation in wall-parallel planes for each radial location.
#           Additionally, compute the vorticity vector field (omega) for each
#           snapshot. Finally, compute one-dimensional two-point correlations
#           in azimuthal (theta) direction; auto-correlations for the original
#           (u'_z) and filtered (u'_zF) streamwise velocity component
#           (representing streaks), for the axial vorticity component
#           (representing streamwise alligned vortices) and for the energy flux,
#           cross-correlations for all of these quantities with the energy flux.
#           Do statistics over all axial locations and all snapshots, and write
#           the resulting 1d correlations to a single ascii file.
# Usage:    python piCorrThStreaksBox2d.py
# Authors:  Daniel Feldmann, Mohammad Umair, Jan Chen
# Date:     28th March 2019
# Modified: 19th September 2019

import timeit
import math
import numpy as np
import h5py

# range of state files to read flow field data
iFirst =  1675000
iLast  =  1675000
iStep  =     5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('Compute eFlux (Box) and 1d correlations for', len(iFiles), 'snapshot(s):', iFiles[0], 'to', iFiles[-1])

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
acUz       = np.zeros(nth) # initialise auto-correlation for u'_z
acUzF      = np.zeros(nth) # initialise auto-correlation for u'_z box filtered
acOmegaZ   = np.zeros(nth) # initialise auto-correlation for omega_z
acPi       = np.zeros(nth) # initialise auto-correlation for Pi
ccUzPi     = np.zeros(nth) # initialise cross-correlation for u'_z and eFlux
ccUzFPi    = np.zeros(nth) # initialise cross-correlation for u'_z filtered and eFlux
ccOmegaZPi = np.zeros(nth) # initialise cross-correlation for omega_z nd eFlux
nt         = 0             # reset ensemble counter

# first and second statistical moments for normalisation
uz1     = 0
uz2     = 0
uzF1    = 0
uzF2    = 0
omegaZ1 = 0
omegaZ2 = 0
pi1     = 0
pi2     = 0

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
    tvort = timeit.default_timer()
    print('Computing vorticity vector field... ', end='', flush=True)
    import vorticity as v
    omegaR, omegaTh, omegaZ = v.omegaCyl(u_r, u_th, u_z, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tvort), 'seconds')
    
    # subtract mean velocity profile (1d) to obtain full (3d) fluctuating velocity field
    u_z = u_z - np.tile(u_zM, (len(z), len(th), 1)).T
    
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

    # fix wall-normal (radial) location
    k = 65
    print("Extracting 1d azimuthal data sets at wall-normal location y+ =", (1-r[k])*ReTau)
    
    tcorr = timeit.default_timer()
    print('Computing 1d correlations... ', end='', flush=True)

    # loop over all axial (z) locations
    for l in range(nz):

        # extract 1d data sub-sets along azimuthal line at constant wall distance
        uz1d     =    u_z[k, :, l] # data structure is (r, theta, z)
        uzF1d    =   u_zF[k, :, l]
        omegaZ1d = omegaZ[k, :, l]
        pi1d     =     pi[k, :, l]

        # compute correlations and sum up axial (spatial) and temporal (ensemble) statistics
        import crossCorrelation as c 
        acUz       = acUz       + c.corr1d(uz1d,     uz1d)     # auto-correlations
        acUzF      = acUzF      + c.corr1d(uzF1d,    uzF1d)
        acOmegaZ   = acOmegaZ   + c.corr1d(omegaZ1d, omegaZ1d)
        acPi       = acPi       + c.corr1d(pi1d,     pi1d)
        ccUzPi     = ccUzPi     + c.corr1d(uz1d,     pi1d)     # cross-correlations
        ccUzFPi    = ccUzFPi    + c.corr1d(uzF1d,    pi1d)
        ccOmegaZPi = ccOmegaZPi + c.corr1d(omegaZ1d, pi1d)

        # sum up first and second statistical moments in time and (homogeneous) theta and z direction for normalisation
        uz1     = uz1     + np.sum(uz1d)
        uz2     = uz2     + np.sum(uz1d**2)
        uzF1    = uzF1    + np.sum(uzF1d)
        uzF2    = uzF2    + np.sum(uzF1d**2)
        omegaZ1 = omegaZ1 + np.sum(omegaZ1d)
        omegaZ2 = omegaZ2 + np.sum(omegaZ1d**2)
        pi1     = pi1     + np.sum(pi1d)
        pi2     = pi2     + np.sum(pi1d**2)

    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tcorr), 'seconds')

    # increase temporal/ensemble counter
    nt = nt + 1

# divide correlation statistics by total number of spatio-temporal samples
acUz       = acUz       / (nt*nz)
acUzF      = acUzF      / (nt*nz)
acOmegaZ   = acOmegaZ   / (nt*nz)
acPi       = acPi       / (nt*nz)
ccUzPi     = ccUzPi     / (nt*nz)
ccUzFPi    = ccUzFPi    / (nt*nz)
ccOmegaZPi = ccOmegaZPi / (nt*nz)

# divide normalisation statistics by total number of spatio-temporal samples
uz1     = uz1     / (nth*nz*nt)
uz2     = uz2     / (nth*nz*nt)
uzF1    = uzF1    / (nth*nz*nt)
uzF2    = uzF2    / (nth*nz*nt)
omegaZ1 = omegaZ1 / (nth*nz*nt)
omegaZ2 = omegaZ2 / (nth*nz*nt)
pi1     = pi1     / (nth*nz*nt)
pi2     = pi2     / (nth*nz*nt)

# compute RMS for normalisation
uzRms      = np.sqrt(uz2 - uz1**2)
uzFRms     = np.sqrt(uzF2 - uzF1**2)
omegaZRms  = np.sqrt(omegaZ2 - omegaZ1**2)
piRms      = np.sqrt(pi2 - pi1**2)
#print('uzMean', uz1)
#print('uzRms', uzRms)

# normalise correlations with local RMS 
acUz       = acUz       / (uzRms     * uzRms)
acUzF      = acUzF      / (uzFRms    * uzFRms)
acOmegaZ   = acOmegaZ   / (omegaZRms * omegaZRms)
acPi       = acPi       / (piRms     * piRms)
ccUzPi     = ccUzPi     / (uzRms     * piRms)
ccUzFPi    = ccUzFPi    / (uzFRms    * piRms)
ccOmegaZPi = ccOmegaZPi / (omegaZRms * piRms)

print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')

# compute centered azimuthal separation/displacement (for nice plotting only)
DeltaTh = (th - (th[-1] - th[0]) / 2.0) * r[k]
# DeltaZ  =   z - ( z[-1] -  z[0]) / 2.0

# write 1d correlations to ascii file
fnam = 'piCorrThStreaksBox2d_pipe0002_'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
f = open(fnam, 'w')
f.write("# One-dimensional two-point correlations in azimuthal (theta) direction\n")
f.write("# At wall-normal location r = %f -> y+ = %f\n" % (r[k], (1.0-r[k])*ReTau))
f.write("# For the following flow variables:\n")
f.write("# + Streamwise velocity u'_z (High-speed and low-speed streaks)\n")
f.write("# + Filtered streamwise velocity u'_zF (Smoothed streaks)\n")
f.write("# + Axial vorticity component omega_z (Streamwise aligned vortices)\n")
f.write("# + Inter-scale energy flux Pi across scale lambda\n")
f.write("# Flux and filtered quantities based on 2d box kernel with:\n")
f.write("# + Azimuthal filter scale: lambdaTh+ = %f viscous units, lambdaTh = %f R\n" % (lambdaThp, lambdaTh))
f.write("# + Axial filter scale:     lambdaZ+  = %f viscous units, lambdaZ  = %f R\n" % (lambdaZp,  lambdaZ))
f.write("# Python post-processing on data set nsPipe/pipe0002 generated in a DNS using nsPipe\n")
f.write("# Temporal (ensemble) averaging over %d sample(s)\n" % (nt))
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# Additional spatial averaging in axial (z) direction over %d points\n" % (nz))
f.write("# 01st column: Azimuthal separation DeltaTh in units of pipe radii (R), nth = %d points\n" % nth)
f.write("# 02nd column: Auto-correlation  u'_z    with u'_z\n")
f.write("# 03rd column: Auto-correlation  u'_zF   with u'_zF\n")
f.write("# 04th column: Auto-correlation  omega_z with omega_z\n")
f.write("# 05th column: Auto-correlation  Pi      with Pi\n")
f.write("# 06th column: Cross-correlation u'_z    with Pi\n")
f.write("# 07th column: Cross-correlation u'_zF   with Pi\n")
f.write("# 08th column: Cross-correlation omega_z with Pi\n")
for i in range(nth):
  f.write("%23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n" % (DeltaTh[i], acUz[i], acUzF[i], acOmegaZ[i], acPi[i], ccUzPi[i], ccUzFPi[i], ccOmegaZPi[i]))
f.close()
print('Written 1d correlations to file', fnam)





print('Done!')
