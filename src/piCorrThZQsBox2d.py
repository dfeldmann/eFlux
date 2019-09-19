#!/usr/bin/env python3
# Purpose:  Read full 3d pipe flow velocity fields from one or more HDF5 files.
#           Read the computational grid in cylindrical co-ordinates from one of
#           these files. Read a statistically converged mean velocity profile
#           from a separate ascii file and subtract it from each snapshot to
#           obtain the fluctuating velocity field. Define the filter widths and
#           compute the inter-scale turbulent kinetic energy flux field (Pi) for
#           each individual snapshot based on a two-dimensional spatial box
#           filter operation in wall-parallel planes for each radial location.
#           Additionally, extract Q events from the velocity field for each
#           snapshot. Finally, compute two-dimensional two-point correlation
#           maps in a selected wall-parallel (theta-z) plane; auto-correlations
#           for the Q events (representing important features of the near-wall
#           cycle) and for the energy flux, cross-correlations for all of the Q
#           events with the energy flux. Do statistics over all snapshots, and
#           write the resulting 2d correlation maps to a single ascii file.
#           TODO: At some point one could compute correlation maps for ALL
#           wall-parallel planes and save the resulting 3d fields in one single
#           h5 file...
# Usage:    python piCorrThZQsBox2d.py
# Authors:  Daniel Feldmann, Mohammad Umair, Jan Chen
# Date:     28th March 2019
# Modified: 19th September 2019

#import sys
#import os.path
import timeit
import math
import numpy as np
import h5py

# range of state files to read flow field data (do modify)
iFirst =  1675000 # 570000
iLast  =  1675000
iStep  =     5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('Compute eFlux (Box) and 2d correlation maps for', len(iFiles), 'snapshot(s):', iFiles[0], 'to', iFiles[-1])

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

# define filter width for each direction seperately (do modify)
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
acQ1   = np.zeros((nth, nz))  # initialise auto-correlation for Q1 outward interactions
acQ2   = np.zeros((nth, nz))  # initialise auto-correlation for Q2 ejection events
acQ3   = np.zeros((nth, nz))  # initialise auto-correlation for Q3 inward interactions
acQ4   = np.zeros((nth, nz))  # initialise auto-correlation for Q4 sweep events
acPi   = np.zeros((nth, nz))  # initialise auto-correlation for Pi
ccQ1Pi = np.zeros((nth, nz))  # initialise cross-correlation for Q1 eFlux
ccQ2Pi = np.zeros((nth, nz))  # initialise cross-correlation for Q2 eFlux
ccQ3Pi = np.zeros((nth, nz))  # initialise cross-correlation for Q3 eFlux
ccQ4Pi = np.zeros((nth, nz))  # initialise cross-correlation for Q4 eFlux
nt     = 0                    # reset ensemble counter

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
    
    # subtract mean velocity profile (1d) to obtain full (3d) fluctuating velocity field
    u_z = u_z - np.tile(u_zM, (len(z), len(th), 1)).T

    # detect and extrct Q events from the instantaneous volocity vector field
    # TODO: compute correlation maps for ALL wall-parallel planes
    #t1 = timeit.default_timer()
    #print('Extract Q events... ', end='', flush=True)
    #q1 = np.zeros((nr, nth, nz))  
    #q2 = np.zeros((nr, nth, nz))  
    #q3 = np.zeros((nr, nth, nz))  
    #q4 = np.zeros((nr, nth, nz))  
    #print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t1), 'seconds')
    
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

    # extract 2d data sub-sets in a wall parallel plane
    k = 65 # (do modify)
    print("Extracting 2d data in wall-parallel plane at y+ =", (1-r[k])*ReTau)
    pi2d =  pi[k, :, :]
    ur   = u_r[k, :, :]
    uz   = u_z[k, :, :]
    #q12d = q1[k, :, :] # TODO: compute correlation maps for ALL wall-parallel planes
    #q22d = q2[k, :, :]
    #q32d = q3[k, :, :]
    #q42d = q4[k, :, :]

    # detect and extract Q events from the 2d volocity sub-set
    tqs = timeit.default_timer()
    print("Extracting Q events from 2d volocity field...", end='', flush=True)
    q1 = np.zeros((ur.shape))
    q2 = np.zeros((ur.shape))
    q3 = np.zeros((ur.shape))
    q4 = np.zeros((ur.shape))
    for i in range(nz):
     for j in range(nth):
      if (uz[j,i]>0) and (ur[j,i]<0): q1[j,i] = ur[j,i]*uz[j,i] # outward interaction: high-speed fluid away from wall
      if (uz[j,i]<0) and (ur[j,i]<0): q2[j,i] = ur[j,i]*uz[j,i] # ejection event:       low-speed fluid away from wall
      if (uz[j,i]<0) and (ur[j,i]>0): q3[j,i] = ur[j,i]*uz[j,i] # inward interaction:   low-speed fluid towards   wall
      if (uz[j,i]>0) and (ur[j,i]>0): q4[j,i] = ur[j,i]*uz[j,i] # sweep event:         high-speed fluid towards   wall
    #ioi = q1 - q3 # unify inward interactions (Q3 being negativ) and outward interactions (Q1 being positive) in one array
    #see = q2 - q4 # unify sweep events (Q4 being negativ) and ejection events (Q2 being positiv) in one array
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tqs), 'seconds')

    # compute correlations and sum up temporal (ensemble) statistics
    t4 = timeit.default_timer()
    print('Computing 2d correlations... ', end='', flush=True)
    import crossCorrelation as c 
    acQ1   = acQ1   + c.corr2d(q1,   q1)   # auto-correlations
    acQ2   = acQ2   + c.corr2d(q2,   q2)
    acQ3   = acQ3   + c.corr2d(q3,   q3)
    acQ4   = acQ4   + c.corr2d(q4,   q4)
    acPi   = acPi   + c.corr2d(pi2d, pi2d)
    ccQ1Pi = ccQ1Pi + c.corr2d(q1,   pi2d) # cross-correlations
    ccQ2Pi = ccQ2Pi + c.corr2d(q2,   pi2d)
    ccQ3Pi = ccQ3Pi + c.corr2d(q3,   pi2d)
    ccQ4Pi = ccQ4Pi + c.corr2d(q4,   pi2d)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t4), 'seconds')

    # increase temporal/ensemble counter
    nt = nt + 1

# divide by total number of temporal samples
acQ1   = acQ1   / nt
acQ2   = acQ2   / nt
acQ3   = acQ3   / nt
acQ14  = acQ4   / nt
ccQ1Pi = ccQ1Pi / nt
ccQ2Pi = ccQ2Pi / nt
ccQ3Pi = ccQ3Pi / nt
ccQ4Pi = ccQ4Pi / nt
print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')

# compute centered azimuthal and axial separation/displacement (for nice plotting only)
DeltaTh = (th - (th[-1] - th[0]) / 2.0) * r[k]
DeltaZ  =   z - ( z[-1] -  z[0]) / 2.0

# write 2d correlation maps to ascii file
fnam = 'piCorrThZQsBox2d_pipe0002_'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
f = open(fnam, 'w')
f.write("# Two-dimensional two-point correlation maps in a theta-z plane\n")
f.write("# At wall-normal location r = %f -> y+ = %f\n" % (r[k], (1.0-r[k])*ReTau))
f.write("# For the following flow variables:\n")
f.write("# + Q1 outward interactions (u'_z > 0 and u'_r < 0)\n")
f.write("# + Q2 ejection events      (u'_z < 0 and u'_r < 0)\n")
f.write("# + Q3 inward interactions  (u'_z < 0 and u'_r > 0)\n")
f.write("# + Q4 sweep events         (u'_z > 0 and u'_r > 0)\n")
f.write("# + Inter-scale energy flux Pi across scale lambda\n")
f.write("# Note that u'_r < 0 represents motion away from the wall in a cylindrical co-ordinate system)\n")
f.write("# Flux based on filtered quantities using a 2d box kernel with:\n")
f.write("# + Azimuthal filter scale: lambdaTh+ = %f viscous units, lambdaTh = %f R\n" % (lambdaThp, lambdaTh))
f.write("# + Axial filter scale:     lambdaZ+  = %f viscous units, lambdaZ  = %f R\n" % (lambdaZp,  lambdaZ))
f.write("# Python post-processing on data set nsPipe/pipe0002 generated in a DNS using nsPipe\n")
f.write("# Temporal (ensemble) averaging over %d sample(s)\n" % (nt))
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# 01st column: Azimuthal separation DeltaTh in units of pipe radii (R), nth = %d points\n" % nth)
f.write("# 02nd column: Axial separation DeltaZ in units of pipe radii (R), nz = %d points\n" % nz)
f.write("# 03rd column: Auto-correlation  Q1 with Q1\n")
f.write("# 04th column: Auto-correlation  Q2 with Q2\n")
f.write("# 05th column: Auto-correlation  Q3 with Q3\n")
f.write("# 06th column: Auto-correlation  Q4 with Q4\n")
f.write("# 07th column: Auto-correlation  Pi with Pi\n")
f.write("# 08th column: Cross-correlation Q1 with Pi\n")
f.write("# 09th column: Cross-correlation Q2 with Pi\n")
f.write("# 10th column: Cross-correlation Q3 with Pi\n")
f.write("# 11th column: Cross-correlation Q4 with Pi\n")
for i in range(nth):
 for j in range(nz):
  f.write("%23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n" % (DeltaTh[i], DeltaZ[j], acQ1[i,j], acQ2[i,j], acQ3[i,j], acQ4[i,j], acPi[i,j], ccQ1Pi[i,j], ccQ2Pi[i,j], ccQ3Pi[i,j], ccQ4Pi[i,j]))
f.close()
print('Written 2d correlation maps to file', fnam)

print('Done!')
