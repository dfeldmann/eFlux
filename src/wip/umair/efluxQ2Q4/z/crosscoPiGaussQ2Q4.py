#!/usr/bin/env python3
#====================================================================================
# Purpose:  Computes the cross-correlation coefficient between the Q2-Q4 events with
#           the Gauss filtered interscale energy flux in axial direction.
#           Reads HDF5 files from given number of snapshots.
#           Computes the fluctuating field by subtracting the average from the statistics
#           file. Computes the filtered interscale energy flux using 2D Spectral filter.
#           Plots and prints the output in ascii format.
# ----------------------------------------------------------------------------------
# IMPORTANT:Make sure the statistics file should correspond to the given number
#           of snapshots 
# ----------------------------------------------------------------------------------
# Usage:    python crosscoPiGaussQ2Q4.py 
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
#--------------------------------------------------
# range of state files to read from flow field data
iFirst =  570000
iLast  =  1265000
iStep  =  5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('=============================================================================================')
print('Computing energy flux statistics from', len(iFiles), 'snapshots:', iFiles[0], 'to', iFiles[-1])
print('=============================================================================================')

#--------------------------------
# read grid from first hdf5 file
fnam = '../../outFiles/fields_pipe0002_'+'{:08d}'.format(iFirst)+'.h5'
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
fnam = '../../statistics00570000to01265000nt0140.dat'
#fnam = '../statistics02900000to02900000nt0001.dat'
print('-----------------------------------------------------------------')
print('Reading mean velocity profiles from', fnam)
u_zM = np.loadtxt(fnam)[:, 3]
u_zR = np.loadtxt(fnam)[:, 7]
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
# Cross correlation of two signals of equal length
# Page:602 Numerical Recipies
# multiplying the Gauss transform of one function by the complex conjugate of the
# Gauss transform of the other gives the Gauss transform of their correlation.
#-----------------------------------------------------------------------------------
def crosscorr(a, b):         # IMPORTANT REMARKS: The normalization must be performed 
    import numpy as np     # with the L2-norm or (RMS) for the particular data series
                           # -------------------------------------------------------
    n1 = np.linalg.norm(a) # taking RMS of 'a' for a particular data series (*)
    n2 = np.linalg.norm(b) # taking RMS of 'b' for a particular data series (*)
    n12= n1*n2             # RMS(a)xRMS(b) needed for normalization
                           # -------------------------------------------------------
    a  = np.fft.fft(a)     # computing FFT of data series 'a'
    b  = np.fft.fft(b)     # computing FFT of data series 'b'
    c  = np.fft.ifft(a * np.conj(b)) # multiplying the FFT of 'a' by complex conjugate
                                     # of FFT of 'b' and then computing the inverse
                                     # FFT of the product to get the cross-correlation
    c  = np.fft.fftshift(c)/n12      # normalizing by the product of RMS values of
    return c.real                    # 'a' & 'b' and extracting only the real part
#===================================================================================
# reset ensemble counter and statistical moments
nt = 0
acE15Q2  = np.zeros(nz) # initialized for the correlation coefficient
acE15Q4  = np.zeros(nz) # initialized for the correlation coefficient

# reset wall-clock time
t0 = timeit.default_timer()

# statistics loop over all state files
for iFile in iFiles:
    #-------------------------------------------------------------------------------
    # read flow field data from next hdf5 file
    fnam = '../../outFiles/fields_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    f = h5py.File(fnam, 'r')
    print("Reading velocity field from file", fnam, end='', flush=True)
    u_r  = np.array(f['fields/velocity/u_r']).transpose(0,2,1)  # do moidify transpose command to match data structures
    u_th = np.array(f['fields/velocity/u_th']).transpose(0,2,1) # openpipeflow u[r,th,z] and nsCouette/nsPipe u[r,th,z]
    u_z  = np.array(f['fields/velocity/u_z']).transpose(0,2,1)  # filter functions were made for u[r,th,z]
    p    = np.array(f['fields/pressure']).transpose(0,2,1)      # filter functions were made for u[r,th,z]
    f.close()
    print(' with data structure u', u_z.shape)
    #-------------------------------------------------------------------------------
    # subtract mean velocity profiles (1d) from flow field (3d)
    u_z  = u_z - np.tile(u_zM, (len(z), len(th), 1)).T
    #-------------------------------------------------------------------------------
    # filter velocity field, single components and mixed terms
    print('-------------------------------------------------------------------------')
    print('Filtering velocities... ', end='', flush=True)
    import filters as f
    t1 = timeit.default_timer()
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
    #-------------------------------------------------------------------------------    
    # sweep (u'>0 & v'< 0) and ejection (u'<0 & v'>0) events
    uvE = np.zeros((u_z.shape))
    uvS = np.zeros((u_z.shape))
    for k in range(nz-1):
      for j in range(nth-1):
        for i in range(nr-1):
         if u_z[i,j,k]>0 and u_r[i,j,k]<0: #sweeps
          uvS[i,j,k] = u_z[i,j,k]*u_r[i,j,k]
         else:
          uvS[i,j,k] = 0.0
          #-------------------------------------------------------------------------------
         if u_z[i,j,k]<0 and u_r[i,j,k]>0: #ejections
          uvE[i,j,k] = u_z[i,j,k]*u_r[i,j,k]
         else:
          uvE[i,j,k] = 0.0
    #-------------------------------------------------------------------------------
    # compute instantaneous energy flux
    t2 = timeit.default_timer()
    print('-------------------------------------------------------------------------')
    print('Computing energy flux... ', end='', flush=True)
    import eflux as ef
    pi = ef.eflux(u_rF, u_thF, u_zF, u_rRF, u_rThF, u_rZF, u_thThF, u_thZF, u_zZF, r, th, z)
    #-------------------------------------------------------------------------------
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t2), 'seconds')
    #-------------------------------------------------------------------------------
    # compute cross-correlation
    print('-------------------------------------------------------------------------')
    print('Computing cross-correlation... ', end='', flush=True)
    t3 = timeit.default_timer()

#    np.seterr(divide='ignore', invalid='ignore')
    print('-------------------------------------------------------------------------')

    k = 61
    print ("Wall-normal plane at y+ =", (1-r[k])*180.4)
    # loop over all theta for averaging
    nth1 = 0
    nth2 = 0
    for l in range(nth):
        eF     =  pi[k, l, :]  # extract 2d filtered eflux 
        uvE15  = uvE[k, l, :]
        uvS15  = uvS[k, l, :]
        if np.linalg.norm(uvE15)==0:
         print('skipping zeros')
        else:
         acE15Q2 = acE15Q2  + crosscorr(uvE15, eF) # compute auto-correaltion and sum up average      
         nth1    = nth1     + 1
        if np.linalg.norm(uvS15)==0:
         print('skipping zeros')
        else:
         acE15Q4 = acE15Q4  + crosscorr(uvS15, eF) # compute auto-correaltion and sum up average      
         nth2    = nth2     + 1
    print('-------------------------------------------------------------------------')
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t3), 'seconds')
    print('-------------------------------------------------------------------------')

    nt = nt + 1 # increase temporal/ensemble counter

#-----------------------------------------------------------------------------------
# divide by total number of spatio-temporal samples compute mean & normalizing by RMS
acE15Q2  = acE15Q2 /(nth1*nt)
acE15Q4  = acE15Q4 /(nth2*nt)

print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')
#=============================================================================================

# write correlation coefficient in ascii file
dz = z - (z[-1]-z[0])/2.0 # Why?
fnam = 'crosscorreFluxGaussQ2Q4event1d'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
print('-------------------------------------------------------------------------')
print('Writing cross-correlation to file', fnam)
f = open(fnam, 'w')
f.write('# One-dimensional two-point auto-correlations in axial (z) direction\n')
f.write('# between eFlux and Q2 & Q4 events\n')
f.write("# At wall-normal location r = %f -> y+ = %f\n" % (r[k], (1-r[k])*180.4))
f.write('# Python post-processing on data set nsPipe/pipe0002 generated in a DNS using nsPipe\n')
f.write('# Temporal (ensemble) averaging over %d samples\n' % (nt))
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# Additional averaging in theta direction over %d points\n" % (nth))
f.write("# 01st column: axial separation dz in units of pipe radii (R)\n")
f.write("# 02nd column: eFlux-Q2 correlation\n")
f.write("# 03rd column: eFlux-Q4 correlation\n")

for i in range(nz-1):
 f.write("%23.16e %23.16e %23.16e\n" % (dz[i], acE15Q2[i], acE15Q4[i]))
f.close()
print('Written cross-correlation to file', fnam)

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
#-----------------------------------------------------------------------------------

line  = np.zeros(nz)

# plot one-dimensional two-point auto-correlation
ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
ax1.set_ylabel(r"$C_{Q_2 \pi}$")
ax1.set_xlim([-8, 8])
ax1.plot(dz, acE15Q2, color=Blue, linestyle='-') # .format(lambdaTh, lambdaZ))
ax1.plot(dz, line, color=Vermillion, linestyle='--', linewidth=0.5) # .format(lambdaTh, lambdaZ))
ax1.plot(line, acE15Q2, color=Vermillion, linestyle='--', linewidth=0.5) # .format(lambdaTh, lambdaZ))
ax1.title.set_text(r"$y^+ = 15$")

# plot one-dimensional two-point auto-correlation
ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
ax2.set_ylabel(r"$C_{Q_4 \pi}$")
ax2.set_xlim([-8, 8])
ax2.plot(dz, acE15Q4, color=Blue, linestyle='-') # .format(lambdaTh, lambdaZ))
ax2.plot(dz, line, color=Vermillion, linestyle='--', linewidth=0.5) # .format(lambdaTh, lambdaZ))
ax2.plot(line, acE15Q4, color=Vermillion, linestyle='--', linewidth=0.5) # .format(lambdaTh, lambdaZ))

# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = 'crosscorreFluxGaussQ2Q4event1d'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.pdf'
 plt.savefig(fnam)
 print('Written file', fnam)
print('=============================================================================================')
fig.clf()





