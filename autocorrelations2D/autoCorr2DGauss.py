#!/usr/bin/env python3
#====================================================================================
# Purpose:  Computes the auto-correlation coefficient between the axial fluctuating 
#           velocity (both filtered and unfiltered), eflux and axial vorticity  with 
#           the Gauss filtered interscale energy flux in axial direction. 
#           Reads HDF5 files from given number of snapshots.
#           Computes the fluctuating field by subtracting the average from the statistics
#           file. Computes the filtered interscale energy flux using 2D Gauss filter.
#           Plots and prints the output in ascii format.
# ----------------------------------------------------------------------------------
# IMPORTANT:Make sure the statistics file should correspond to the given number
#           of snapshots 
# ----------------------------------------------------------------------------------
# Usage:    python autoCorr2DGauss.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 03rd May 2019
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
fnam = '../statistics00570000to01265000nt0140.dat'
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
acUz   = np.zeros((nth,nz)) # initialized for the correlation coefficient
acUzF  = np.zeros((nth,nz)) # initialized for the correlation coefficient
acPi   = np.zeros((nth,nz)) # initialized for the correlation coefficient
acOmgZ = np.zeros((nth,nz))
# reset wall-clock time
t0 = timeit.default_timer()

# statistics loop over all state files
for iFile in iFiles:
    #-------------------------------------------------------------------------------
    # read flow field data from next hdf5 file
    fnam = '../outFiles/fields_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    f = h5py.File(fnam, 'r')
    print("Reading velocity field from file", fnam, end='', flush=True)
    u_r  = np.array(f['fields/velocity/u_r']).transpose(0,2,1)  # do moidify transpose command to match data structures
    u_th = np.array(f['fields/velocity/u_th']).transpose(0,2,1) # openpipeflow u[r,th,z] and nsCouette/nsPipe u[r,th,z]
    u_z  = np.array(f['fields/velocity/u_z']).transpose(0,2,1)  # filter functions were made for u[r,th,z]
    p    = np.array(f['fields/pressure']).transpose(0,2,1)      # filter functions were made for u[r,th,z]
    f.close()
    print(' with data structure u', u_z.shape)
    #-------------------------------------------------------------------------------
    import vorticity as vort
    omgR, omgTh, omgZ = vort.omg(u_r, u_th, u_z, r, th, z)
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

    # compute instantaneous energy flux
    t2 = timeit.default_timer()
    print('-------------------------------------------------------------------------')
    print('Computing energy flux... ', end='', flush=True)
    import eflux as p
    pi = p.eflux(u_rF, u_thF, u_zF, u_rRF, u_rThF, u_rZF, u_thThF, u_thZF, u_zZF, r, th, z)
    #-------------------------------------------------------------------------------
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t2), 'seconds')
    #-------------------------------------------------------------------------------
    # compute auto-correlation
    print('-------------------------------------------------------------------------')
    print('Computing auto-correlation... ', end='', flush=True)
    t3 = timeit.default_timer()
    k = 61
    print ("Wall-normal plane at y+ =", (1-r[k])*180.4)
    # extract data and compute auto-correlations    
    uz     = u_z  [k, :, :]  
    eF     = pi   [k, :, :]
    uzF    = u_zF [k, :, :]
    omg    = omgZ [k, :, :]
    acUz   = acUz   + crosscorr(uz, uz)   # compute auto-correaltion and sum up average
    acUzF  = acUzF  + crosscorr(uzF, uzF) # compute auto-correaltion and sum up average
    acPi   = acPi   + crosscorr(eF, eF)   # compute auto-correaltion and sum up average
    acOmgZ = acOmgZ + crosscorr(omg, omg)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t3), 'seconds')
    print('-------------------------------------------------------------------------')

    nt = nt + 1 # increase temporal/ensemble counter

#-----------------------------------------------------------------------------------
# divide by total number of spatio-temporal samples compute mean & normalizing by RMS
acUz   = acUz/(nth*nt)
acUzF  = acUzF/(nth*nt)
acPi   = acPi/(nth*nt)
acOmgZ = acOmgZ/(nth*nt)
print('-------------------------------------------------------------------------')
print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')
#=============================================================================================

# write correlation coefficient in ascii file
dz = z - (z[-1]-z[0])/2.0
rdth= (th- (th[-1]-th[0])/2.0) * r[k]
fnam = 'autoCorr2dGauss'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
print('-------------------------------------------------------------------------')
print('Writing 2D auto-correlation to file', fnam)
f = open(fnam, 'w')
f.write('# Two-dimensional two-point auto-correlations in theta-z\n')
f.write('# For Uz, Uz(Gauss Filtered), eFlux and axial vorticity component.\n')
f.write("# At wall-normal location r = %f -> y+ = %f\n" % (r[k], (1-r[k])*180.4))
f.write('# Python post-processing on data set nsPipe/pipe0002 generated in a DNS using nsPipe\n')
f.write('# Temporal (ensemble) averaging over %d samples\n' % (nt))
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# Additional averaging in theta and axial direction")
f.write("# 01st column: arclength separation rdth in units of pipe radii (R)\n")
f.write("# 02nd column: axial separation dz in units of pipe radii (R)\n")
f.write("# 03rd column: Uz correlation\n")
f.write("# 04th column: UzF correlation\n")
f.write("# 05th column: eFlux correlation\n")
f.write("# 06th column: OmgZ correlation\n")
print('Written auto-correlation to file', fnam)
for i in range(nth-1):
 for j in range(nz-1):
  f.write("%23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n" % (rdth[i], dz[j], acUz[i,j], acUzF[i,j], acPi[i,j], acOmgZ[i,j]) )
f.close()
print('-------------------------------------------------------------------------')
print('Written 2D auto-correlation to the file', fnam)
print('-------------------------------------------------------------------------')
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

#-----------------------------------------------------------------------------------
# plot first axial velocity signal
#ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)
#ax1.set_xlabel(r"$z$ in $R$")
#ax1.set_ylabel(r"$u^{\prime}_z$ in $U_{c,\text{HP}}$")
#ax1.plot(z, uz, color=Vermillion, linestyle='-',  label=r"First signal")
#ax1.legend(loc='best', ncol=4)

# plot second axial velocity signal
#ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)
#ax2.set_xlabel(r"$z$ in $R$")
#ax2.set_ylabel(r"$\Pi^{\lambda}$ in ")
#ax2.plot(z, eF, color=Vermillion, linestyle='-',  label=r"Second signal")
#ax2.legend(loc='best', ncol=4)

# plot one-dimensional two-point auto-correlation
#ax3 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)
#ax3.set_xlabel(r"$\Delta z$ in $R$")
#ax3.set_ylabel(r"$C_{u_r \pi}(\Delta z)$")
#ax3.set_ylim([-0.1, 1.0])
#ax3.plot(dz, acU, color=Blue, linestyle='-',  label=r"Unfiltered") # .format(lambdaTh, lambdaZ))
#ax3.plot(dz, acrF, color=Vermillion, linestyle='-',  label=r"Filtered") # .format(lambdaTh, lambdaZ))
#ax3.legend(loc='best')

# plot one-dimensional two-point auto-correlation
#ax4 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)
#ax4.set_xlabel(r"$\Delta z$ in $R$")
#ax4.set_ylabel(r"$C_{u_{\theta} \pi}(\Delta z)$")
#ax3.set_ylim([-0.1, 1.0])
#ax4.plot(dz, acth, color=Blue, linestyle='-',  label=r"Correlation $r$") # .format(lambdaTh, lambdaZ))
#ax4.plot(dz, acthF, color=Vermillion, linestyle='-',  label=r"Correlation $\theta$") # .format(lambdaTh, lambdaZ))
#ax4.legend(loc='best')

# plot one-dimensional two-point auto-correlation
#ax5 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1)
#ax5.set_xlabel(r"$\Delta z$ in $R$")
#ax5.set_ylabel(r"$C_{u_z \pi}(\Delta z)$")
#ax3.set_ylim([-0.1, 1.0])
#ax5.plot(dz, acz, color=Blue, linestyle='-',  label=r"Correlation $z$") # .format(lambdaTh, lambdaZ))
#ax5.plot(dz, aczF, color=Vermillion, linestyle='-',  label=r"Correlation $z$") # .format(lambdaTh, lambdaZ))
#ax5.legend(loc='best')


# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = 'piCorr1dGaussZ'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.pdf'
 plt.savefig(fnam)
 print('Written file', fnam)
print('=============================================================================================')
fig.clf()





