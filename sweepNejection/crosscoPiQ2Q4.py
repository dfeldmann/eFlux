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
#--------------------------------------------------
# range of state files to read from flow field data
iFirst =  875000
iLast  =  875000
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
fnam = '../statistics00570000to00875000nt0062.dat'
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
def gauss2dThZ(u, rPos, lambdaTh, lambdaZ, th, z):
# 2d Gauss filter for 2d data slices in
# wall-parallel (theta-z) planes of a cylindrical co-ordinate frame work
# (r,th,z). Applied in Gauss space via FFT.
# Parameters:
# u: 2d scalar field
# rPos: radial location of wall-parallel data slice
# lambdaTh: filter width in theta direction, arc length in unit distance
# lambdaZ: filter width in z direction
# th: 1d grid vector, theta co-ordinate in unit radian
# z: 1d grid vector, z co-ordinate in unit length
# Returns: A 2d scalar field which is 2d-filtered in theta and z direction

    # sample spacing in each direction in unit length (pipe radii R, gap width d, etc)
    deltaTh = (th[1] - th[0]) * rPos # equidistant but r dependent
    deltaZ  =  (z[1] -  z[0])        # equidistant

    # set-up wavenumber vector in units of cycles per unit distance (1/R)
    kTh = np.fft.fftfreq(len(th), d=deltaTh) # homogeneous direction theta #=kX=kappaX/(2*pi)
    kZ  = np.fft.fftfreq(len(z),  d=deltaZ)  # homogeneous direction z

    # construct 2d filter kernel
    gTh = np.exp(-((2*np.pi*kTh)**2*lambdaTh**2/24)) # Gauss exp kernel in theta direction
    gZ  = np.exp(-((2*np.pi*kZ )**2*lambdaZ**2/24))   # Gauss exp kernel in axial direction
    g2d = np.outer(gTh, gZ)                          # 2d filter kernel in theta-z plane

    # apply 2d filter kernel in Gauss space via FFT
    uFiltered=np.fft.ifft2(np.fft.fft2(u)*g2d)
    return uFiltered.real


#--------------------------------------------------------------------------------
def gauss2d(u, lambdaTh, lambdaZ, r, th, z, option):
# 2d Gauss filter for 3d data in a cylindrical
# co-ordinate frame work (r,th,z). Applied in Gauss space via FFT.
# Parameters:
# u: 3d scalar field
# lambdaTh: filter width in theta direction, arc length in unit distance
# lambdaZ: filter width in z direction
# r: 1d grid vector, radial (wall-normal) co-ordinate in unit length
# th: 1d grid vector, azimuthal co-ordinate in unit radian
# z: 1d grid vector, axial co-ordinate in unit length
# Returns: A 3d scalar field which is 2d-filtered in theta and z direction only

    # construct output array of correct shape filled with zeros
    uFiltered = np.zeros((len(r), len(th), len(z)))

    # filter each wall-normal (theta-z) plane separately, this can be done in parallel
    uFiltered=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gauss2dThZ)(u[i,:,:],r[i],lambdaTh, lambdaZ, th ,z) for i in range(len(r))))

    # uncomment for Härtel hack: constant angle instead of constant arc length
    #rRef = 0.88889 # radial reference location where given arc length is converted to used filter angle
    #uFiltered = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gauss2dThZ)(u[i,:,:], rRef, lambdaTh, lambdaZ, th, z) for i in range(len(r))))

    # simple serial version
    # for i in range(len(r)):
    #uFiltered[i] = gauss2dThZ(u[i,:,:], r[i], lambdaTh, lambdaZ, th, z)

    return uFiltered

#===================================================================================


#===================================================================================
# Define energy flux (pi = t_ij * S_ij)
def eflux(u1, u2, u3, u11, u12, u13, u22, u23, u33, r, th, z):
#------------------------------------------------------------
    # indices: indicates  co-ordinate direction: 1=r, 2=theta, 3=z
    
    pi  = np.zeros((u1.shape)) # constructing an array of dimension(r) filled with zeros 
    r3d = np.tile(r, (len(z), len(th), 1)).T # Changing a 1D array to a 3D array as we have to divide u(:,:,:) by r. 
                                             # In python we have to reshape our array to 3D to perform the division.

    # increment for spatial derivatives
    dth = th[1] - th[0]
    dz  =  z[1] -  z[0]

    #calculating strain tensor
    s11 = np.gradient(u1, r, axis=0)  #!r not equidistant, therefore positional information is needed
    s22 = 1.0/r3d*np.gradient(u2,dth,axis=1)+u1/r3d
    s33 = np.gradient(u3,dz,axis=2)
    s12 = 0.5*(1/r3d*np.gradient(u1,dth,axis=1)+np.gradient(u2,r,axis=0)-u2/r3d)
    s13 = 0.5*(np.gradient(u1,dz,axis=2)+np.gradient(u3,r,axis=0))
    s21 = s12
    s23 = 0.5*(1/r3d*np.gradient(u3,dth,axis=1)+np.gradient(u2,dz,axis=2))
    s32 = s23  
    s31 = s13  
            
    # calculate stress tensor tau for filter length scale l(see Xiao et al. 2009)
    #-----------------------------------------------------------------------------------
    # Note that the stress tensor tau has been calculated assuming the small filter width
    # So if someone wants to go for a larger filter width, the formulation will change.
    #-----------------------------------------------------------------------------------
    t11 = u11-u1*u1
    t12 = u12-u1*u2
    t13 = u13-u1*u3
    t21 = t12
    t22 = u22-u2*u2
    t23 = u23-u2*u3
    t31 = t13
    t32 = t23
    t33 = u33-u3*u3

    pi = (t11*s11+t12*s12+t13*s13+t21*s21+t22*s22+t23*s23+t31*s31+t32*s32+t33*s33)
    return -pi


#===================================================================================
# Cross correlation of two signals of equal length
# Page:602 Numerical Recipies
# multiplying the Fourier transform of one function by the complex conjugate of the
# Fourier transform of the other gives the Fourier transform of their correlation.
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
    c  = np.fft.fftshift(c)#/n12      # normalizing by the product of RMS values of
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
    # subtract mean velocity profiles (1d) from flow field (3d)
    u_z  = u_z - np.tile(u_zM, (len(z), len(th), 1)).T
    #-------------------------------------------------------------------------------
    # filter velocity field, single components and mixed terms
    print('-------------------------------------------------------------------------')
    print('Filtering velocities... ', end='', flush=True)
    t1 = timeit.default_timer()
    u_rF    = gauss2d(u_r,       lambdaTh, lambdaZ, r, th, z, 1)
    u_thF   = gauss2d(u_th,      lambdaTh, lambdaZ, r, th, z, 1)
    u_zF    = gauss2d(u_z,       lambdaTh, lambdaZ, r, th, z, 1)
    u_rRF   = gauss2d(u_r*u_r,   lambdaTh, lambdaZ, r, th, z, 1)
    u_rThF  = gauss2d(u_r*u_th,  lambdaTh, lambdaZ, r, th, z, 1)
    u_rZF   = gauss2d(u_r*u_z,   lambdaTh, lambdaZ, r, th, z, 1)
    u_thThF = gauss2d(u_th*u_th, lambdaTh, lambdaZ, r, th, z, 1)
    u_thZF  = gauss2d(u_th*u_z,  lambdaTh, lambdaZ, r, th, z, 1)
    u_zZF   = gauss2d(u_z*u_z,   lambdaTh, lambdaZ, r, th, z, 1)
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
    pi = eflux(u_rF, u_thF, u_zF, u_rRF, u_rThF, u_rZF, u_thThF, u_thZF, u_zZF, r, th, z)
    #-------------------------------------------------------------------------------
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t2), 'seconds')
    #-------------------------------------------------------------------------------
    # compute cross-correlation
    print('-------------------------------------------------------------------------')
    print('Computing cross-correlation... ', end='', flush=True)
    t3 = timeit.default_timer()

#    np.seterr(divide='ignore', invalid='ignore')
    print('-------------------------------------------------------------------------')

    k = 63
    print ("Wall-normal plane at y+ =", (1-r[k])*180.4)
    # loop over all theta for averaging
    for l in range(nth):
        eF    =  pi[k, l, :]  # extract 2d filtered eflux 
        uvE15  = uvE[k, l, :]
        uvS15  = uvS[k, l, :]
        acE15Q2 = acE15Q2  + crosscorr(uvE15, eF) # compute auto-correaltion and sum up average      
        acE15Q4 = acE15Q4  + crosscorr(uvS15, eF) # compute auto-correaltion and sum up average      

    print('-------------------------------------------------------------------------')
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t3), 'seconds')
    print('-------------------------------------------------------------------------')

    nt = nt + 1 # increase temporal/ensemble counter

#-----------------------------------------------------------------------------------
# divide by total number of spatio-temporal samples compute mean & normalizing by RMS
acE15Q2  = acE15Q2 /(nth*nt)
acE15Q4  = acE15Q4 /(nth*nt)

print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')
#=============================================================================================

# write correlation coefficient in ascii file
dz = z - (z[-1]-z[0])/2.0 # Why?
fnam = 'crosscorreFluxQ2Q4event1d'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
print('-------------------------------------------------------------------------')
print('Writing cross-correlation to file', fnam)
f = open(fnam, 'w')
f.write('# One-dimensional two-point auto-correlations in axial (z) direction\n')
f.write('# For all three velocity components (u_r, u_th, u_z) and pressure (p)\n')
f.write("# At wall-normal location r = %f -> y+ = %f\n" % (r[k], (1-r[k])*180.4))
f.write('# Python post-processing on data set nsPipe/pipe0003 generated in a DNS using nsPipe\n')
f.write('# Temporal (ensemble) averaging over %d samples\n' % (nt))
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# Additional averaging in theta direction over %d points\n" % (nth))
f.write("# 01st column: axial separation dz in units of pipe radii (R)\n")
f.write("# 02nd column: u_r correlation in units of RMS\n")
f.write("# 03rd column: u_th correlation in units of RMS\n")
f.write("# 04th column: u_z correlation in units of RMS\n")
for i in range(nz-1):
 f.write("%23.16e %23.16e %23.16e\n" % (dz[i], acE15Q2[i], acE15Q4[i]))
f.close()
print('Written auto-correlation to file', fnam)

#=================================================================================
# plot data as graph, (0) none, (1) interactive, (2) pdf
plot = 2
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
ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
ax1.set_ylabel(r"$C_{Q_2 \pi}$")
ax1.plot(dz, acE15Q2, color=Blue, linestyle='-') # .format(lambdaTh, lambdaZ))
ax1.plot(dz, line, color=Vermillion, linestyle='--', linewidth=0.5) # .format(lambdaTh, lambdaZ))
ax1.plot(line, acE15Q2, color=Vermillion, linestyle='--', linewidth=0.5) # .format(lambdaTh, lambdaZ))
ax1.title.set_text(r"$y^+ = 15$")

# plot one-dimensional two-point auto-correlation
ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
ax2.set_ylabel(r"$C_{Q_4 \pi}$")
ax2.plot(dz, acE15Q4, color=Blue, linestyle='-') # .format(lambdaTh, lambdaZ))
ax2.plot(dz, line, color=Vermillion, linestyle='--', linewidth=0.5) # .format(lambdaTh, lambdaZ))
ax2.plot(line, acE15Q4, color=Vermillion, linestyle='--', linewidth=0.5) # .format(lambdaTh, lambdaZ))

# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = 'crosscorreFluxQ2Q4event1d'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.pdf'
 plt.savefig(fnam)
 print('Written file', fnam)
print('=============================================================================================')
fig.clf()





