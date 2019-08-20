# This script contains the definitions of Fourier, Gauss and Box filters
# So far only the 2d filters revised by Umair, TODO: add all 3d filters
# here Jan has already implemented
# To use a filter, put 'import filters as f' on the header of your main script
# Fourier filter:  f.fourier2d(u, rPos, lambdaTh, lambdaZ, th, z, rect)
# Gauss   filter:  f.gauss2d  (u, lambdaTh, lambdaZ, r, th, z)
# Box     filter:  f.box2d    box2d(u, lambdaTh, lambdaZ, r, th, z)

import sys
import os.path
import timeit
import math
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

def fourier2dThZ(u, rPos, lambdaTh, lambdaZ, th, z, rect):
#--------------------------------------------------------------------------
# 2d Fourier space cut-off (aka sharp spectral) filter for 2d data slices in
# wall-parallel (theta-z) planes of a cylindrical co-ordinate frame work
# (r,th,z). Applied in Fourier space via FFT.
#
# Parameters:
# u: 2d scalar field
# rPos: radial location of wall-parallel data slice
# lambdaTh: filter width in theta direction, arc length in unit distance
# lambdaZ: filter width in z direction
# th: 1d grid vector, theta co-ordinate in unit radian
# z: 1d grid vector, z co-ordinate in unit length
# rect: switch for rectangular or elliptical (circular) kernel formulation
#
# Returns: A 2d scalar field which is 2d-filtered in theta and z direction
#-------------------------------------------------------------------------

    # sample spacing in each direction in unit length (pipe radii R, gap width d, etc)
    deltaTh = (th[1] - th[0]) * rPos # equidistant but r dependent
    deltaZ  = ( z[1] -  z[0])        # equidistant

    # set-up wavenumber vector in units of cycles per unit distance (1/R)
    kTh = np.fft.fftfreq(len(th), d=deltaTh) # homogeneous direction theta
    kZ  = np.fft.fftfreq(len(z),  d=deltaZ)  # homogeneous direction z

    # define cut-off wavenumber in units of cycles per unit distance (1/R)
    # --------------------------------------------------------------------
    # 
    # To be consistent with Pope we take the cutoff wave no. as given on page no. 563
    # kappaC=pi/lambda, where 'lambda' is the filter width, and since Python is giving out 
    # the values of 'k' in cycles per metre, So, we multiply 'k' with '2pi' to get the 
    # the similar formulation as given in Pope (Page 563)

    kappaThC = np.pi/lambdaTh # Cutoff wave no. in azimuthal(Th) direction
    kappaZC  = np.pi/lambdaZ  # Cutoff wave no. in axial(z)      direction

    # construct 2d filter kernel
    # --------------------------------------------------------------------
    #
    # As states earlier to be consistent with Pope we multilply the wave no. 'kTh', obtained
    # by using the Python's fft module, with '2pi' in the definition of Heaviside function.
    # -------------------------------------------------------------------- 
    if rect == 1:
     gTh = np.heaviside(kappaThC - abs(2.0*np.pi*kTh), 1) # 1d step function in theta direction
     gZ  = np.heaviside(kappaZC  - abs(2.0*np.pi*kZ),  1) # 1d step function in z direction
     g2d = np.outer(gTh, gZ)                # 2d rectangular step function in theta-z-plane

    else:
     # please implement elliptic kernel here
     sys.exit('\nERROR: Set rect=1. Circular/elliptical kernel not implemented yet...')

    # apply 2d filter kernel in Fourier space via FFT
    uFiltered = np.fft.ifft2(np.fft.fft2(u) * g2d)

    return uFiltered.real

def fourier2d(u, lambdaTh, lambdaZ, r, th, z, rect=1):
#--------------------------------------------------------------------------
# 2d Fourier space cut-off (aka sharp spectral) filter for 3d data in a cylindrical
# co-ordinate frame work (r,th,z). Applied in Fourier space via FFT.
#
# Parameters:
# u: 3d scalar field
# lambdaTh: filter width in theta direction, arc length in unit distance
# lambdaZ: filter width in z direction
# r: 1d grid vector, radial (wall-normal) co-ordinate in unit length
# th: 1d grid vector, azimuthal co-ordinate in unit radian
# z: 1d grid vector, axial co-ordinate in unit length
#
# Returns: A 3d scalar field which is 2d-filtered in theta and z direction only
#--------------------------------------------------------------------------

    # construct output array of correct shape filled with zeros
    uFiltered = np.zeros((len(r), len(th), len(z)))

    # filter each wall-normal (theta-z) plane separately, this can be done in parallel
    #uFiltered = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(fourier2dThZ)(u[i,:,:], r[i], lambdaTh, lambdaZ, th, z, rect) for i in range(len(r))))

    # uncomment for Härtel hack: constant angle instead of constant arc length
    rRef = 0.986 # radial reference location where given arc length is converted to used filter angle
    uFiltered = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(fourier2dThZ)(u[i,:,:], rRef, lambdaTh, lambdaZ, th, z, rect) for i in range(len(r))))

    # simple serial version
    # for i in range(len(r)):
    #uFiltered[i] = fourier2dThZ(u[i,:,:], r[i], lambdaTh, lambdaZ, th, z, rect)

    return uFiltered


def gauss2dThZ(u, rPos, lambdaTh, lambdaZ, th, z):
# 2d Gauss filter for 2d data slices in
# wall-parallel (theta-z) planes of a cylindrical co-ordinate frame work
# (r,th,z). Applied in Fourier space via FFT.
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

    # apply 2d filter kernel in Fourier space via FFT
    uFiltered=np.fft.ifft2(np.fft.fft2(u)*g2d)
    return uFiltered.real


def gauss2d(u, lambdaTh, lambdaZ, r, th, z):
# 2d Gauss filter for 3d data in a cylindrical
# co-ordinate frame work (r,th,z). Applied in Fourier space via FFT.
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
    #uFiltered=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gauss2dThZ)(u[i,:,:],r[i],lambdaTh, lambdaZ, th ,z) for i in range(len(r))))

    # uncomment for Härtel hack: constant angle instead of constant arc length
    rRef = 0.986 # radial reference location where given arc length is converted to used filter angle
    uFiltered = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gauss2dThZ)(u[i,:,:], rRef, lambdaTh, lambdaZ, th, z) for i in range(len(r))))

    # simple serial version
    # for i in range(len(r)):
    #uFiltered[i] = gauss2dThZ(u[i,:,:], r[i], lambdaTh, lambdaZ, th, z)

    return uFiltered

def box2dThZ(u, rPos, lambdaTh, lambdaZ, th, z):
# 2d Box filter for 2d data slices in
# wall-parallel (theta-z) planes of a cylindrical co-ordinate frame work
# (r,th,z). Applied in Fourier space via FFT.
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
    gTh = np.sinc(kTh*lambdaTh) # Box sinc kernel in theta direction
    gZ  = np.sinc(kZ*lambdaZ)   # Box sinc kernel in axial direction
    g2d = np.outer(gTh, gZ)     # 2d filter kernel in theta-z plane

    # apply 2d filter kernel in Fourier space via FFT
    uFiltered=np.fft.ifft2(np.fft.fft2(u)*g2d)

    return uFiltered.real


def box2d(u, lambdaTh, lambdaZ, r, th, z):
# 2d Box filter for 3d data in a cylindrical
# co-ordinate frame work (r,th,z). Applied in Fourier space via FFT.
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
    #uFiltered=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(box2dThZ)(u[i,:,:],r[i], lambdaTh, lambdaZ, th ,z) for i in range(len(r))))

    # uncomment for Härtel hack: constant angle instead of constant arc length
    rRef = 0.986 # radial reference location where given arc length is converted to used filter angle
    uFiltered = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(box2dThZ)(u[i,:,:], rRef, lambdaTh, lambdaZ, th, z) for i in range(len(r))))

    # simple serial version
    # for i in range(len(r)):
    #uFiltered[i] = box2dThZ(u[i,:,:], r[i], lambdaTh, lambdaZ, th, z)

    return uFiltered

