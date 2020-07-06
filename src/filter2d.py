# This file contains definitions of functions for two-dimensional (2d) Fourier, Gauss
# and box filters apllied to three-dimensional (3d) data defined in a cylindrical
# co-ordinate frame work (r, th, z). All filters are applied in Fourier space making use
# of the convolution theorem and 2d FFT operations, regardless of the type of the filter
# kernel. To use the filter, put 'import filters as f' on the header of your main python
# script and call the filter operation as follows:
# Fourier: f.fourier2d(u, lambdaTh, lambdaZ, r, th, z, rect)
# Gauss:   f.gauss2d(u, lambdaTh, lambdaZ, r, th, z)
# Box:     f.box2d(u, lambdaTh, lambdaZ, r, th, z)

import sys
import os.path
import timeit
import math
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

def fourier2dThZ(u, rPos, lambdaTh, lambdaZ, th, z, rect):
    # 2d Fourier filter (aka sharp spectral cut-off) applied to a 2d data slice in a
    # wall-parallel th-z-plane defined in a cylindrical co-ordinate frame work (r,th,z).
    # Applied in Fourier space via FFT and convolution theorem.
    # Input parameters:
    # u:         2d scalar field, e.g. one velocity component
    # rPos:      radial location (r) of wall-parallel data slice
    # lambdaTh:  filter width in theta direction, arc length in unit length
    # lambdaZ:   filter width in z direction in unit length
    # th:        1d azimuthal (th) grid vector in unit radian
    # z:         1d axial (z) grid vector in unit length
    # rect:      switch for rectangular or elliptical (circular) kernel formulation
    # Returns:
    # uFiltered: 2d scalar field, which is 2d-filtered in theta and z direction

    # set-up sample spacing in unit length (pipe radii R, gap width d, etc)
    deltaTh = (th[1]-th[0])*rPos # equidistant but r dependent
    deltaZ  = ( z[1]- z[0])      # equidistant
    
    # set-up wavenumber vector in unit cycle per length (1/R)
    kTh = np.fft.fftfreq(len(th), d=deltaTh) # homogeneous azimuthal (th) direction
    kZ  = np.fft.fftfreq(len(z),  d=deltaZ)  # homogeneous axial (z) direction
    
    # set-up wavenumber vector in unit radian per length (2pi/R)
    kappaTh = 2.0*np.pi*kTh # homogeneous azimuthal (th) direction
    kappaZ  = 2.0*np.pi*kZ  # homogeneous axial (z) direction
    
    # define cut-off wavenumber in unit radian per length (2pi/R)
    kappaThC = np.pi/lambdaTh # azimuthal (th) direction
    kappaZC  = np.pi/lambdaZ  # axial (z) direction
    
    # construct 2d filter kernel
    if rect == 1:
       gTh = np.heaviside(kappaThC-abs(kappaTh), 1) # 1d azimuthal (th) kernel, step function
       gZ  = np.heaviside(kappaZC -abs(kappaZ),  1) # 1d axial (z) kernel, step function
       g2d = np.outer(gTh, gZ)                      # 2d (rectangular) kernel in th-z-plane
    else:
       # please implement elliptical kernel here
       sys.exit('\nERROR: Set rect=1. Circular/elliptical kernel not implemented yet...'
    
    # report kernel
    # if report == 1:
    # printKernel() please implement output in form of plot and ascii file
       
    # apply 2d filter kernel in Fourier space via FFT
    uFiltered = np.fft.ifft2(np.fft.fft2(u)*g2d)
    
    return uFiltered.real

def fourier2d(u, lambdaTh, lambdaZ, r, th, z, rect=1):
    # 2d Fourier filter (aka sharp spectral cut-off) applied to a 3d data set defined in a
    # cylindrical co-ordinate frame work (r, th, z). Wrapper function, which simply calls the
    # function fourier2dThZ() multiple times (in parallel) for every wall normal loaction (r).
    # Input parameters:
    # u:         2d scalar field, e.g. one velocity component
    # lambdaTh:  filter width in theta direction, arc length in unit length
    # lambdaZ:   filter width in z direction in unit length
    # r:         1d radial (r) grid vector in unit length
    # th:        1d azimuthal (th) grid vector in unit radian
    # z:         1d axial (z) grid vector in unit length
    # rect:      switch for rectangular or elliptical (circular) kernel formulation
    # Returns:
    # uFiltered: 3d scalar field, which is 2d-filtered in theta and z direction for every radial location (r)
                
    # construct output array of correct shape filled with zeros
    uFiltered = np.zeros((len(r), len(th), len(z)))
    
    # filter each wall-normal (theta-z) plane separately, this can be done in parallel
    # uFiltered = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(fourier2dThZ)(u[i,:,:], r[i], lambdaTh, lambdaZ, th, z, rect) for i in range(len(r))))
    
    # uncomment for Härtel hack: constant angle instead of constant arc length
    rRef = 0.986 # radial reference location, where given arc length is converted to used filter angle
    uFiltered = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(fourier2dThZ)(u[i,:,:], rRef, lambdaTh, lambdaZ, th, z, rect) for i in range(len(r))))
    
    # simple serial version
    # for i in range(len(r)):
    # uFiltered[i] = fourier2dThZ(u[i,:,:], r[i], lambdaTh, lambdaZ, th, z, rect)
    
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

