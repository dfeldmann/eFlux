#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:58:14 2018
pipe0002, according to 1994, Haertel at al
@author: Jan Chen
Last Edit on 19. Dec 2018
"""
#import h5py #for h5 files/nsPipe, nsCouette
from netCDF4 import Dataset #for netCdf files/openpieflow
import numpy as np 
import math
#import os.path
from joblib import Parallel, delayed
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Print amount of processor in use
print("using "+ str(multiprocessing.cpu_count())+" cpus")

#define netCdf reader
def ncdump(nc_fid, verb=True):
    def print_ncattr(key):
        try:
            print ("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print ('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print ("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print ("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print ('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print ("NetCDF dimension information:")
        for dim in nc_dims:
            print ("\tName:", dim)
            print ("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print ("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print ('\tName:', var)
                print ("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print ("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars
#!!!
# open NetCDF file, load data, define file name and working path
#nc_f = '/home/jc/Desktop/MasterThesis/Analysis/pipe0062/outFiles/state00019982.cdf.dat'  # Your filename

#Define input filenames
n=0
pathIn="outFiles/"
pathOut="filteredFiles/isotropic/"
filenameIn="state00000{0}.cdf.phy"
filename=(pathIn+filenameIn.format(700+n), pathIn+'vel_meanstdv00000100to00001100.dat')
filenameOut='_3+Iso3dOpt2'

#extract data from file
nc_f = filename[0]  # Your filename
nc_fid = Dataset(nc_f, 'r') 

# print some infos
print ('Open file', nc_f, '(',nc_fid.file_format,')')
nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)

#extract grid data
r = np.array(nc_fid.variables['r'][:]) # radial coordinates
#print ('Read', r)
th = np.array(nc_fid.variables['th'][:]) # azimutal coordinates
#print ('Read', th)
z = np.array(nc_fid.variables['z'][:])# radial coordinates

timeIn=nc_fid.getncattr('t') # time
Re=nc_fid.getncattr('Re') # radial coordinates
print('Re= '+str(Re))
alpha=nc_fid.getncattr('alpha')# radial coordinates
# define filter functions and filter widths

#load converges profile <u_z>
#assume u_th, u_r converged -> only turbulent 
u_zM= np.loadtxt(filename[1])[:,5]

#calculate distances globally, for later calculation
dth = th[1]-th[0]
dz = z[1]-z[0]
thR=np.outer(r,th) #circumference dependent on r

#set filter width "normalised" with ReTau, in this case isotropic
deltaR=3/180 #
deltaThR=3/180 #Re_tau=180 and 
deltaZ=3/180
print("deltaR=", deltaR, "deltaThR=", deltaThR, "deltaZ=",deltaZ)

#---Define Filter
#1d filters with different boundary Conditions
def box1dInterp(uField, delta, r, option):
    rLenFine=math.ceil(r[len(r)-1]/(r[len(r)-1]-r[len(r)-2])) #define length of yI
    rI=np.linspace(r[0],r[len(r)-1],rLenFine)
    uI=np.interp(rI,r,uField)
    #!consider component direction! Option1:Axis-Mirrored, Wall-zeroed; O2:A-mirrored, wall-mirrorec, O3:A-extended, W-zeroed, O4:A-extended, W-mirrored
    if option == 1:#Axis mirrored, wall zeroed -> Physical convolution      
        uModified=np.append(np.append(uI[len(uI)//2:0:-1], uI), np.zeros(len(uI)//2))
        rModified=np.append(rI,rI[len(rI)-1]+rI[1:len(rI)]) #shifted by 0.5 R, appended array include 0-point, as in later data n0 is not associate with r=0
        rShift=np.append(rModified[0]-rModified[:0:-1],rModified)
        N=math.ceil(round(delta/(rI[1]-rI[0])))
        gB=1/(N+1)*np.heaviside(delta/2-abs(rShift),1)
        uFiltered=np.convolve(gB,uModified,"same")[3*(len(gB)-1)//8+1:5*(len(gB)-1)//8+2]
    if option == 2:#Axis mirrored, wall mirrored -> Fourier Convolution
        uModified=np.append(uI[len(uI)-1::-1], uI)
        kR=np.fft.fftfreq(2*len(rI), d=rI[1]-rI[0])
        uFiltered=np.fft.ifft(np.fft.fft(uModified)*np.sinc(kR*delta)).real[len(uModified)//2:]#[3*(len(gB)-1)//8+1:5*(len(gB)-1)//8+2]       
    if option == 3:#Axis constant, wall zeroed-> Physical convolution
        uModified=np.append(np.append(np.full((len(uI)//2), uI[0]), uI), np.zeros(len(uI)//2))
        rModified=np.append(rI,rI[len(rI)-1]+rI[1:len(rI)]) #shifted by 0.5 R, appended array include 0-point, as in later data n0 is not associate with r=0
        rShift=np.append(rModified[0]-rModified[:0:-1],rModified)
        N=math.ceil(round(delta/(rI[1]-rI[0])))
        gB=1/(N+1)*np.heaviside(delta/2-abs(rShift),1)
        uFiltered=np.convolve(gB,uModified,"same")[3*(len(gB)-1)//8+1:5*(len(gB)-1)//8+2]
    if option == 4:#Axis constant, wall mirrored-> Physical convolution
        uModified=np.append(np.append(np.full((len(uI)//2), uI[0]), uI), uI[len(uI)-2:len(uI)//2-2:-1])
        rModified=np.append(rI,rI[len(rI)-1]+rI[1:len(rI)]) #shifted by 0.5 R, appended array include 0-point, as in later data n0 is not associate with r=0
        rShift=np.append(rModified[0]-rModified[:0:-1],rModified)
        N=math.ceil(round(delta/(rI[1]-rI[0])))
        gB=1/(N+1)*np.heaviside(delta/2-abs(rShift),1)
        uFiltered=np.convolve(gB,uModified,"same")[3*(len(gB)-1)//8+1:5*(len(gB)-1)//8+2]
    uFiltered=np.interp(r,rI,uFiltered)
    uFiltered[len(r)-1]=0
    return uFiltered
#%
def sharp1dInterp(u, delta, r, option):
   # rN=r-r[0] #normalize so that r starts at 0 for numerical operations, not needed here since r starts at 0
    rLenFine=math.ceil(r[len(r)-1]/(r[len(r)-1]-r[len(r)-2])) #define length of yI
    rI=np.linspace(r[0],r[len(r)-1],rLenFine)
    uI=np.interp(rI,r,u)  
    if option == 1:#Axis mirrored, wall zeroed -> Physical convolution      
        uModified=np.append(np.append(uI[len(uI)//2:0:-1], uI), np.zeros(len(uI)//2))
        rModified=np.append(rI,rI[len(rI)-1]+rI[1:len(rI)]) #shifted by 0.5 R, appended array include 0-point, as in later data n0 is not associate with r=0
        rShift=np.append(rModified[0]-rModified[:0:-1],rModified)
        N=math.ceil(round(delta/(rI[1]-rI[0])))
        gS=1/(N+1)*np.sinc(rShift/delta)
        uFiltered=np.convolve(gS,uModified,"same")[3*(len(gS)-1)//8+1:5*(len(gS)-1)//8+2]
    if option == 2:#Axis mirrored, wall mirrored -> Fourier Convolution
        uModified=np.append(uI[len(uI)-1::-1], uI)
        kR=np.fft.fftfreq(2*len(rI), d=rI[1]-rI[0])
        uFiltered=np.fft.ifft(np.fft.fft(uModified)*np.heaviside(1/(delta)-abs(2*kR),1)).real[len(uModified)//2:]    
    if option == 3:#Axis constant, wall zeroed-> Physical convolution
        uModified=np.append(np.append(np.full((len(uI)//2), uI[0]), uI), np.zeros(len(uI)//2))
        rModified=np.append(rI,rI[len(rI)-1]+rI[1:len(rI)]) #
        rShift=np.append(rModified[0]-rModified[:0:-1],rModified)
        N=math.ceil(round(delta/(rI[1]-rI[0])))
        gS=1/(N+1)*np.sinc(rShift/delta)
        uFiltered=np.convolve(gS,uModified,"same")[3*(len(gS)-1)//8+1:5*(len(gS)-1)//8+2]
    if option == 4:#Axis constant, wall mirrored-> Physical convolution
        uModified=np.append(np.append(np.full((len(uI)//2), uI[0]), uI), uI[len(uI)-2:len(uI)//2-2:-1])
        rModified=np.append(rI,rI[len(rI)-1]+rI[1:len(rI)]) #shifted by 0.5 R, appended array include 0-point, as in later data n0 is not associate with r=0
        rShift=np.append(rModified[0]-rModified[:0:-1],rModified)
        N=math.ceil(round(delta/(rI[1]-rI[0])))
        gS=1/(N+1)*np.sinc(rShift/delta)
        uFiltered=np.convolve(gS,uModified,"same")[3*(len(gS)-1)//8+1:5*(len(gS)-1)//8+2]
    uFiltered=np.interp(r,rI,uFiltered)
    uFiltered[len(r)-1]=0
    return uFiltered

def gauss1dInterp(u, delta, r, option):
    rLenFine=math.ceil(r[len(r)-1]/(r[len(r)-1]-r[len(r)-2])) #define length of yI
    rI=np.linspace(r[0],r[len(r)-1],rLenFine)
    uI=np.interp(rI,r,u)
    if option == 1:#Axis mirrored, wall zeroed -> Physical convolution      
        uModified=np.append(np.append(uI[len(uI)//2:0:-1], uI), np.zeros(len(uI)//2))
        rModified=np.append(rI,rI[len(rI)-1]+rI[1:len(rI)]) #shifted by 0.5 R, appended array include 0-point, as in later data n0 is not associate with r=0
        rShift=np.append(rModified[0]-rModified[:0:-1],rModified)
        N=math.ceil(round(delta/(rI[1]-rI[0])))
        gG=1/(N+1)*(6/np.pi)**0.5*np.exp(-6*rShift**2/delta**2)
        uFiltered=np.convolve(gG,uModified,"same")[3*(len(gG)-1)//8+1:5*(len(gG)-1)//8+2]
    if option == 2:#Axis mirrored, wall mirrored -> Fourier Convolution
        uModified=np.append(uI[len(uI)-1::-1], uI)
        kR=np.fft.fftfreq(2*len(rI), d=rI[1]-rI[0])
        uFiltered=np.fft.ifft(np.fft.fft(uModified)*np.exp(-((2*np.pi*kR)**2*delta**2/24))).real[len(uModified)//2:] 
    if option == 3:#Axis constant, wall zeroed-> Physical convolution
        uModified=np.append(np.append(np.full((len(uI)//2), uI[0]), uI), np.zeros(len(uI)//2))
        rModified=np.append(rI,rI[len(rI)-1]+rI[1:len(rI)]) #
        rShift=np.append(rModified[0]-rModified[:0:-1],rModified)
        N=math.ceil(round(delta/(rI[1]-rI[0])))
        gG=1/(N+1)*(6/np.pi)**0.5*np.exp(-6*rShift**2/delta**2)
        uFiltered=np.convolve(gG,uModified,"same")[3*(len(gG)-1)//8+1:5*(len(gG)-1)//8+2]
    if option == 4:#Axis constant, wall mirrored-> Physical convolution
        uModified=np.append(np.append(np.full((len(uI)//2), uI[0]), uI), uI[len(uI)-2:len(uI)//2-2:-1])
        rModified=np.append(rI,rI[len(rI)-1]+rI[1:len(rI)]) #shifted by 0.5 R, appended array include 0-point, as in later data n0 is not associate with r=0
        rShift=np.append(rModified[0]-rModified[:0:-1],rModified)
        N=math.ceil(round(delta/(rI[1]-rI[0])))
        gG=1/(N+1)*(6/np.pi)**0.5*np.exp(-6*rShift**2/delta**2)
        uFiltered=np.convolve(gG,uModified,"same")[3*(len(gG)-1)//8+1:5*(len(gG)-1)//8+2]
    uFiltered=np.interp(r,rI,uFiltered)
 #   uFiltered[len(r)-1]=0
    return uFiltered

#1d filter for 1d filter on 3d velocity used in gauss1dZV1, gauss1dThV1
def gauss1dZ(u, rPos, deltaThR, deltaZ, th, z): #output in Fourier modes
    kZ=np.fft.fftfreq(len(z), d=z[1]-z[0])
    g=np.outer(np.ones(th.shape),np.exp(-((2*np.pi*kZ)**2*deltaZ**2/24)))
    uFiltered=np.fft.ifft2(np.fft.fft2(u)*g)
    return uFiltered.real

def gauss1dTh(u, rPos, deltaThR, deltaZ, th, z): #output in Fourier modes
    kTh=np.fft.fftfreq(len(th), d=th[1]-th[0]) #=kX=kappaX/(2*pi); kappaX=Wavenumber
    g=np.outer(np.exp(-((2*np.pi*kTh)**2*deltaThR**2/24)),np.ones(z.shape))
    uFiltered=np.fft.ifft2(np.fft.fft2(u)*g)
    return uFiltered.real

#2d filter
def box2dZTh(u, rPos, deltaThR, deltaZ, th, z):
    kTh=np.fft.fftfreq(len(th), d=th[1]-th[0]) #=kX=kappaX/(2*pi) !!need to take into account radial position->transferfunction g
    kZ=np.fft.fftfreq(len(z), d=z[1]-z[0])
    g=np.outer(np.sinc(kTh/rPos*deltaThR),np.sinc(kZ*deltaZ))
    uFiltered=np.fft.ifft2(np.fft.fft2(u)*g)
    return uFiltered.real

def sharp2dZTh(u, rPos, deltaThR, deltaZ, th, z): #output in Fourier modes
    kTh=np.fft.fftfreq(len(th), d=th[1]-th[0]) #=kX=kappaX/(2*pi); kappaX=Wavenumber
    kZ=np.fft.fftfreq(len(z), d=z[1]-z[0])
    kThC=1/deltaThR #1/pi for kXC, kYC and kY kY in this fct
    kZC=1/deltaZ   
    g=np.outer(np.heaviside(kThC-abs(2*kTh/rPos),1),np.heaviside(kZC-abs(2*kZ),1))
    uFiltered=np.fft.ifft2(np.fft.fft2(u)*g)
    return uFiltered.real

def gauss2dZTh(u, rPos, deltaThR, deltaZ, th, z):
    kTh=np.fft.fftfreq(len(th), d=rPos*(th[1]-th[0])) #=kX=kappaX/(2*pi)
    kZ=np.fft.fftfreq(len(z), d=z[1]-z[0])
    g=np.outer(np.exp(-((2*np.pi*kTh)**2*deltaThR**2/24)),np.exp(-((2*np.pi*kZ)**2*deltaZ**2/24)))
    uFiltered=np.fft.ifft2(np.fft.fft2(u)*g)
    return uFiltered.real

#3d filter on 3d vel field
def box3d(u, deltaR, deltaThR, deltaZ, r, th, z, option):
    uFiltered=np.zeros((len(r),len(th),len(z)))
    uFiltered=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(box2dZTh)(u[i],r[i],deltaThR,deltaZ, th ,z) for i in range(len(r))))
    #Filter in physical space in r
    u_r2d=uFiltered.reshape(len(r),len(th)*len(z))
    u_r2dF=np.zeros((len(r),(len(th)*len(z))))
    u_r2dF=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(box1dInterp)(u_r2d[:,i],deltaR,r,option) for i in range(len(th)*len(z))))
    uFiltered=u_r2dF.transpose().reshape(u.shape)
    #uFiltered=u_r2dF.transpose().reshape(u.shape)
    return uFiltered

def sharp3d(u, deltaR, deltaThR, deltaZ, r, th, z, option):
    uFiltered=np.zeros((len(r),len(th),len(z)))
    uFiltered=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(sharp2dZTh)(u[i],r[i],deltaThR,deltaZ, th ,z) for i in range(len(r))))
    #Filter in physical space in r
    u_r2d=uFiltered.reshape(len(r),len(th)*len(z))
    u_r2dF=np.zeros((len(r),(len(th)*len(z))))
    u_r2dF=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(sharp1dInterp)(u_r2d[:,i],deltaR,r,option) for i in range(len(th)*len(z))))
    uFiltered=u_r2dF.transpose().reshape(u.shape)
    #uFiltered=u_r2dF.transpose().reshape(u.shape)
    return uFiltered

def gauss3d(u, deltaR, deltaThR, deltaZ, r, th, z, option):
    uFiltered=np.zeros((len(r),len(th),len(z)))
    uFiltered=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gauss2dZTh)(u[i],r[i],deltaThR,deltaZ, th ,z) for i in range(len(r))))
    #Filter in physical space in r
    u_r2d=uFiltered.reshape(len(r),len(th)*len(z))
    u_r2dF=np.zeros((len(r),(len(th)*len(z))))
    u_r2dF=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gauss1dInterp)(u_r2d[:,i],deltaR,r,option) for i in range(len(th)*len(z))))
    uFiltered=u_r2dF.transpose().reshape(u.shape)
    #uFiltered=u_r2dF.transpose().reshape(u.shape)
    return uFiltered

#2d filter on 3d vel field
def gauss2d(u, deltaR, deltaThR, deltaZ, r, th, z, option):
    uFiltered=np.zeros((len(r),len(th),len(z)))
    uFiltered=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gauss2dZTh)(u[i],r[i],deltaThR,deltaZ, th ,z) for i in range(len(r))))
    return uFiltered

#1d filter on 3d vel field
def gauss1dZV1(u, deltaR, deltaThR, deltaZ, r, th, z, option):
    uFiltered=np.zeros((len(r),len(th),len(z)))
    uFiltered=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gauss1dZ)(u[i],r[i],deltaThR,deltaZ, th ,z) for i in range(len(r))))
    return uFiltered

def gauss1dThV1(u, deltaR, deltaThR, deltaZ, r, th, z, option):
    uFiltered=np.zeros((len(r),len(th),len(z)))
    uFiltered=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gauss1dTh)(u[i],r[i],deltaThR,deltaZ, th ,z) for i in range(len(r))))
    return uFiltered

def gauss1dRV1(u, deltaR, deltaThR, deltaZ, r, th, z, option):
    u_r2d=u.reshape(len(r),len(th)*len(z))
    u_r2dF=np.zeros((len(r),(len(th)*len(z))))
    u_r2dF=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(gauss1dInterp)(u_r2d[:,i],deltaR,r,option) for i in range(len(th)*len(z))))
    uFiltered=u_r2dF.transpose().reshape(u.shape)
    #uFiltered=u_r2dF.transpose().reshape(u.shape)
    return uFiltered

#eflux calculation
def eflux(u1,u2,u3,u11,u12,u13,u22,u23,u33,option): #1=r; 2=th, 3=z   
    pi=np.zeros((u1.shape))
    r3d=np.tile(r,(len(z),len(th),1)).T
    #calculate strain tensor (following is based on strain tensor)
    if option == 1:       
        s11= np.gradient(u1, r, axis=0)  #!r not equidistant, therefore positional information needed
        s22= 1/r3d*np.gradient(u2,dth,axis=1)+u1/r3d
        s33= np.gradient(u3,dz,axis=2)
        s12= .5*(1/r3d*np.gradient(u1,dth,axis=1)+np.gradient(u2,r,axis=0)-u2/r3d)
        s13= .5*(np.gradient(u1,dz,axis=2)+np.gradient(u3,r,axis=0))
        s21=s12
        s23=.5*(1/r3d*np.gradient(u3,dth,axis=1)+np.gradient(u2,dz,axis=2))
        s32=s23  
        s31=s13  
    #filtered rate of strain tensor
    else:
        s11= np.gradient(u1, r, axis=0)  #!r not equidistant, therefore positional information needed
        s22= 1/r3d*np.gradient(u2,dth,axis=1)+u1/r3d
        s33= np.gradient(u3,dz,axis=2)
        s12= 1/r3d*np.gradient(u1,dth,axis=1)-u2/r3d
        s13= np.gradient(u1,dz,axis=2)
        s21= np.gradient(u2,r,axis=0)
        s23= np.gradient(u2,dz,axis=2)
        s31= np.gradient(u3,r,axis=0)  
        s32= 1/r3d*np.gradient(u3,dth,axis=1)      
    #calculate stress tensor tau for filter length scale l(see Xiao et al. 2009)
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

#% For loop over 100 state files
piSum=0
n=0
for i in range(100): #amount of files     
    nc_f = pathIn+filenameIn.format(700+n)  # Your filename
    nc_fid = Dataset(nc_f, 'r') 
    #extract/copy data from NetCDF file
    u_r = np.array(nc_fid.variables['u_r'][:,:,:])# radial velocity component
    u_th = np.array(nc_fid.variables['u_th'][:,:,:])# azimuthal velocity component
    u_z = np.array(nc_fid.variables['u_z'][:,:,:]-np.tile(u_zM,(len(z),len(th),1)).T) # axial velocity component subtracted Mean
    u_rF=gauss3d(u_r,deltaR,deltaThR,deltaZ,r,th,z,2)
    u_thF=gauss3d(u_th,deltaR,deltaThR,deltaZ,r,th,z,2)
    u_zF=gauss3d(u_z,deltaR,deltaThR,deltaZ,r,th,z,2)
    u_rRF=gauss3d(u_r*u_r,deltaR,deltaThR,deltaZ,r,th,z,2)
    u_rThF=gauss3d(u_r*u_th,deltaR,deltaThR,deltaZ,r,th,z,2)
    u_rZF=gauss3d(u_r*u_z,deltaR,deltaThR,deltaZ,r,th,z,2)
    u_thThF=gauss3d(u_th*u_th,deltaR,deltaThR,deltaZ,r,th,z,2)
    u_thZF=gauss3d(u_th*u_z,deltaR,deltaThR,deltaZ,r,th,z,2)
    u_zZF=gauss3d(u_z*u_z,deltaR,deltaThR,deltaZ,r,th,z,2)
    #Calculate pi and take mean over space
    pi=eflux(u_rF,u_thF,u_zF,u_rRF,u_rThF,u_rZF,u_thThF,u_thZF,u_zZF,2)
    pi=pi.mean(1).mean(1)
    piSum=piSum+pi
    n=n+1
piExp0=piSum/n

#write out pi and r starting from wall
f=open(pathOut+"piExpVarDelta{0}".format(n)+filenameOut+".dat", "w+")
for i in range(len(r)):
    f.write("%0.8E %0.8E\n" % (1-r[::-1][i], piExp0[::-1][i]))
f.close()
print(pathOut+"piExpVarDelta{0}".format(n)+filenameOut+".dat") #print path/filename
print("deltaR=", deltaR, "deltaThR=", deltaThR, "deltaZ=",deltaZ) #pring filter width


