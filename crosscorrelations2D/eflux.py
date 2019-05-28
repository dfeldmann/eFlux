import sys
import os.path
import timeit
import math
import numpy as np
import h5py
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

