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
# Usage:    python piCorr1dBoxZ.py 
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
#--------------------------------------------------
# range of state files to read from flow field data
iFirst =  570000
iLast  =  875000
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
#fnam = 'statistics00570000to00570000nt0001.dat'
fnam = '../../statistics00570000to00875000nt0062.dat'

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
    gTh = np.sinc(kTh*np.pi*lambdaTh) # Box sinc kernel in theta direction
    gZ  = np.sinc(kZ*np.pi*lambdaZ)    # Box sinc kernel in axial direction
    g2d = np.outer(gTh, gZ)          # 2d filter kernel in theta-z plane

    # apply 2d filter kernel in Fourier space via FFT
    uFiltered=np.fft.ifft2(np.fft.fft2(u)*g2d)

    return uFiltered.real


#--------------------------------------------------------------------------------
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
    uFiltered=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(box2dThZ)(u[i,:,:],r[i], lambdaTh, lambdaZ, th ,z) for i in range(len(r))))

    # uncomment for Härtel hack: constant angle instead of constant arc length
    #rRef = 0.88889 # radial reference location where given arc length is converted to used filter angle
    #uFiltered = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(box2dThZ)(u[i,:,:], rRef, lambdaTh, lambdaZ, th, z) for i in range(len(r))))

    # simple serial version
    # for i in range(len(r)):
    #uFiltered[i] = box2dThZ(u[i,:,:], r[i], lambdaTh, lambdaZ, th, z)

    return uFiltered



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
# reset ensemble counter and statistical moments
nt = 0
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
    step=f['grid'].attrs.__getitem__('step')
    timeIn=f['grid'].attrs.__getitem__('time')
    Re=f['setup'].attrs.__getitem__('Re')
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
    u_rF    = box2d(u_r,       lambdaTh, lambdaZ, r, th, z)
    u_thF   = box2d(u_th,      lambdaTh, lambdaZ, r, th, z)
    u_zF    = box2d(u_z,       lambdaTh, lambdaZ, r, th, z)
    u_rRF   = box2d(u_r*u_r,   lambdaTh, lambdaZ, r, th, z)
    u_rThF  = box2d(u_r*u_th,  lambdaTh, lambdaZ, r, th, z)
    u_rZF   = box2d(u_r*u_z,   lambdaTh, lambdaZ, r, th, z)
    u_thThF = box2d(u_th*u_th, lambdaTh, lambdaZ, r, th, z)
    u_thZF  = box2d(u_th*u_z,  lambdaTh, lambdaZ, r, th, z)
    u_zZF   = box2d(u_z*u_z,   lambdaTh, lambdaZ, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t1), 'seconds')
    #-------------------------------------------------------------------------------    

    # compute instantaneous energy flux
    t2 = timeit.default_timer()
    print('-------------------------------------------------------------------------')
    print('Computing energy flux... ', end='', flush=True)
    pi = np.array(eflux(u_rF, u_thF, u_zF, u_rRF, u_rThF, u_rZF, u_thThF, u_thZF, u_zZF, r, th, z)).transpose(0,2,1)
    #-------------------------------------------------------------------------------
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t2), 'seconds')
    #-------------------------------------------------------------------------------
    print('-------------------------------------------------------------------------')
    #-------------------------------------------------------------------------------
    fnam = 'efluxBox_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    out = h5py.File(fnam, 'w')

    fields=out.create_group("fields")

    scale=out.create_group("scale")
    scale.attrs.create("deltaR", data=lambdaR)
    scale.attrs.create("deltaTh", data=lambdaTh)
    scale.attrs.create("deltaZ", data=lambdaZ)

    eFlux=out.create_group("fields/pi")
    eFlux.create_dataset("pi", data=pi)

    grid=out.create_group("grid")
    grid.attrs.create("step", data=step)
    grid.attrs.create("time", data=timeIn)
    grid.create_dataset("r", data=r)
    grid.create_dataset("th", data=th)
    grid.create_dataset("z", data=z)

    setup=out.create_group("setup")
    setup.attrs.create("Re", data=Re)

    out.close()
    #-------------------------------------------------------------------------------
    #% write XMF file
    #for further info visit https://pymotw.com/2/xml/etree/ElementTree/create.html 
    #& http://www.xdmf.org/index.php/XDMF_Model_and_Format
    import xml.etree.cElementTree as ET
    from xml.dom import minidom

    def prettify(elem):
   # """Return a pretty-printed XML string for the Element."""
     rough_string = ET.tostring(elem, 'utf-8')
     reparsed = minidom.parseString(rough_string)
     return reparsed.toprettyxml(indent="  ")

    root = ET.Element(" ")
    root.append(ET.Comment('DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []'))

    xdmf = ET.Element('Xdmf')
    xdmf.set("version", "2.0")
    domain=ET.SubElement(xdmf, "Domain")
    grid=ET.SubElement(domain, "Grid")#, {"Topology":"3DRectMesh"})

    topology=ET.SubElement(grid,'Topology')#('''<root><Topology ToplogyType="3DRectMesh" /></root>''')#ET.SubElement
    topology.set("TopologyType","3DRectMesh")
    topology.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))

    grid.extend(topology)
    grid.set("Name", "mesh")
    grid.set("GridType", "Uniform")

    geometry=ET.SubElement(grid, "Geometry")
    geometry.set("GeometryType","VXVYVZ")

    dataItemTh=ET.SubElement(geometry, "DataItem")
    dataItemTh.set("Dimensions",str(len(th)))
    dataItemTh.set("Name", "th")
    dataItemTh.set("NumberType", "Float")
    dataItemTh.set("Precision", "8")
    dataItemTh.set("Format", "HDF")
    dataItemTh.text = fnam+":/grid/th"

    dataItemZ=ET.SubElement(geometry, "DataItem")
    dataItemZ.set("Dimensions",str(len(z)))
    dataItemZ.set("Name", "z")
    dataItemZ.set("NumberType", "Float")
    dataItemZ.set("Precision", "8")
    dataItemZ.set("Format", "HDF")
    dataItemZ.text = fnam+":/grid/z"

    dataItemR=ET.SubElement(geometry, "DataItem")
    dataItemR.set("Dimensions",str(len(r)))
    dataItemR.set("Name", "r")
    dataItemR.set("NumberType", "Float")
    dataItemR.set("Precision", "8")
    dataItemR.set("Format", "HDF")
    dataItemR.text = fnam+":/grid/r"

    time = ET.SubElement(grid, "Time")
    time.set("Value",str(timeIn))
    grid.extend(time)

    attributePi=ET.SubElement(grid, "Attribute")
    attributePi.set("Name", "pi")
    attributePi.set("AttributeType", "Scalar")
    attributePi.set("Center","Node")

    dataItemPi=ET.SubElement(attributePi, "DataItem")
    dataItemPi.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemPi.set("NumberType", "Float")
    dataItemPi.set("Precision", "8")
    dataItemPi.set("Format", "HDF")
    dataItemPi.text = fnam+":/fields/pi/pi"

  
    fnam = 'efluxBox_pipe0002_'+'{:08d}'.format(iFile)+'.xmf'
    with open(fnam, "w+") as f:
    #f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []> \n')
        print(prettify(xdmf), file=f)
        print

    #add Declatation, workaround to ET    
    declaration='<!DOCTYPE Xdmf SYSTEM "xdmf.dtd" []>\n'
    f = open(fnam, "r")
    contents = f.readlines()
    f.close()

    contents.insert(1, declaration)

    f = open(fnam, "w")
    contents = "".join(contents)
    f.write(contents)
    f.close()
print('-------------------------------------------------------------------------')
print('-------------------------------JOB-ENDED---------------------------------')
print('-------------------------------------------------------------------------')


