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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
#===================================================================================


#===================================================================================
def fourier2d(u, lambdaTh, lambdaZ, r, th, z, rect):
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
    u_rF    = fourier2d(u_r,       lambdaTh, lambdaZ, r, th, z, 1)
    u_thF   = fourier2d(u_th,      lambdaTh, lambdaZ, r, th, z, 1)
    u_zF    = fourier2d(u_z,       lambdaTh, lambdaZ, r, th, z, 1)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t1), 'seconds')
    #-------------------------------------------------------------------------------    
    print('-------------------------------------------------------------------------')
    #-----------------------------------------------------------------------------------
    fnam = 'filteredfieldFourier_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    out = h5py.File(fnam, 'w')

    fields=out.create_group("fields")

    scale=out.create_group("scale")
    scale.attrs.create("deltaR", data=lambdaR)
    scale.attrs.create("deltaTh", data=lambdaTh)
    scale.attrs.create("deltaZ", data=lambdaZ)

#    eflux=out.create_group("fields/pi")
#    eflux.create_dataset("pi", data=pi)

    grid=out.create_group("grid")
    grid.attrs.create("step", data=step)
    grid.attrs.create("time", data=timeIn)
    grid.create_dataset("r", data=r)
    grid.create_dataset("th", data=th)
    grid.create_dataset("z", data=z)

    setup=out.create_group("setup")
    setup.attrs.create("Re", data=Re)

    velocity=out.create_group("fields/velocity")
    velocity.create_dataset("u_rF", data=np.array(u_rF).transpose(0,2,1))
    velocity.create_dataset("u_thF",data=np.array(u_thF).transpose(0,2,1))
    velocity.create_dataset("u_zF", data=np.array(u_zF).transpose(0,2,1))

    out.close()
    #-----------------------------------------------------------------------------------
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

    attributeU_rF=ET.SubElement(grid, "Attribute")
    attributeU_rF.set("Name", "u_rF")
    attributeU_rF.set("AttributeType", "Scalar")
    attributeU_rF.set("Center","Node")
    dataItemU_rF=ET.SubElement(attributeU_rF, "DataItem")
    dataItemU_rF.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemU_rF.set("NumberType", "Float")
    dataItemU_rF.set("Precision", "8")
    dataItemU_rF.set("Format", "HDF")
    dataItemU_rF.text = fnam+":/fields/velocity/u_rF"

    attributeU_thF=ET.SubElement(grid, "Attribute")
    attributeU_thF.set("Name", "u_thF")
    attributeU_thF.set("AttributeType", "Scalar")
    attributeU_thF.set("Center","Node")
    dataItemU_thF=ET.SubElement(attributeU_thF, "DataItem")
    dataItemU_thF.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemU_thF.set("NumberType", "Float")
    dataItemU_thF.set("Precision", "8")
    dataItemU_thF.set("Format", "HDF")
    dataItemU_thF.text = fnam+":/fields/velocity/u_thF"

    attributeU_zF=ET.SubElement(grid, "Attribute")
    attributeU_zF.set("Name", "u_zF")
    attributeU_zF.set("AttributeType", "Scalar")
    attributeU_zF.set("Center","Node")
    dataItemU_zF=ET.SubElement(attributeU_zF, "DataItem")
    dataItemU_zF.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemU_zF.set("NumberType", "Float")
    dataItemU_zF.set("Precision", "8")
    dataItemU_zF.set("Format", "HDF")
    dataItemU_zF.text = fnam+":/fields/velocity/u_zF"

    attributeVelocity=ET.SubElement(grid, "Attribute")
    attributeVelocity.set("Name", "velocity")
    attributeVelocity.set("AttributeType", "Vector")
    attributeVelocity.set("Center","Node")

    dataItemVelocityFunction=ET.SubElement(attributeVelocity, "DataItem")
    dataItemVelocityFunction.set("ItemType", "Function")
    dataItemVelocityFunction.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th))+" 3")
    dataItemVelocityFunction.set("Function","JOIN($0, $1, $2)")

    attributeVelocity.set("Center","Node")

    dataItemVelocityR=ET.SubElement(dataItemVelocityFunction, "DataItem")
    dataItemVelocityR.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemVelocityR.set("NumberType", "Float")
    dataItemVelocityR.set("Precision", "8")
    dataItemVelocityR.set("Format", "HDF")
    dataItemVelocityR.text = fnam+":/fields/velocity/u_rF"

    dataItemVelocityTh=ET.SubElement(dataItemVelocityFunction, "DataItem")
    dataItemVelocityTh.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemVelocityTh.set("NumberType", "Float")
    dataItemVelocityTh.set("Precision", "8")
    dataItemVelocityTh.set("Format", "HDF")
    dataItemVelocityTh.text = fnam+":/fields/velocity/u_thF"

    dataItemVelocityZ=ET.SubElement(dataItemVelocityFunction, "DataItem")
    dataItemVelocityZ.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemVelocityZ.set("NumberType", "Float")
    dataItemVelocityZ.set("Precision", "8")
    dataItemVelocityZ.set("Format", "HDF")
    dataItemVelocityZ.text = fnam+":/fields/velocity/u_zF"


  
    fnam = 'filteredfieldFourier_pipe0002_'+'{:08d}'.format(iFile)+'.xmf'
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


