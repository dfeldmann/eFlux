#!/usr/bin/env python3
# Purpose:  Read full 3d pipe flow velocity fields from one or more HDF5 files.
#           Read the computational grid in cylindrical co-ordinates from one of
#           these files. Read a statistically converged mean velocity profile
#           from a separate ascii file and subtract it from the each snapshot to
#           obtain the fluctuating velocity field. Define filter widths and
#           compute the inter-scale turbulent kinetic energy flux field for each
#           individual snapshot based on a two-dimensional spatial filter
#           operation in wall-parallel planes for each radial location. Write
#           the resulting full 3d energy flux fields to individual HDF5 files
#           for each snapshot.
# Usage:    python piFieldGauss2d.py
# Authors:  Daniel Feldmann, Mohammad Umair, Jan Chen
# Date:     28th March 2019
# Modified: 16th August 2019

import sys
import os.path
import timeit
import math
import numpy as np
import h5py

# range of state files to read flow field data
iFirst =  1675000
iLast  =  1675000
iStep  =     5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('Compute instantaneous energy flux for', len(iFiles), 'snapshot(s):', iFiles[0], 'to', iFiles[-1])

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

# define filter width for each direction seperately
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

# reset wall-clock time
t0 = timeit.default_timer()

# loop over all state files
for iFile in iFiles:
    
    # read flow field data from next HDF5 file
    fnam = fpath+'fields_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
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
    
    # subtract mean velocity profile (1d) to obtain full (3d) fluctuating velocity field
    u_z  = u_z - np.tile(u_zM, (len(z), len(th), 1)).T
    
    # filter velocity field
    print('Filtering velocity components and mixed terms... ', end='', flush=True)
    t1 = timeit.default_timer()
    import filter2d as f2
    u_rF    = f2.gauss2d(u_r,       lambdaTh, lambdaZ, r, th, z)
    u_thF   = f2.gauss2d(u_th,      lambdaTh, lambdaZ, r, th, z)
    u_zF    = f2.gauss2d(u_z,       lambdaTh, lambdaZ, r, th, z)
    u_rRF   = f2.gauss2d(u_r*u_r,   lambdaTh, lambdaZ, r, th, z)
    u_rThF  = f2.gauss2d(u_r*u_th,  lambdaTh, lambdaZ, r, th, z)
    u_rZF   = f2.gauss2d(u_r*u_z,   lambdaTh, lambdaZ, r, th, z)
    u_thThF = f2.gauss2d(u_th*u_th, lambdaTh, lambdaZ, r, th, z)
    u_thZF  = f2.gauss2d(u_th*u_z,  lambdaTh, lambdaZ, r, th, z)
    u_zZF   = f2.gauss2d(u_z*u_z,   lambdaTh, lambdaZ, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t1), 'seconds')

    # compute instantaneous energy flux
    t2 = timeit.default_timer()
    print('Computing energy flux... ', end='', flush=True)
    import eFlux
    pi = np.array(eFlux.eFlux(u_rF, u_thF, u_zF, u_rRF, u_rThF, u_rZF, u_thThF, u_thZF, u_zZF, r, th, z)).transpose(0,2,1)
    #pi = np.zeros(u_z.shape)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t2), 'seconds')

    # store result as individual HDF5 file    
    fnam = 'piFieldGauss2d_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    out = h5py.File(fnam, 'w') # open HDF5 file for writing
    fields = out.create_group("fields")
    scale  = out.create_group("scale")
    scale.attrs.create("lambdaTh", data=lambdaTh)
    scale.attrs.create("lambdaZ", data=lambdaZ)
    eFlux = out.create_group("fields/pi")
    eFlux.create_dataset("pi", data=pi)
    grid = out.create_group("grid")
    grid.create_dataset("r", data=r)
    grid.create_dataset("th", data=th)
    grid.create_dataset("z", data=z)
    grid.attrs.create("step", data=step)
    grid.attrs.create("time", data=timeIn)
    setup = out.create_group("setup")
    setup.attrs.create("Re", data=Re)
    out.close() # close HDF5 file
    print('Written file:', fnam)
    
    # write corresponding XDMF meta data file
    # for further info visit
    # https://pymotw.com/2/xml/etree/ElementTree/create.html 
    # http://www.xdmf.org/index.php/XDMF_Model_and_Format
    import xml.etree.cElementTree as ET
    from xml.dom import minidom

    def prettify(elem):
        # Return a pretty-printed XML string for the Element
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    root = ET.Element(" ")
    root.append(ET.Comment('DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []'))

    xdmf = ET.Element('Xdmf')
    xdmf.set("version", "2.0")
    domain=ET.SubElement(xdmf, "Domain")
    grid=ET.SubElement(domain, "Grid") #, {"Topology":"3DRectMesh"})

    topology=ET.SubElement(grid,'Topology') #('''<root><Topology ToplogyType="3DRectMesh" /></root>''')#ET.SubElement
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
  
    # create corresponding file name by replacing file suffix
    fnam = str.replace(fnam, '.h5', '.xmf')
    with open(fnam, "w+") as f:
        print(prettify(xdmf), file=f)
        print

    # add declaration, workaround to ET    
    declaration='<!DOCTYPE Xdmf SYSTEM "xdmf.dtd" []>\n'
    f = open(fnam, "r")
    contents = f.readlines()
    f.close()

    contents.insert(1, declaration)

    f = open(fnam, "w")
    contents = "".join(contents)
    f.write(contents)
    f.close()
    print('Written file:', fnam)

print('Done!')
