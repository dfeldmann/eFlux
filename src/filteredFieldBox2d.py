#!/usr/bin/env python3
# Purpose:  Read full 3d pipe flow velocity fields from one or more HDF5 files.
#           Read the computational grid in cylindrical co-ordinates from one of
#           these files. Read a statistically converged mean velocity profile
#           from a separate ascii file and subtract it from the each snapshot to
#           obtain the fluctuating velocity field (u'_z). Define filter a scale
#           and compute the filtered flow field (u'_rF, u'_thF, u'_zF, p'F) for
#           each individual snapshot based on a two-dimensional spatial box
#           filter operation in wall-parallel planes for each radial location.
#           Write the resulting full 3d filtered flow field and the unfiltered
#           axial velocity fluctuation field (u'_z) to individual HDF5 files for
#           each snapshot. Also, write a corresponding XDMF meta data file for
#           easy data handling in visualisation software like ParaView and such.
# Usage:    python filteredFieldBox2d.py
# Authors:  Daniel Feldmann, Mohammad Umair, Jan Chen
# Date:     28th March 2019
# Modified: 01st October 2019

import timeit
import numpy as np
import h5py

# range of state files to read flow field data
iFirst =  1675000 # 570000
iLast  =  1675000
iStep  =     5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('Compute filtered flow field (box) for', len(iFiles), 'snapshot(s):', iFiles[0], 'to', iFiles[-1])

# path to data files (do modify)
fpath = '../../outFiles/'

# read grid from first HDF5 file
fnam = fpath+'field_pipe0002_'+'{:08d}'.format(iFirst)+'.h5'
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
    fnam = fpath+'field_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
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
    u_z  = u_z - np.tile(u_zM, (nz, nth, 1)).T

    # filter velocity field
    print('Filtering velocity components and pressure... ', end='', flush=True)
    t1 = timeit.default_timer()
    import filter2d as f2
    u_rF  = f2.box2d(u_r,  lambdaTh, lambdaZ, r, th, z)
    u_thF = f2.box2d(u_th, lambdaTh, lambdaZ, r, th, z)
    u_zF  = f2.box2d(u_z,  lambdaTh, lambdaZ, r, th, z)
    pF    = f2.box2d(p,    lambdaTh, lambdaZ, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t1), 'seconds')

    # store result as individual HDF5 file
    fnam = 'filteredFieldBox2d_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    out = h5py.File(fnam, 'w') # open HDF5 file for writing
    fields = out.create_group("fields")
    scale = out.create_group("scale")
    scale.attrs.create("deltaTh", data=lambdaTh)
    scale.attrs.create("deltaZ", data=lambdaZ)
    grid = out.create_group("grid")
    grid.create_dataset("r", data=r)
    grid.create_dataset("th", data=th)
    grid.create_dataset("z", data=z)
    grid.attrs.create("step", data=step)
    grid.attrs.create("time", data=timeIn)
    setup = out.create_group("setup")
    setup.attrs.create("Re", data=Re)
    velocity = out.create_group("fields/velocity")
    velocity.create_dataset("u_rF",  data=np.array(u_rF).transpose(0,2,1))
    velocity.create_dataset("u_thF", data=np.array(u_thF).transpose(0,2,1))
    velocity.create_dataset("u_zF",  data=np.array(u_zF).transpose(0,2,1))
    velocity.create_dataset("u_z",   data=np.array(u_z).transpose(0,2,1))
    pressure = out.create_group("fields/pressure")
    pressure.create_dataset("pF", data=np.array(pF).transpose(0,2,1))
    out.close() # close HDF5 file
    print('Written file:', fnam)


    # write corresponding XDMF meta data file
    # for further info visit
    # https://pymotw.com/2/xml/etree/ElementTree/create.html
    # http://www.xdmf.org/index.php/XDMF_Model_and_Format
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
    topology.set("Dimensions",str(nr)+" "+str(nz)+" "+str(nth))

    grid.extend(topology)
    grid.set("Name", "mesh")
    grid.set("GridType", "Uniform")

    geometry=ET.SubElement(grid, "Geometry")
    geometry.set("GeometryType","VXVYVZ")

    dataItemTh=ET.SubElement(geometry, "DataItem")
    dataItemTh.set("Dimensions",str(nth))
    dataItemTh.set("Name", "th")
    dataItemTh.set("NumberType", "Float")
    dataItemTh.set("Precision", "8")
    dataItemTh.set("Format", "HDF")
    dataItemTh.text = fnam+":/grid/th"

    dataItemZ=ET.SubElement(geometry, "DataItem")
    dataItemZ.set("Dimensions",str(nz))
    dataItemZ.set("Name", "z")
    dataItemZ.set("NumberType", "Float")
    dataItemZ.set("Precision", "8")
    dataItemZ.set("Format", "HDF")
    dataItemZ.text = fnam+":/grid/z"

    dataItemR=ET.SubElement(geometry, "DataItem")
    dataItemR.set("Dimensions",str(nr))
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
    dataItemU_rF.set("Dimensions",str(nr)+" "+str(nz)+" "+str(nth))
    dataItemU_rF.set("NumberType", "Float")
    dataItemU_rF.set("Precision", "8")
    dataItemU_rF.set("Format", "HDF")
    dataItemU_rF.text = fnam+":/fields/velocity/u_rF"

    attributeU_thF=ET.SubElement(grid, "Attribute")
    attributeU_thF.set("Name", "u_thF")
    attributeU_thF.set("AttributeType", "Scalar")
    attributeU_thF.set("Center","Node")
    dataItemU_thF=ET.SubElement(attributeU_thF, "DataItem")
    dataItemU_thF.set("Dimensions",str(nr)+" "+str(nz)+" "+str(nth))
    dataItemU_thF.set("NumberType", "Float")
    dataItemU_thF.set("Precision", "8")
    dataItemU_thF.set("Format", "HDF")
    dataItemU_thF.text = fnam+":/fields/velocity/u_thF"

    attributeU_zF=ET.SubElement(grid, "Attribute")
    attributeU_zF.set("Name", "u_zF")
    attributeU_zF.set("AttributeType", "Scalar")
    attributeU_zF.set("Center","Node")
    dataItemU_zF=ET.SubElement(attributeU_zF, "DataItem")
    dataItemU_zF.set("Dimensions",str(nr)+" "+str(nz)+" "+str(nth))
    dataItemU_zF.set("NumberType", "Float")
    dataItemU_zF.set("Precision", "8")
    dataItemU_zF.set("Format", "HDF")
    dataItemU_zF.text = fnam+":/fields/velocity/u_zF"

    attributeU_z=ET.SubElement(grid, "Attribute")
    attributeU_z.set("Name", "u_z")
    attributeU_z.set("AttributeType", "Scalar")
    attributeU_z.set("Center","Node")
    dataItemU_z=ET.SubElement(attributeU_z, "DataItem")
    dataItemU_z.set("Dimensions",str(nr)+" "+str(nz)+" "+str(nth))
    dataItemU_z.set("NumberType", "Float")
    dataItemU_z.set("Precision", "8")
    dataItemU_z.set("Format", "HDF")
    dataItemU_z.text = fnam+":/fields/velocity/u_z"

    attributePF=ET.SubElement(grid, "Attribute")
    attributePF.set("Name", "pF")
    attributePF.set("AttributeType", "Scalar")
    attributePF.set("Center","Node")
    dataItemPF=ET.SubElement(attributePF, "DataItem")
    dataItemPF.set("Dimensions",str(nr)+" "+str(nz)+" "+str(nth))
    dataItemPF.set("NumberType", "Float")
    dataItemPF.set("Precision", "8")
    dataItemPF.set("Format", "HDF")
    dataItemPF.text = fnam+":/fields/pressure/pF"

    attributeVelocityF=ET.SubElement(grid, "Attribute")
    attributeVelocityF.set("Name", "velocity")
    attributeVelocityF.set("AttributeType", "Vector")
    attributeVelocityF.set("Center","Node")

    dataItemVelocityFFunction=ET.SubElement(attributeVelocityF, "DataItem")
    dataItemVelocityFFunction.set("ItemType", "Function")
    dataItemVelocityFFunction.set("Dimensions",str(nr)+" "+str(nz)+" "+str(nth)+" 3")
    dataItemVelocityFFunction.set("Function","JOIN($0, $1, $2)")

    attributeVelocityF.set("Center","Node")

    dataItemVelocityFR=ET.SubElement(dataItemVelocityFFunction, "DataItem")
    dataItemVelocityFR.set("Dimensions",str(nr)+" "+str(nz)+" "+str(nth))
    dataItemVelocityFR.set("NumberType", "Float")
    dataItemVelocityFR.set("Precision", "8")
    dataItemVelocityFR.set("Format", "HDF")
    dataItemVelocityFR.text = fnam+":/fields/velocity/u_rF"

    dataItemVelocityFTh=ET.SubElement(dataItemVelocityFFunction, "DataItem")
    dataItemVelocityFTh.set("Dimensions",str(nr)+" "+str(nz)+" "+str(nth))
    dataItemVelocityFTh.set("NumberType", "Float")
    dataItemVelocityFTh.set("Precision", "8")
    dataItemVelocityFTh.set("Format", "HDF")
    dataItemVelocityFTh.text = fnam+":/fields/velocity/u_thF"

    dataItemVelocityFZ=ET.SubElement(dataItemVelocityFFunction, "DataItem")
    dataItemVelocityFZ.set("Dimensions",str(nr)+" "+str(nz)+" "+str(nth))
    dataItemVelocityFZ.set("NumberType", "Float")
    dataItemVelocityFZ.set("Precision", "8")
    dataItemVelocityFZ.set("Format", "HDF")
    dataItemVelocityFZ.text = fnam+":/fields/velocity/u_zF"

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