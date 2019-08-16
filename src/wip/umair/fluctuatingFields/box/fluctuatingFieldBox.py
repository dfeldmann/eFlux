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
fnam = '../../statistics00570000to00875000nt0062.dat'

print('-----------------------------------------------------------------')
print('Reading mean velocity profiles from', fnam)
u_zM = np.loadtxt(fnam)[:, 3]
u_zR = np.loadtxt(fnam)[:, 7]
#--------------------------------------------------
# parallel stuff
import multiprocessing
from joblib import Parallel, delayed
print('=============================================================================================')
print("Running on", multiprocessing.cpu_count(), "cores")
print('=============================================================================================')

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
    u_rF    = u_r
    u_thF   = u_th
    u_zF    = u_z
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t1), 'seconds')
    #-------------------------------------------------------------------------------    

    print('-------------------------------------------------------------------------')

    #-----------------------------------------------------------------------------------
    fnam = 'filteredfieldBox_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    out = h5py.File(fnam, 'w')

    fields=out.create_group("fields")

    scale=out.create_group("scale")
    scale.attrs.create("deltaR", data=lambdaR)
    scale.attrs.create("deltaTh", data=lambdaTh)
    scale.attrs.create("deltaZ", data=lambdaZ)

    grid=out.create_group("grid")
    grid.attrs.create("step", data=step)
    grid.attrs.create("time", data=timeIn)
    grid.create_dataset("r", data=r)
    grid.create_dataset("th", data=th)
    grid.create_dataset("z", data=z)

    setup=out.create_group("setup")
    setup.attrs.create("Re", data=Re)

    velocity=out.create_group("fields/velocity")
    velocity.create_dataset("u_rF",  data=np.array(u_rF).transpose(0,2,1))
    velocity.create_dataset("u_thF", data=np.array(u_thF).transpose(0,2,1))
    velocity.create_dataset("u_zF",  data=np.array(u_zF).transpose(0,2,1))

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


  
    fnam = 'filteredfieldBox_pipe0002_'+'{:08d}'.format(iFile)+'.xmf'
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


