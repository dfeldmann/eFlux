#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------
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
# import scipy as sc
from mpl_toolkits.axes_grid1 import make_axes_locatable

#--------------------------------------------------
# range of state files to read from flow field data
iFirst = 875000
iLast  = 875000
iStep  =   5000
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

#------------------------------------------------
# parallel stuff
import multiprocessing
from joblib import Parallel, delayed
print('=============================================================================================')
print("Running on", multiprocessing.cpu_count(), "cores")
print('=============================================================================================')


for iFile in iFiles:

    # read flow field data from next hdf5 file
    fnam = '../outFiles/fields_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    f = h5py.File(fnam, 'r')
    print("Reading velocity field from file", fnam, end='', flush=True)
    Ur  = np.array(f['fields/velocity/u_r']).transpose(0,2,1)  # do moidify transpose command to match data structures
    Uth = np.array(f['fields/velocity/u_th']).transpose(0,2,1) # openpipeflow u[r,th,z] and nsCouette/nsPipe u[r,th,z]
    Uz  = np.array(f['fields/velocity/u_z']).transpose(0,2,1)  # filter functions were made for u[r,th,z]
    step = f['grid'].attrs.__getitem__('step')
    timeIn = f['grid'].attrs.__getitem__('time')
    Re   = f['setup'].attrs.__getitem__('Re')
    f.close()
    print(' with data structure u', Uz.shape)

#    uz  = Uz - np.tile(UzM, (len(z), len(th), 1)).T

    # calculating vorticity field
    print('-------------------------------------------------------------------------')
    print('Calculating Voriticity Field... ', end='', flush=True)
    t1 = timeit.default_timer()
    #------------------------------------------------------------
    # indices: indicates  co-ordinate direction: 1=r, 2=theta, 3=z
    vort  = np.zeros((Ur.shape))             # constructing an array of dimension(r) filled with zeros 
    r3d = np.tile(r, (len(z), len(th), 1)).T # Changing a 1D array to a 3D array as we have to divide u(:,:,:) by r. 
                                             # In python we have to reshape our array to 3D to perform the division.
    # increment for spatial derivatives
    dth = th[1] - th[0]
    dz  =  z[1] -  z[0]

    # calculating velocity gradients
    dUthdr = np.gradient(Uth,r, axis=0)
    dUzdr  = np.gradient(Ur,dth,axis=0)

    dUrdth = (1.0/r3d)*np.gradient(Ur,dth,axis=1) - Uth/r3d
    dUzdth = (1.0/r3d)*np.gradient(Uz,dth,axis=1)

    dUrdz  = np.gradient(Ur ,dz,axis=2)
    dUthdz = np.gradient(Uth,dz,axis=2)

    #calculating components of vorticity vector
    omgR  = np.array(dUzdth - dUthdz).transpose(0,2,1)
    omgTh = np.array(dUrdz  - dUzdr ).transpose(0,2,1)
    omgZ  = np.array(dUthdr - dUrdth).transpose(0,2,1)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t1), 'seconds')
    print('-------------------------------------------------------------------------')

#===================================================================================
    fnam = 'vort_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
   # fnam1 = 'vorticityField/vort_pipe0003_'+'{:08d}'.format(iFile)+'.h5' 
    #pathOut="vorticityField/"
    #filenameOut='vort'+fnam[1:len(fnam)-5]

    #out = h5py.File(pathOut+filenameOut+".h5", 'w')
    out = h5py.File(fnam, 'w')
    fields=out.create_group("fields")

    vorticity=out.create_group("fields/vorticity")
    vorticity.create_dataset("omgR" , data=omgR )
    vorticity.create_dataset("omgTh", data=omgTh)
    vorticity.create_dataset("omgZ" , data=omgZ )

    grid=out.create_group("grid")
    grid.attrs.create("step", data=step)
    grid.attrs.create("time", data=timeIn)
    grid.create_dataset("r", data=r)
    grid.create_dataset("th", data=th)
    grid.create_dataset("z", data=z)
    setup=out.create_group("setup")
    setup.attrs.create("Re", data=Re)
    out.close()

#===================================================================================
#% write XMF file
#for further info visit https://pymotw.com/2/xml/etree/ElementTree/create.html 
#& http://www.xdmf.org/index.php/XDMF_Model_and_Format
    import xml.etree.cElementTree as ET
    from xml.dom import minidom
    

    def prettify(elem):
     """Return a pretty-printed XML string for the Element.
     """
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

    attributeomgR=ET.SubElement(grid, "Attribute")
    attributeomgR.set("Name", "omgR")
    attributeomgR.set("AttributeType", "Scalar")
    attributeomgR.set("Center","Node")

    dataItemomgR=ET.SubElement(attributeomgR, "DataItem")
    dataItemomgR.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemomgR.set("NumberType", "Float")
    dataItemomgR.set("Precision", "8")
    dataItemomgR.set("Format", "HDF")
    dataItemomgR.text = fnam+":/fields/vorticity/omgR"

    attributeomgTh=ET.SubElement(grid, "Attribute")
    attributeomgTh.set("Name", "omgTh")
    attributeomgTh.set("AttributeType", "Scalar")
    attributeomgTh.set("Center","Node")

    dataItemomgTh=ET.SubElement(attributeomgTh, "DataItem")
    dataItemomgTh.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemomgTh.set("NumberType", "Float")
    dataItemomgTh.set("Precision", "8")
    dataItemomgTh.set("Format", "HDF")
    dataItemomgTh.text = fnam+":/fields/vorticity/omgTh"

    attributeomgZ=ET.SubElement(grid, "Attribute")
    attributeomgZ.set("Name", "omgZ")
    attributeomgZ.set("AttributeType", "Scalar")
    attributeomgZ.set("Center","Node")

    dataItemomgZ=ET.SubElement(attributeomgZ, "DataItem")
    dataItemomgZ.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemomgZ.set("NumberType", "Float")
    dataItemomgZ.set("Precision", "8")
    dataItemomgZ.set("Format", "HDF")
    dataItemomgZ.text = fnam+":/fields/vorticity/omgZ"

    attributeVorticity=ET.SubElement(grid, "Attribute")
    attributeVorticity.set("Name", "Vorticity")
    attributeVorticity.set("AttributeType", "Vector")
    attributeVorticity.set("Center","Node")

    dataItemVorticityFunction=ET.SubElement(attributeVorticity, "DataItem")
    dataItemVorticityFunction.set("ItemType", "Function")
    dataItemVorticityFunction.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th))+" 3")
    dataItemVorticityFunction.set("Function","JOIN($0, $1, $2)")

    attributeVorticity.set("Center","Node")
    dataItemVorticityR=ET.SubElement(dataItemVorticityFunction, "DataItem")
    dataItemVorticityR.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemVorticityR.set("NumberType", "Float")
    dataItemVorticityR.set("Precision", "8")
    dataItemVorticityR.set("Format", "HDF")
    dataItemVorticityR.text = fnam+":/fields/vorticity/omgR"

    dataItemVorticityTh=ET.SubElement(dataItemVorticityFunction, "DataItem")
    dataItemVorticityTh.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemVorticityTh.set("NumberType", "Float")
    dataItemVorticityTh.set("Precision", "8")
    dataItemVorticityTh.set("Format", "HDF")
    dataItemVorticityTh.text = fnam+":/fields/vorticity/omgTh"

    dataItemVorticityZ=ET.SubElement(dataItemVorticityFunction, "DataItem")
    dataItemVorticityZ.set("Dimensions",str(len(r))+" "+str(len(z))+" "+str(len(th)))
    dataItemVorticityZ.set("NumberType", "Float")
    dataItemVorticityZ.set("Precision", "8")
    dataItemVorticityZ.set("Format", "HDF")
    dataItemVorticityZ.text = fnam+":/fields/vorticity/omgZ"

    fnam = 'vort_pipe0002_'+'{:08d}'.format(iFile)+'.xmf'

#ET.write(xml_declaration="false")
    #with open(pathOut+filenameOut+".xmf", "w+") as f:
    with open(fnam, "w+") as f:
        #f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []> \n')
        print(prettify(xdmf), file=f)
        print

#add Declatation, workaround to ET    
    declaration='<!DOCTYPE Xdmf SYSTEM "xdmf.dtd" []>\n'
    #f = open(pathOut+filenameOut+".xmf", "r")
    f = open(fnam, "r")
    contents = f.readlines()
    f.close()

    contents.insert(1, declaration)

    f = open(fnam, "w")
    contents = "".join(contents)
    f.write(contents)
    f.close()

