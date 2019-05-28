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
iFirst =  1005000
iLast  =  1005000
iStep  =  5000
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

#-------------------------------------------
# read mean velocity profiles from ascii file
fnam = '../statistics00570000to01005000nt0088.dat'
print('-----------------------------------------------------------------')
print('Reading mean velocity profiles from', fnam)
u_zM = np.loadtxt(fnam)[:, 3]
u_zR = np.loadtxt(fnam)[:, 7]
#--------------------------------------------------
print('=============================================================================================')
import multiprocessing
from joblib import Parallel, delayed
print('=============================================================================================')
print("Running on", multiprocessing.cpu_count(), "cores")
print('=============================================================================================')

#===================================================================================
# reset wall-clock time
t0 = timeit.default_timer()

# statistics loop over all state files
for iFile in iFiles:
    #-------------------------------------------------------------------------------
    # read flow field data from next hdf5 file
    fnam = '../outFiles/fields_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    f = h5py.File(fnam, 'r')
    print("Reading velocity field from file", fnam, end='', flush=True)
    u_r  = np.array(f['fields/velocity/u_r']).transpose(0,2,1)  # do moidify transpose command to match data structures
    u_th = np.array(f['fields/velocity/u_th']).transpose(0,2,1) # openpipeflow u[r,th,z] and nsCouette/nsPipe u[r,th,z]
    u_z  = np.array(f['fields/velocity/u_z']).transpose(0,2,1)  # filter functions were made for u[r,th,z]
    p    = np.array(f['fields/pressure']).transpose(0,2,1)      # filter functions were made for u[r,th,z]
    f.close()
    print(' with data structure u', u_z.shape)
    #-------------------------------------------------------------------------------
    # subtract mean velocity profiles (1d) from flow field (3d)
    u_z  = u_z - np.tile(u_zM, (len(z), len(th), 1)).T
    #-------------------------------------------------------------------------------
    # filter velocity field, single components and mixed terms
    print('-------------------------------------------------------------------------')
    t3 = timeit.default_timer()
    k = 63
    print ("Extracting Streaks from Wall-normal plane at y+ =", (1-r[k])*180.4)
    uZy15 = u_z[k, :, :]

print('-------------------------------------------------------------------------')
print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')
#=============================================================================================
# plot data as graph, (0) none, (1) interactive, (2) pdf
plot = 2
if plot not in [1, 2]: sys.exit() # skip everything below
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
r"\usepackage[utf8x]{inputenc}",
r"\usepackage[T1]{fontenc}",
r"\usepackage[detect-all]{siunitx}",
r'\usepackage{amsmath, amstext, amssymb}',
r'\usepackage{xfrac}',
r'\usepackage{lmodern, palatino, eulervm}']
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'font.size': 9})

# create figure suitable for A4 format
def mm2inch(*tupl):
 inch = 25.4
 if isinstance(tupl[0], tuple):
  return tuple(i/inch for i in tupl[0])
 else:
   return tuple(i/inch for i in tupl)
#fig = plt.figure(num=None, figsize=mm2inch(160.0, 140.0), dpi=150)
fig = plt.figure(num=None, figsize=mm2inch(160.0, 60.0), dpi=150)

# line colours appropriate for colour-blind
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Black         = '#000000'
#-----------------------------------------------------------------------------------

uZy15am  = np.max(np.abs(uZy15 ))

ncl = 600 # number of countur levels, increase to make plot smoother but larger
comap = 'RdGy'
#comap = 'PiYG'
# plot 

ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
#ax1.set_xlabel(r"${\Delta z}$ in $d$")
#ax1.set_ylabel(r"${r\Delta\theta}$ in $d$")
cl2 = np.linspace(-uZy15am, +uZy15am, ncl) # set contour levels manually
cf2 = ax1.contourf(z, th*r[k], uZy15, cl2,  cmap=comap)
ax1.set_aspect('equal') # makes axis ratio natural
#dvr = make_axes_locatable(ax1) # devider
#cbx = dvr.append_axes("bottom", size="5%", pad=0.3) # make colorbar axis
#cb1 = plt.colorbar(cf2, cax=cbx, ticks=[-uZy15am, 0, +uZy15am], orientation='horizontal') # set colorbar scale
#cbx.xaxis.set_ticks_position("bottom")
plt.tick_params(
    axis='both',        # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    top=False,
    right=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)  # labels along the bottom edge are off
plt.xlim(left=0,right=25)
#--------------------------------------------------------
# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = 'streaks2D'+'{:08d}'.format(iLast)+'.jpg'
 plt.savefig(fnam)
 print('Written file', fnam)
print('=============================================================================================')
fig.clf()





