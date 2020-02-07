#!/usr/bin/env python3
# Purpose:  Read full 3d pipe flow velocity fields from one or more HDF5 files.
#           Read the computational grid in cylindrical co-ordinates from one of
#           these files. Read a statistically converged mean velocity profile
#           from a separate ascii file and subtract it from each snapshot to
#           obtain the fluctuating velocity field. Define the filter widths and
#           compute the inter-scale turbulent kinetic energy flux field (Pi) for
#           each individual snapshot based on a two-dimensional spatial Gauss
#           filter operation in wall-parallel planes for each radial location.
#           Additionally, extract Q events from the velocity field for each
#           snapshot. Finally, compute two-dimensional two-point correlation
#           maps in a selected wall-parallel (theta-z) plane; auto-correlations
#           for the Q events (representing important features of the near-wall
#           cycle) and for the energy flux, cross-correlations for all of the Q
#           events with the energy flux. Do statistics over all snapshots, and
#           write the resulting 2d correlation maps to a single ascii file.
#           TODO: At some point one could compute correlation maps for ALL
#           wall-parallel planes and save the resulting 3d fields in one single
#           h5 file...
# Usage:    python piCorrThZQsGauss2d.py
# Authors:  Daniel Feldmann, Mohammad Umair, Jan Chen
# Date:     28th March 2019
# Modified: 19th September 2019

import sys
import timeit
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
plot = 1

# range of state files to read flow field data (do modify)
iFirst =   570000
iLast  =  4070000 # 1675000
iStep  =    10000 #    5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('Compute eFlux (Gauss) and 2d correlation maps with Q events for', len(iFiles), 'snapshot(s):', iFiles[0], 'to', iFiles[-1])

# path to data files (do modify)
fpath = '../../outFiles/'

# read grid from first HDF5 file
fnam = fpath+'field_pipe0002_'+'{:08d}'.format(iFirst)+'.h5'
print('Reading grid from', fnam, 'with:')
f  = h5py.File(fnam, 'r')    # open hdf5 file for read only
r  = np.array(f['grid/r'])   # radial co-ordinate
th = np.array(f['grid/th'])  # azimuthal co-ordinate
z  = np.array(f['grid/z'])   # axial co-ordainte
f.close()                    # close hdf5 file

# report grid size
nr  = len(r)
nth = len(th)
nz  = len(z)
print(nr, 'radial (r) points')
print(nth, 'azimuthal (th) points')
print(nz, 'axial (z) points')

# read mean velocity profiles from ascii file (do modify)
fnam = '../../onePointStatistics/statistics00570000to04070000nt0351.dat' 
print('Reading mean velocity profile from', fnam)
rM   = np.loadtxt(fnam)[:, 0] # 1st column r
u_zM = np.loadtxt(fnam)[:, 3] # 4th column <u_z>

# define filter width for each direction seperately (do modify)
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

# prepare arrays for statistics
acQ1   = np.zeros((nth, nz))  # initialise auto-correlation for Q1 outward interactions
acQ2   = np.zeros((nth, nz))  # initialise auto-correlation for Q2 ejection events
acQ3   = np.zeros((nth, nz))  # initialise auto-correlation for Q3 inward interactions
acQ4   = np.zeros((nth, nz))  # initialise auto-correlation for Q4 sweep events
acPi   = np.zeros((nth, nz))  # initialise auto-correlation for Pi
ccQ1Pi = np.zeros((nth, nz))  # initialise cross-correlation for Q1 eFlux
ccQ2Pi = np.zeros((nth, nz))  # initialise cross-correlation for Q2 eFlux
ccQ3Pi = np.zeros((nth, nz))  # initialise cross-correlation for Q3 eFlux
ccQ4Pi = np.zeros((nth, nz))  # initialise cross-correlation for Q4 eFlux
nt     = 0                    # reset ensemble counter

# first and second statistical moments for normalisation
q11 = 0
q12 = 0
q21 = 0
q22 = 0
q31 = 0
q32 = 0
q41 = 0
q42 = 0
pi1 = 0
pi2 = 0

# reset wall-clock time
t0 = timeit.default_timer()

# loop over all state files
for iFile in iFiles:
    
    # read flow field data from next HDF5 file
    fnam = fpath+'field_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
    f = h5py.File(fnam, 'r')
    print("Reading velocity field from file", fnam, end='', flush=True)
    u_r  = np.array(f['fields/velocity/u_r']).transpose(0,2,1)  # do modify transpose to match data structures
    u_th = np.array(f['fields/velocity/u_th']).transpose(0,2,1) # filter functions were made for u[r, th, z]
    u_z  = np.array(f['fields/velocity/u_z']).transpose(0,2,1)
    step=f['grid'].attrs.__getitem__('step')
    timeIn=f['grid'].attrs.__getitem__('time')
    Re=f['setup'].attrs.__getitem__('Re')
    f.close()
    print(' with data structure u',u_z.shape)
    
    # subtract mean velocity profile (1d) to obtain full (3d) fluctuating velocity field
    u_z = u_z - np.tile(u_zM, (nz, nth, 1)).T

    # detect and extrct Q events from the instantaneous velocity vector field
    # TODO: compute correlation maps for ALL wall-parallel planes
    #t1 = timeit.default_timer()
    #print('Extract Q events... ', end='', flush=True)
    #q1 = np.zeros((nr, nth, nz))  
    #q2 = np.zeros((nr, nth, nz))  
    #q3 = np.zeros((nr, nth, nz))  
    #q4 = np.zeros((nr, nth, nz))  
    #print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-t1), 'seconds')
    
    # filter velocity field
    print('Filtering velocity components and mixed terms... ', end='', flush=True)
    tfilter = timeit.default_timer()
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
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tfilter), 'seconds')

    # compute instantaneous energy flux
    tflux = timeit.default_timer()
    print('Computing energy flux... ', end='', flush=True)
    import eFlux
    pi = eFlux.eFlux(u_rF, u_thF, u_zF, u_rRF, u_rThF, u_rZF, u_thThF, u_thZF, u_zZF, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tflux), 'seconds')

    # extract 2d data sub-sets in a wall parallel plane
    k = 65 # (do modify)
    print("Extracting 2d data in wall-parallel plane at y+ =", (1-r[k])*ReTau)
    ur2d = u_r[k, :, :] # data structure is (r, theta, z)
    uz2d = u_z[k, :, :]
    pi2d =  pi[k, :, :]
    #q12d = q1[k, :, :] # TODO: compute correlation maps for ALL wall-parallel planes
    #q22d = q2[k, :, :]
    #q32d = q3[k, :, :]
    #q42d = q4[k, :, :]

    # detect and extract Q events from the 2d velocity sub-set
    tqs = timeit.default_timer()
    print("Extracting Q events from 2d velocity field...", end='', flush=True)
    q1 = np.zeros(ur2d.shape)
    q2 = np.zeros(ur2d.shape)
    q3 = np.zeros(ur2d.shape)
    q4 = np.zeros(ur2d.shape)
    for i in range(nz):
     for j in range(nth):
      if (uz2d[j,i]>0) and (ur2d[j,i]<0): q1[j,i] = ur2d[j,i]*uz2d[j,i] # outward interaction: high-speed fluid away from wall
      if (uz2d[j,i]<0) and (ur2d[j,i]<0): q2[j,i] = ur2d[j,i]*uz2d[j,i] # ejection event:       low-speed fluid away from wall
      if (uz2d[j,i]<0) and (ur2d[j,i]>0): q3[j,i] = ur2d[j,i]*uz2d[j,i] # inward interaction:   low-speed fluid towards   wall
      if (uz2d[j,i]>0) and (ur2d[j,i]>0): q4[j,i] = ur2d[j,i]*uz2d[j,i] # sweep event:         high-speed fluid towards   wall
    ioi = q1 - q3 # unify inward interactions (Q3 being negativ) and outward interactions (Q1 being positive) in one array
    see = q2 - q4 # unify sweep events (Q4 being negativ) and ejection events (Q2 being positiv) in one array
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tqs), 'seconds')

    # compute correlations and sum up temporal (ensemble) statistics
    tcorr = timeit.default_timer()
    print('Computing 2d correlations... ', end='', flush=True)
    import crossCorrelation as c 
    acQ1   = acQ1   + c.corr2d(q1,   q1)   # auto-correlations
    acQ2   = acQ2   + c.corr2d(q2,   q2)
    acQ3   = acQ3   + c.corr2d(q3,   q3)
    acQ4   = acQ4   + c.corr2d(q4,   q4)
    acPi   = acPi   + c.corr2d(pi2d, pi2d)
    ccQ1Pi = ccQ1Pi + c.corr2d(q1,   pi2d) # cross-correlations
    ccQ2Pi = ccQ2Pi + c.corr2d(q2,   pi2d)
    ccQ3Pi = ccQ3Pi + c.corr2d(q3,   pi2d)
    ccQ4Pi = ccQ4Pi + c.corr2d(q4,   pi2d)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tcorr), 'seconds')

    # sum up first and second statistical moments in time and (homogeneous) theta and z direction for normalisation
    q11 = q11 + np.sum(np.sum(q1,      axis=1), axis=0) # sum over all elements of 2d data sub set
    q12 = q12 + np.sum(np.sum(q1**2,   axis=1), axis=0)
    q21 = q21 + np.sum(np.sum(q2,      axis=1), axis=0)
    q22 = q22 + np.sum(np.sum(q2**2,   axis=1), axis=0)
    q31 = q31 + np.sum(np.sum(q3,      axis=1), axis=0)
    q32 = q32 + np.sum(np.sum(q3**2,   axis=1), axis=0)
    q41 = q41 + np.sum(np.sum(q4,      axis=1), axis=0)
    q42 = q42 + np.sum(np.sum(q4**2,   axis=1), axis=0)
    pi1 = pi1 + np.sum(np.sum(pi2d,    axis=1), axis=0)
    pi2 = pi2 + np.sum(np.sum(pi2d**2, axis=1), axis=0)

    # increase temporal/ensemble counter
    nt = nt + 1

# divide correlation statistics by total number of temporal samples
acQ1   = acQ1   / nt
acQ2   = acQ2   / nt
acQ3   = acQ3   / nt
acQ14  = acQ4   / nt
ccQ1Pi = ccQ1Pi / nt
ccQ2Pi = ccQ2Pi / nt
ccQ3Pi = ccQ3Pi / nt
ccQ4Pi = ccQ4Pi / nt

# divide normalisation statistics by total number of spatio-temporal samples
q11 = q11 / (nth*nz*nt)
q12 = q12 / (nth*nz*nt)
q21 = q21 / (nth*nz*nt)
q22 = q22 / (nth*nz*nt)
q31 = q31 / (nth*nz*nt)
q32 = q32 / (nth*nz*nt)
q41 = q41 / (nth*nz*nt)
q42 = q42 / (nth*nz*nt)
pi1 = pi1 / (nth*nz*nt)
pi2 = pi2 / (nth*nz*nt)

# compute RMS for normalisation
q1Rms = np.sqrt(q12 - q11**2)
q2Rms = np.sqrt(q22 - q21**2)
q3Rms = np.sqrt(q32 - q31**2)
q4Rms = np.sqrt(q42 - q41**2)
piRms = np.sqrt(pi2 - pi1**2)

# normalise correlations with local RMS 
acQ1   = acQ1   / (q1Rms*q1Rms)
acQ2   = acQ2   / (q2Rms*q2Rms)
acQ3   = acQ3   / (q3Rms*q3Rms)
acQ4   = acQ4   / (q4Rms*q4Rms)
acPi   = acPi   / (piRms*piRms)
ccQ1Pi = ccQ1Pi / (q1Rms*piRms)
ccQ2Pi = ccQ2Pi / (q2Rms*piRms)
ccQ3Pi = ccQ3Pi / (q3Rms*piRms)
ccQ4Pi = ccQ4Pi / (q4Rms*piRms)

print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')

# compute centered azimuthal and axial separation/displacement (for nice plotting only)
DeltaTh = (th - (th[-1] - th[0]) / 2.0) * r[k]
DeltaZ  =   z - ( z[-1] -  z[0]) / 2.0

# find and report absolute maxima of 2d data sets
amacQ1   = np.max(np.abs(acQ1))   # auto-correlation Q1 max
amacQ2   = np.max(np.abs(acQ2))   # auto-correlation Q2 max
amacQ3   = np.max(np.abs(acQ3))   # auto-correlation Q3 max
amacQ4   = np.max(np.abs(acQ4))   # auto-correlation Q4 max
amacPi   = np.max(np.abs(acPi))   # auto-correlation Pi max
amccQ1Pi = np.max(np.abs(ccQ1Pi)) # cross-correlation Q1 Pi max
amccQ2Pi = np.max(np.abs(ccQ2Pi)) # cross-correlation Q2 Pi max
amccQ3Pi = np.max(np.abs(ccQ3Pi)) # cross-correlation Q3 Pi max
amccQ4Pi = np.max(np.abs(ccQ4Pi)) # cross-correlation Q4 Pi max
print("Absolute maximum auto-correlation value  Q1 Q1:", amacQ1)
print("Absolute maximum auto-correlation value  Q2 Q2:", amacQ2)
print("Absolute maximum auto-correlation value  Q3 Q3:", amacQ3)
print("Absolute maximum auto-correlation value  Q4 Q4:", amacQ4)
print("Absolute maximum auto-correlation value  Pi Pi:", amacPi)
print("Absolute maximum cross-correlation value Q1 Pi:", amccQ1Pi)
print("Absolute maximum cross-correlation value Q2 Pi:", amccQ2Pi)
print("Absolute maximum cross-correlation value Q3 Pi:", amccQ3Pi)
print("Absolute maximum cross-correlation value Q4 Pi:", amccQ4Pi)

# write 2d correlation maps to ascii file
fnam = 'piCorrThZQsGauss2d_pipe0002_'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
f = open(fnam, 'w')
f.write("# Two-dimensional two-point correlation maps in a theta-z plane\n")
f.write("# At wall-normal location r = %f -> y+ = %f\n" % (r[k], (1.0-r[k])*ReTau))
f.write("# For the following flow variables:\n")
f.write("# + Q1 outward interactions (u'_z > 0 and u'_r < 0)\n")
f.write("# + Q2 ejection events      (u'_z < 0 and u'_r < 0)\n")
f.write("# + Q3 inward interactions  (u'_z < 0 and u'_r > 0)\n")
f.write("# + Q4 sweep events         (u'_z > 0 and u'_r > 0)\n")
f.write("# + Inter-scale energy flux Pi across scale lambda\n")
f.write("# Note that u'_r < 0 represents motion away from the wall in a cylindrical co-ordinate system)\n")
f.write("# Flux based on filtered quantities using a 2d Gauss kernel with:\n")
f.write("# + Azimuthal filter scale: lambdaTh+ = %f viscous units, lambdaTh = %f R\n" % (lambdaThp, lambdaTh))
f.write("# + Axial filter scale:     lambdaZ+  = %f viscous units, lambdaZ  = %f R\n" % (lambdaZp,  lambdaZ))
f.write("# Python post-processing on data set nsPipe/pipe0002 generated in a DNS using nsPipe\n")
f.write("# Temporal (ensemble) averaging over %d sample(s)\n" % (nt))
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# 01st column: Azimuthal separation DeltaTh in units of pipe radii (R), nth = %d points\n" % nth)
f.write("# 02nd column: Axial separation DeltaZ in units of pipe radii (R), nz = %d points\n" % nz)
f.write("# 03rd column: Auto-correlation  Q1 with Q1\n")
f.write("# 04th column: Auto-correlation  Q2 with Q2\n")
f.write("# 05th column: Auto-correlation  Q3 with Q3\n")
f.write("# 06th column: Auto-correlation  Q4 with Q4\n")
f.write("# 07th column: Auto-correlation  Pi with Pi\n")
f.write("# 08th column: Cross-correlation Q1 with Pi\n")
f.write("# 09th column: Cross-correlation Q2 with Pi\n")
f.write("# 10th column: Cross-correlation Q3 with Pi\n")
f.write("# 11th column: Cross-correlation Q4 with Pi\n")
for i in range(nth):
 for j in range(nz):
  f.write("%23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n" % (DeltaTh[i], DeltaZ[j], acQ1[i,j], acQ2[i,j], acQ3[i,j], acQ4[i,j], acPi[i,j], ccQ1Pi[i,j], ccQ2Pi[i,j], ccQ3Pi[i,j], ccQ4Pi[i,j]))
f.close()
print('Written 2d correlation maps to file', fnam)

# plotting
if plot not in [1, 2]: sys.exit() # skip everything below
print('Creating plot (using LaTeX)...')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
r"\usepackage[utf8]{inputenc}",
r"\usepackage[T1]{fontenc}",
r'\usepackage{lmodern, palatino, eulervm}',
#r'\usepackage{mathptmx}',
r"\usepackage[detect-all]{siunitx}",
r'\usepackage{amsmath, amstext, amssymb}',
r'\usepackage{xfrac}']
#mpl.rcParams.update({'font.family': 'sans-serif'})
mpl.rcParams.update({'font.family' : 'serif'})
mpl.rcParams.update({'font.size' : 8})
mpl.rcParams.update({'lines.linewidth'   : 0.75})
mpl.rcParams.update({'axes.linewidth'    : 0.75})
mpl.rcParams.update({'xtick.major.size'  : 2.00})
mpl.rcParams.update({'xtick.major.width' : 0.75})
mpl.rcParams.update({'xtick.minor.size'  : 1.00})
mpl.rcParams.update({'xtick.minor.width' : 0.75})
mpl.rcParams.update({'ytick.major.size'  : 2.00})
mpl.rcParams.update({'ytick.major.width' : 0.75})
mpl.rcParams.update({'ytick.minor.size'  : 1.00})
mpl.rcParams.update({'ytick.minor.width' : 0.75})

# create figure suitable for A4 format
def mm2inch(*tupl):
 inch = 25.4
 if isinstance(tupl[0], tuple):
  return tuple(i/inch for i in tupl[0])
 else:
   return tuple(i/inch for i in tupl)
#fig = plt.figure(num=None, figsize=mm2inch(134.0, 70.0), dpi=300) # , constrained_layout=False) 
fig = plt.figure(num=None, dpi=200) # , constrained_layout=False)

# conservative colour palette appropriate for colour-blind (http://mkweb.bcgsc.ca/colorblind/)
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Grey          = '#999999'
Black         = '#000000'
exec(open("./colourMaps.py").read()) # many thanks to github.com/nesanders/colorblind-colormap 
VermBlue = CBWcm['VeBu']             # from Vermillion (-) via White (0) to Blue (+)

# axes grid for multiple subplots with common colour bar
from mpl_toolkits.axes_grid1 import ImageGrid
ig = ImageGrid(fig, 111, nrows_ncols=(5, 2), direction='column', axes_pad=(0.6, 0.15), cbar_size=0.07, cbar_mode='each', cbar_location='right', cbar_pad=0.05)

# convert spatial separation from outer to inner units
DeltaTh = DeltaTh * ReTau
DeltaZ  = DeltaZ  * ReTau

# define sub-set for plotting (Here in plus units)
xmin = -200.0 # np.min(DeltaZ)
xmax =  200.0 # np.max(DeltaZ)
ymin = -100.0 # np.min(DeltaTh)
ymax =  100.0 # np.max(DeltaTh)

# plot auto-correlation Pi
ig[0].set_xlim(left=xmin, right=xmax)
ig[0].set_ylabel(r"$\Delta\theta r^{+}$")
ig[0].set_ylim(bottom=ymin, top=ymax)
im0 = ig[0].imshow(acPi, vmin=-amacPi, vmax=+amacPi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[0].set_aspect('equal')
fmt = FormatStrFormatter('%5.1f') # colourbar ticks format
cb0 = ig.cbar_axes[0].colorbar(im0, format=fmt)
cb0.ax.set_ylabel(r"$C_{\Pi\Pi}$")
cb0.ax.set_yticks([-1.0, 0.0, +1.0])

# plot auto-correlation Q1
ig[1].set_xlim(left=xmin, right=xmax)
ig[1].set_ylabel(r"$\Delta\theta r^{+}$")
ig[1].set_ylim(bottom=ymin, top=ymax)
im1 = ig[1].imshow(acQ1, vmin=-amacQ1, vmax=+amacQ1, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[1].set_aspect('equal')
cb1 = ig.cbar_axes[1].colorbar(im1, format=fmt)
cb1.ax.set_ylabel(r"$C_{Q_{1}Q_{1}}$")
cb1.ax.set_yticks([-1.0, 0.0, +1.0])

# plot auto-correlation Q2
ig[2].set_xlim(left=xmin, right=xmax)
ig[2].set_ylabel(r"$\Delta\theta r^{+}$")
ig[2].set_ylim(bottom=ymin, top=ymax)
im2 = ig[2].imshow(acQ2, vmin=-amacQ2, vmax=+amacQ2, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[2].set_aspect('equal')
cb2 = ig.cbar_axes[2].colorbar(im2, format=fmt)
cb2.ax.set_ylabel(r"$C_{Q_{2}Q_{2}}$")
cb2.ax.set_yticks([-1.0, 0.0, +1.0])
# TODO: remove empty colorbar

# plot auto-correlation Q3
ig[3].set_xlim(left=xmin, right=xmax)
ig[3].set_ylabel(r"$\Delta\theta r^{+}$")
ig[3].set_ylim(bottom=ymin, top=ymax)
im3 = ig[3].imshow(acQ3, vmin=-amacQ3, vmax=+amacQ3, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[3].set_aspect('equal')
cb3 = ig.cbar_axes[3].colorbar(im3, format=fmt)
cb3.ax.set_ylabel(r"$C_{Q_{3}Q_{3}}$")
cb3.ax.set_yticks([-1.0, 0.0, +1.0])

# plot auto-correlation Q4
ig[4].set_xlabel(r"$\Delta z^{+}$")
ig[4].set_xlim(left=xmin, right=xmax)
ig[4].set_ylabel(r"$\Delta\theta r^{+}$")
ig[4].set_ylim(bottom=ymin, top=ymax)
im4 = ig[4].imshow(acQ4, vmin=-amacQ4, vmax=+amacQ4, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[4].set_aspect('equal')
cb4 = ig.cbar_axes[4].colorbar(im4, format=fmt)
cb4.ax.set_ylabel(r"$C_{Q_{4}Q_{4}}$")
cb4.ax.set_yticks([-1.0, 0.0, +1.0])

# empty space for filter kernel label
filterBox = dict(boxstyle="square, pad=0.3", lw=0.5, fc='w', ec=Black)
ig[5].axis("off")
ig[5].text(0.0, 0.0, r"Gauss", ha="center", va="center", rotation=0, bbox=filterBox)

# plot cross-correlation Q1 Pi
ig[6].set_xlim(left=xmin, right=xmax)
ig[6].set_ylim(bottom=ymin, top=ymax)
im6 = ig[6].imshow(ccQ1Pi, vmin=-amccQ1Pi, vmax=+amccQ1Pi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[6].set_aspect('equal')
cb6 = ig.cbar_axes[6].colorbar(im6, format=fmt)
cb6.ax.set_ylabel(r"$C_{Q_{1}\Pi}$")
cb6.ax.set_yticks([-amccQ1Pi, 0.0, +amccQ1Pi])

# plot cross-correlation Q2 Pi
ig[7].set_xlim(left=xmin, right=xmax)
ig[7].set_ylim(bottom=ymin, top=ymax)
im7 = ig[7].imshow(ccQ2Pi, vmin=-amccQ2Pi, vmax=+amccQ2Pi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[7].set_aspect('equal')
cb7 = ig.cbar_axes[7].colorbar(im7, format=fmt)
cb7.ax.set_ylabel(r"$C_{Q_{2}\Pi}$")
cb7.ax.set_yticks([-amccQ2Pi, 0.0, +amccQ2Pi])

# plot cross-correlation Q3 Pi
ig[8].set_xlim(left=xmin, right=xmax)
ig[8].set_ylim(bottom=ymin, top=ymax)
im8 = ig[8].imshow(ccQ3Pi, vmin=-amccQ3Pi, vmax=+amccQ3Pi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[8].set_aspect('equal')
cb8 = ig.cbar_axes[8].colorbar(im8, format=fmt)
cb8.ax.set_ylabel(r"$C_{Q_{3}\Pi}$")
cb8.ax.set_yticks([-amccQ3Pi, 0.0, +amccQ3Pi])

# plot cross-correlation Q4 Pi
ig[9].set_xlabel(r"$\Delta z^{+}$")
ig[9].set_xlim(left=xmin, right=xmax)
ig[9].set_ylim(bottom=ymin, top=ymax)
im9 = ig[9].imshow(ccQ4Pi, vmin=-amccQ4Pi, vmax=+amccQ4Pi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[9].set_aspect('equal')
cb9 = ig.cbar_axes[9].colorbar(im9, format=fmt)
cb9.ax.set_ylabel(r"$C_{Q_{4}\Pi}$")
cb9.ax.set_yticks([-amccQ4Pi, 0.0, +amccQ4Pi])

# plot mode interactive or pdf
if plot != 2:
 #plt.tight_layout()
 plt.show()
else:
 #fig.tight_layout()
 fnam = str.replace(fnam, '.dat', '.pdf')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()

print('Done!')
