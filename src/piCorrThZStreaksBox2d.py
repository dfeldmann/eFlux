#!/usr/bin/env python3
# Purpose:  Read full 3d pipe flow velocity fields from one or more HDF5 files.
#           Read the computational grid in cylindrical co-ordinates from one of
#           these files. Read a statistically converged mean velocity profile
#           from a separate ascii file and subtract it from each snapshot to
#           obtain the fluctuating velocity field. Define the filter widths and
#           compute the inter-scale turbulent kinetic energy flux field (Pi) for
#           each individual snapshot based on a two-dimensional spatial Box
#           filter operation in wall-parallel planes for each radial location.
#           Additionally, compute the vorticity vector field (omega) for each
#           snapshot. Finally, compute two-dimensional two-point correlation
#           maps in a selected wall-parallel (theta-z) plane; auto-correlations
#           for the original (u'_z) and filtered (u'_zF) streamwise velocity
#           component (representing streaks), for the axial vorticity component
#           (representing streamwise alligned vortices) and for the energy flux,
#           cross-correlations for all of the quantities with the energy flux.
#           Do statistics over all snapshots, and  write the resulting 2d
#           correlation maps to a single ascii file. TODO: At some point one
#           could compute correlation maps for ALL wall-parallel planes and save
#           the resulting 3d fields in one single h5 file...
# Usage:    python piCorrThZStreaksBox2d.py
# Authors:  Daniel Feldmann, Mohammad Umair, Jan Chen
# Date:     28th March 2019
# Modified: 30th December 2019

import sys
import timeit
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
plot = 0

# range of state files to read flow field data
iFirst =   570000
iLast  =  1675000
iStep  =     5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('Compute eFlux (Box) and 2d correlation maps with streaks for', len(iFiles), 'snapshot(s):', iFiles[0], 'to', iFiles[-1])

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

# prepare arrays for statistics
acUz        = np.zeros((nth, nz))  # initialise auto-correlation for u'_z
acUzF       = np.zeros((nth, nz))  # initialise auto-correlation for u'_z Box filtered
acOmegaZ    = np.zeros((nth, nz))  # initialise auto-correlation for omega_z
acPi        = np.zeros((nth, nz))  # initialise auto-correlation for Pi
ccUzPi      = np.zeros((nth, nz))  # initialise cross-correlation for u'_z and eFlux
ccUzFPi     = np.zeros((nth, nz))  # initialise cross-correlation for u'_z filtered and eFlux
ccOmegaZPi  = np.zeros((nth, nz))  # initialise cross-correlation for omega_z nd eFlux
ccUzPip     = np.zeros((nth, nz))  # initialise cross-correlation for u'_z and eFlux > 0 fwd only
ccUzFPip    = np.zeros((nth, nz))  # initialise cross-correlation for u'_z filtered and eFlux > 0 fwd only
ccOmegaZPip = np.zeros((nth, nz))  # initialise cross-correlation for omega_z and eFlux > 0 fwd only
ccUzPin     = np.zeros((nth, nz))  # initialise cross-correlation for u'_z and eFlux < 0 bwd only
ccUzFPin    = np.zeros((nth, nz))  # initialise cross-correlation for u'_z filtered and eFlux < 0 bwd only
ccOmegaZPin = np.zeros((nth, nz))  # initialise cross-correlation for omega_z and eFlux < 0 bwd only
nt          = 0                    # reset ensemble counter

# first and second statistical moments for normalisation
uz1     = 0 # streaks
uz2     = 0
uzF1    = 0 # filtered streaks
uzF2    = 0
omegaZ1 = 0 # streamwise vortices
omegaZ2 = 0
pi1     = 0 # total flux
pi2     = 0
pip1    = 0 # only forward (positive) flux
pip2    = 0
pin1    = 0 # only backward (negative) flux
pin2    = 0

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

    # compute instantaneous vorticity vector field
    tvort = timeit.default_timer()
    print('Computing vorticity vector field... ', end='', flush=True)
    import vorticity as v
    omegaR, omegaTh, omegaZ = v.omegaCyl(u_r, u_th, u_z, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tvort), 'seconds')

    # subtract mean velocity profile (1d) to obtain full (3d) fluctuating velocity field
    u_z = u_z - np.tile(u_zM, (nz, nth, 1)).T

    # filter velocity field
    print('Filtering velocity components and mixed terms... ', end='', flush=True)
    tfilter = timeit.default_timer()
    import filter2d as f2
    u_rF    = f2.box2d(u_r,       lambdaTh, lambdaZ, r, th, z)
    u_thF   = f2.box2d(u_th,      lambdaTh, lambdaZ, r, th, z)
    u_zF    = f2.box2d(u_z,       lambdaTh, lambdaZ, r, th, z)
    u_rRF   = f2.box2d(u_r*u_r,   lambdaTh, lambdaZ, r, th, z)
    u_rThF  = f2.box2d(u_r*u_th,  lambdaTh, lambdaZ, r, th, z)
    u_rZF   = f2.box2d(u_r*u_z,   lambdaTh, lambdaZ, r, th, z)
    u_thThF = f2.box2d(u_th*u_th, lambdaTh, lambdaZ, r, th, z)
    u_thZF  = f2.box2d(u_th*u_z,  lambdaTh, lambdaZ, r, th, z)
    u_zZF   = f2.box2d(u_z*u_z,   lambdaTh, lambdaZ, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tfilter), 'seconds')

    # compute instantaneous energy flux
    tflux = timeit.default_timer()
    print('Computing energy flux... ', end='', flush=True)
    import eFlux
    pi = eFlux.eFlux(u_rF, u_thF, u_zF, u_rRF, u_rThF, u_rZF, u_thThF, u_thZF, u_zZF, r, th, z)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tflux), 'seconds')

    # extract 2d data sub-sets in a wall parallel plane
    k = 65
    print("Extracting 2d data in wall-parallel plane at y+ =", (1-r[k])*ReTau)
    uz2d     =    u_z[k, :, :]
    uzF2d    =   u_zF[k, :, :]
    omegaZ2d = omegaZ[k, :, :]
    pi2d     =     pi[k, :, :]

    # isolate forward/backward flux events in the 2d velocity subset
    pip2d = np.where(pi2d > 0, pi2d, 0) # only positive, zero elsewhere
    pin2d = np.where(pi2d < 0, pi2d, 0) # only negative, zero elsewhere

    # compute correlations and sum up temporal (ensemble) statistics
    tcorr = timeit.default_timer()
    print('Computing 2d correlations... ', end='', flush=True)
    import crossCorrelation as c 
    acUz        = acUz        + c.corr2d(uz2d,     uz2d)     # auto-correlations
    acUzF       = acUzF       + c.corr2d(uzF2d,    uzF2d)
    acOmegaZ    = acOmegaZ    + c.corr2d(omegaZ2d, omegaZ2d)
    acPi        = acPi        + c.corr2d(pi2d,     pi2d)
    ccUzPi      = ccUzPi      + c.corr2d(uz2d,     pi2d)     # cross-correlations full flux
    ccUzFPi     = ccUzFPi     + c.corr2d(uzF2d,    pi2d)
    ccOmegaZPi  = ccOmegaZPi  + c.corr2d(omegaZ2d, pi2d)
    ccUzPip     = ccUzPip     + c.corr2d(uz2d,     pip2d)    # cross-correlations forward flux
    ccUzFPip    = ccUzFPip    + c.corr2d(uzF2d,    pip2d)
    ccOmegaZPip = ccOmegaZPip + c.corr2d(omegaZ2d, pip2d)
    ccUzPin     = ccUzPin     + c.corr2d(uz2d,     pin2d)    # cross-correlations full flux
    ccUzFPin    = ccUzFPin    + c.corr2d(uzF2d,    pin2d)
    ccOmegaZPin = ccOmegaZPin + c.corr2d(omegaZ2d, pin2d)
    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tcorr), 'seconds')

    # sum up first and second statistical moments in time and (homogeneous) theta and z direction for normalisation
    uz1     = uz1     + np.sum(np.sum(uz2d,        axis=1), axis=0) # sum over all elements of 2d data sub set
    uz2     = uz2     + np.sum(np.sum(uz2d**2,     axis=1), axis=0)
    uzF1    = uzF1    + np.sum(np.sum(uzF2d,       axis=1), axis=0)
    uzF2    = uzF2    + np.sum(np.sum(uzF2d**2,    axis=1), axis=0)
    omegaZ1 = omegaZ1 + np.sum(np.sum(omegaZ2d,    axis=1), axis=0)
    omegaZ2 = omegaZ2 + np.sum(np.sum(omegaZ2d**2, axis=1), axis=0)
    pi1     = pi1     + np.sum(np.sum(pi2d,        axis=1), axis=0)
    pi2     = pi2     + np.sum(np.sum(pi2d**2,     axis=1), axis=0)
    pip1    = pip1    + np.sum(np.sum(pip2d,       axis=1), axis=0)
    pip2    = pip2    + np.sum(np.sum(pip2d**2,    axis=1), axis=0)
    pin1    = pin1    + np.sum(np.sum(pin2d,       axis=1), axis=0)
    pin2    = pin2    + np.sum(np.sum(pin2d**2,    axis=1), axis=0)

    # increase temporal/ensemble counter
    nt = nt + 1

# divide correlation statistics by total number of temporal samples
acUz        = acUz        / nt
acUzF       = acUzF       / nt
acOmegaZ    = acOmegaZ    / nt
acPi        = acPi        / nt
ccUzPi      = ccUzPi      / nt
ccUzFPi     = ccUzFPi     / nt
ccOmegaZPi  = ccOmegaZPi  / nt
ccUzPip     = ccUzPip     / nt
ccUzFPip    = ccUzFPip    / nt
ccOmegaZPip = ccOmegaZPip / nt
ccUzPin     = ccUzPin     / nt
ccUzFPin    = ccUzFPin    / nt
ccOmegaZPin = ccOmegaZPin / nt

# divide normalisation statistics by total number of spatio-temporal samples
uz1     = uz1     / (nth*nz*nt)
uz2     = uz2     / (nth*nz*nt)
uzF1    = uzF1    / (nth*nz*nt)
uzF2    = uzF2    / (nth*nz*nt)
omegaZ1 = omegaZ1 / (nth*nz*nt)
omegaZ2 = omegaZ2 / (nth*nz*nt)
pi1     = pi1     / (nth*nz*nt)
pi2     = pi2     / (nth*nz*nt)
pip1    = pip1    / (nth*nz*nt)
pip2    = pip2    / (nth*nz*nt)
pin1    = pin1    / (nth*nz*nt)
pin2    = pin2    / (nth*nz*nt)

# compute RMS for normalisation
uzRms     = np.sqrt(uz2     - uz1**2)
uzFRms    = np.sqrt(uzF2    - uzF1**2)
omegaZRms = np.sqrt(omegaZ2 - omegaZ1**2)
piRms     = np.sqrt(pi2     - pi1**2)
pipRms    = np.sqrt(pip2    - pip1**2)
pinRms    = np.sqrt(pin2    - pin1**2)

# normalise correlations with local RMS 
acUz        = acUz        / (uzRms     * uzRms)
acUzF       = acUzF       / (uzFRms    * uzFRms)
acOmegaZ    = acOmegaZ    / (omegaZRms * omegaZRms)
acPi        = acPi        / (piRms     * piRms)
ccUzPi      = ccUzPi      / (uzRms     * piRms)
ccUzFPi     = ccUzFPi     / (uzFRms    * piRms)
ccOmegaZPi  = ccOmegaZPi  / (omegaZRms * piRms)
ccUzPip     = ccUzPip     / (uzRms     * pipRms)
ccUzFPip    = ccUzFPip    / (uzFRms    * pipRms)
ccOmegaZPip = ccOmegaZPip / (omegaZRms * pipRms)
ccUzPin     = ccUzPin     / (uzRms     * pinRms)
ccUzFPin    = ccUzFPin    / (uzFRms    * pinRms)
ccOmegaZPin = ccOmegaZPin / (omegaZRms * pinRms)

print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')

# compute centered azimuthal and axial separation/displacement (for nice plotting only)
DeltaTh = (th - (th[-1] - th[0]) / 2.0) * r[k]
DeltaZ  =   z - ( z[-1] -  z[0]) / 2.0

# find and report absolute maxima of 2d data sets
amacUz        = np.max(np.abs(acUz))        # max auto-correlation u'_z
amacUzF       = np.max(np.abs(acUzF))       # max auto-correlation filtered u'_z
amacOmegaZ    = np.max(np.abs(acOmegaZ))    # max auto-correlation omega_z
amacPi        = np.max(np.abs(acPi))        # max auto-correlation Pi
amccUzPi      = np.max(np.abs(ccUzPi))      # max cross-correlation u'_z Pi
amccUzFPi     = np.max(np.abs(ccUzFPi))     # max cross-correlation filtered u'_z Pi
amccOmegaZPi  = np.max(np.abs(ccOmegaZPi))  # max cross-correlation omega_z Pi
amccUzPip     = np.max(np.abs(ccUzPip))     # max cross-correlation u'_z Pi > 0
amccUzFPip    = np.max(np.abs(ccUzFPip))    # max cross-correlation filtered u'_z Pi > 0
amccOmegaZPip = np.max(np.abs(ccOmegaZPip)) # max cross-correlation omega_z Pi > 0
amccUzPin     = np.max(np.abs(ccUzPin))     # max cross-correlation u'_z Pi < 0
amccUzFPin    = np.max(np.abs(ccUzFPin))    # max cross-correlation filtered u'_z Pi < 0
amccOmegaZPin = np.max(np.abs(ccOmegaZPin)) # max cross-correlation omega_z Pi < 0
print("Absolute maximum auto-correlation value  u'_z    u'_z   :", amacUz)
print("Absolute maximum auto-correlation value  u'_zF   u'_zF  :", amacUzF)
print("Absolute maximum auto-correlation value  omega_z omega_z:", amacOmegaZ)
print("Absolute maximum auto-correlation value  Pi      Pi     :", amacPi)
print("Absolute maximum cross-correlation value u'_z    Pi     :", amccUzPi)
print("Absolute maximum cross-correlation value u'_zF   Pi     :", amccUzFPi)
print("Absolute maximum cross-correlation value omega_z Pi     :", amccOmegaZPi)
print("Absolute maximum cross-correlation value u'_z    Pi > 0 :", amccUzPip)
print("Absolute maximum cross-correlation value u'_zF   Pi > 0 :", amccUzFPip)
print("Absolute maximum cross-correlation value omega_z Pi > 0 :", amccOmegaZPip)
print("Absolute maximum cross-correlation value u'_z    Pi < 0 :", amccUzPin)
print("Absolute maximum cross-correlation value u'_zF   Pi < 0 :", amccUzFPin)
print("Absolute maximum cross-correlation value omega_z Pi < 0 :", amccOmegaZPin)

# write 2d correlation map to ascii file
fnam = 'piCorrThZStreaksBox2d_pipe0002_'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
f = open(fnam, 'w')
f.write("# Two-dimensional two-point correlation maps in a theta-z plane\n")
f.write("# At wall-normal location r = %f -> y+ = %f\n" % (r[k], (1.0-r[k])*ReTau))
f.write("# For the following flow variables:\n")
f.write("# + Streamwise velocity u'_z (High-speed and low-speed streaks)\n")
f.write("# + Filtered streamwise velocity u'_zF (Smoothed streaks)\n")
f.write("# + Axial vorticity component omega_z (Streamwise aligned vortices)\n")
f.write("# + Inter-scale energy flux Pi (full, only positive, only negative) across scale lambda\n")
f.write("# Flux and filtered quantities based on 2d Box kernel with:\n")
f.write("# + Azimuthal filter scale: lambdaTh+ = %f viscous units, lambdaTh = %f R\n" % (lambdaThp, lambdaTh))
f.write("# + Axial filter scale:     lambdaZ+  = %f viscous units, lambdaZ  = %f R\n" % (lambdaZp,  lambdaZ))
f.write("# Python post-processing on data set nsPipe/pipe0002 generated in a DNS using nsPipe\n")
f.write("# Temporal (ensemble) averaging over %d sample(s)\n" % (nt))
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# 01st column: Azimuthal separation DeltaTh in units of pipe radii (R), nth = %d points\n" % nth)
f.write("# 02nd column: Axial separation DeltaZ in units of pipe radii (R), nz = %d points\n" % nz)
f.write("# 03rd column: Auto-correlation  u'_z    with u'_z\n")
f.write("# 04th column: Auto-correlation  u'_zF   with u'_zF\n")
f.write("# 05th column: Auto-correlation  omega_z with omega_z\n")
f.write("# 06th column: Auto-correlation  Pi      with Pi\n")
f.write("# 07th column: Cross-correlation u'_z    with Pi\n")
f.write("# 08th column: Cross-correlation u'_zF   with Pi\n")
f.write("# 09th column: Cross-correlation omega_z with Pi\n")
f.write("# 10th column: Cross-correlation u'_z    with Pi > 0\n")
f.write("# 11th column: Cross-correlation u'_zF   with Pi > 0\n")
f.write("# 12th column: Cross-correlation omega_z with Pi > 0\n")
f.write("# 13th column: Cross-correlation u'_z    with Pi < 0\n")
f.write("# 14th column: Cross-correlation u'_zF   with Pi < 0\n")
f.write("# 15th column: Cross-correlation omega_z with Pi < 0\n")
for i in range(nth):
 for j in range(nz):
  f.write("%23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n" % (DeltaTh[i], DeltaZ[j], acUz[i,j], acUzF[i,j], acOmegaZ[i,j], acPi[i,j], ccUzPi[i,j], ccUzFPi[i,j], ccOmegaZPi[i,j], ccUzPip[i,j], ccUzFPip[i,j], ccOmegaZPip[i,j], ccUzPin[i,j], ccUzFPin[i,j], ccOmegaZPin[i,j]))
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
ig = ImageGrid(fig, 111, nrows_ncols=(4, 4), direction='column', axes_pad=(0.6, 0.15), cbar_size=0.07, cbar_mode='each', cbar_location='right', cbar_pad=0.05)

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

# plot auto-correlation streaks
ig[1].set_xlim(left=xmin, right=xmax)
ig[1].set_ylabel(r"$\Delta\theta r^{+}$")
ig[1].set_ylim(bottom=ymin, top=ymax)
im1 = ig[1].imshow(acUz, vmin=-amacUz, vmax=+amacUz, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[1].set_aspect('equal')
cb1 = ig.cbar_axes[1].colorbar(im1, format=fmt)
cb1.ax.set_ylabel(r"$C_{u^{\prime}_{z} u^{\prime}_{z}}$")
cb1.ax.set_yticks([-1.0, 0.0, +1.0])

# plot auto-correlation filtered streaks
ig[2].set_xlim(left=xmin, right=xmax)
ig[2].set_ylabel(r"$\Delta\theta r^{+}$")
ig[2].set_ylim(bottom=ymin, top=ymax)
im2 = ig[2].imshow(acUzF, vmin=-amacUzF, vmax=+amacUzF, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[2].set_aspect('equal')
cb2 = ig.cbar_axes[2].colorbar(im2, format=fmt)
cb2.ax.set_ylabel(r"$C_{\overline{u^{\prime}_{z}} \overline{u^{\prime}_{z}}}$")
cb2.ax.set_yticks([-1.0, 0.0, +1.0])

# plot auto-correlation vortices
ig[3].set_xlim(left=xmin, right=xmax)
ig[3].set_ylabel(r"$\Delta\theta r^{+}$")
ig[3].set_ylim(bottom=ymin, top=ymax)
im3 = ig[3].imshow(acOmegaZ, vmin=-amacOmegaZ, vmax=+amacOmegaZ, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[3].set_aspect('equal')
cb3 = ig.cbar_axes[3].colorbar(im3, format=fmt)
cb3.ax.set_ylabel(r"$C_{\omega_{z}\omega_{z}}$")
cb3.ax.set_yticks([-1.0, 0.0, +1.0])

# empty space for filter kernel label
filterBox = dict(boxstyle="square, pad=0.3", lw=0.5, fc='w', ec=Black)
ig[4].axis("off")
ig[4].text(0.0, 0.0, r"Box", ha="center", va="center", rotation=0, bbox=filterBox)
# TODO: remove empty colorbar

# plot cross-correlation streaks Pi full
ig[5].set_xlim(left=xmin, right=xmax)
ig[5].set_ylim(bottom=ymin, top=ymax)
im5 = ig[5].imshow(ccUzPi, vmin=-amccUzPi, vmax=+amccUzPi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[5].set_aspect('equal')
cb5 = ig.cbar_axes[5].colorbar(im5, format=fmt)
cb5.ax.set_ylabel(r"$C_{u^{\prime}_{z}\Pi}$")
cb5.ax.set_yticks([-amccUzPi, 0.0, +amccUzPi])

# plot cross-correlation filtered streaks Pi full
ig[6].set_xlim(left=xmin, right=xmax)
ig[6].set_ylim(bottom=ymin, top=ymax)
im6 = ig[6].imshow(ccUzFPi, vmin=-amccUzFPi, vmax=+amccUzFPi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[6].set_aspect('equal')
cb6 = ig.cbar_axes[6].colorbar(im6, format=fmt)
cb6.ax.set_ylabel(r"$C_{\overline{u^{\prime}_{z}}\Pi}$")
cb6.ax.set_yticks([-amccUzFPi, 0.0, +amccUzFPi])

# plot cross-correlation vortices Pi full
ig[7].set_xlim(left=xmin, right=xmax)
ig[7].set_ylim(bottom=ymin, top=ymax)
im7 = ig[7].imshow(ccOmegaZPi, vmin=-amccOmegaZPi, vmax=+amccOmegaZPi, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[7].set_aspect('equal')
cb7 = ig.cbar_axes[7].colorbar(im7, format=fmt)
cb7.ax.set_ylabel(r"$C_{\omega_{z}\Pi}$")
cb7.ax.set_yticks([-amccOmegaZPi, 0.0, +amccOmegaZPi])

# empty
ig[8].axis("off")

# plot cross-correlation streaks Pi > 0
ig[9].set_xlim(left=xmin, right=xmax)
ig[9].set_ylim(bottom=ymin, top=ymax)
im9 = ig[9].imshow(ccUzPip, vmin=-amccUzPip, vmax=+amccUzPip, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[9].set_aspect('equal')
cb9 = ig.cbar_axes[9].colorbar(im9, format=fmt)
cb9.ax.set_ylabel(r"$C_{u^{\prime}_{z}\Pi^{+}}$")
cb9.ax.set_yticks([-amccUzPip, 0.0, +amccUzPip])

# plot cross-correlation filtered streaks Pi > 0
ig[10].set_xlim(left=xmin, right=xmax)
ig[10].set_ylim(bottom=ymin, top=ymax)
im10 = ig[10].imshow(ccUzFPip, vmin=-amccUzFPip, vmax=+amccUzFPip, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[10].set_aspect('equal')
cb10 = ig.cbar_axes[10].colorbar(im10, format=fmt)
cb10.ax.set_ylabel(r"$C_{\overline{u^{\prime}_{z}}\Pi^{+}}$")
cb10.ax.set_yticks([-amccUzFPip, 0.0, +amccUzFPip])

# plot cross-correlation vortices Pi > 0
ig[11].set_xlim(left=xmin, right=xmax)
ig[11].set_ylim(bottom=ymin, top=ymax)
im11 = ig[11].imshow(ccOmegaZPip, vmin=-amccOmegaZPip, vmax=+amccOmegaZPip, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[11].set_aspect('equal')
cb11 = ig.cbar_axes[11].colorbar(im11, format=fmt)
cb11.ax.set_ylabel(r"$C_{\omega_{z}\Pi^{+}}$")
cb11.ax.set_yticks([-amccOmegaZPip, 0.0, +amccOmegaZPip])

# empty
ig[12].axis("off")

# plot cross-correlation streaks Pi < 0
ig[13].set_xlim(left=xmin, right=xmax)
ig[13].set_ylim(bottom=ymin, top=ymax)
im13 = ig[13].imshow(ccUzPin, vmin=-amccUzPin, vmax=+amccUzPin, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[13].set_aspect('equal')
cb13 = ig.cbar_axes[13].colorbar(im13, format=fmt)
cb13.ax.set_ylabel(r"$C_{u^{\prime}_{z}\Pi^{-}}$")
cb13.ax.set_yticks([-amccUzPin, 0.0, +amccUzPin])

# plot cross-correlation filtered streaks Pi < 0
ig[14].set_xlim(left=xmin, right=xmax)
ig[14].set_ylim(bottom=ymin, top=ymax)
im14 = ig[14].imshow(ccUzFPin, vmin=-amccUzFPin, vmax=+amccUzFPin, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[14].set_aspect('equal')
cb14 = ig.cbar_axes[14].colorbar(im14, format=fmt)
cb14.ax.set_ylabel(r"$C_{\overline{u^{\prime}_{z}}\Pi^{-}}$")
cb14.ax.set_yticks([-amccUzFPin, 0.0, +amccUzFPin])

# plot cross-correlation vortices Pi < 0
ig[15].set_xlim(left=xmin, right=xmax)
ig[15].set_ylim(bottom=ymin, top=ymax)
im15 = ig[15].imshow(ccOmegaZPin, vmin=-amccOmegaZPin, vmax=+amccOmegaZPin, cmap=VermBlue, interpolation='bilinear', extent=[np.min(DeltaZ), np.max(DeltaZ), np.min(DeltaTh), np.max(DeltaTh)], origin='lower')
ig[15].set_aspect('equal')
cb15 = ig.cbar_axes[15].colorbar(im15, format=fmt)
cb15.ax.set_ylabel(r"$C_{\omega_{z}\Pi^{-}}$")
cb15.ax.set_yticks([-amccOmegaZPin, 0.0, +amccOmegaZPin])

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
