#!/usr/bin/env python3
# Purpose:  Read full 3d pipe flow velocity fields from one or more HDF5 files.
#           Read the computational grid in cylindrical co-ordinates from one of
#           these files. Read a statistically converged mean velocity profile
#           from a separate ascii file and subtract it from each snapshot to
#           obtain the fluctuating velocity field. Define the filter widths and
#           compute the inter-scale turbulent kinetic energy flux field (Pi) for
#           each individual snapshot based on a two-dimensional spatial Gauss
#           filter operation in wall-parallel planes for each radial location.
#           Additionally, compute the vorticity vector field (omega) for each
#           snapshot. Finally, compute one-dimensional two-point correlations
#           in axial (z) direction; auto-correlations for the original (u'_z)
#           and filtered (u'_zF) streamwise velocity component (representing
#           streaks), for the axial vorticity component (representing streamwise
#           alligned vortices) and for the energy flux, cross-correlations for
#           all of these quantities with the energy flux. Do statistics over all
#           azimuthal (theta) locations and all snapshots, and write the
#           resulting 1d correlations to a single ascii file. Optionally, plot
#           the results interactively or as pdf figure file.
# Usage:    python piCorrZStreaksGauss2d.py
# Authors:  Daniel Feldmann, Mohammad Umair, Jan Chen
# Date:     28th March 2019
# Modified: 22nd December 2019

import timeit
import math
import numpy as np
import h5py

# plot mode: (0) none, (1) interactive, (2) pdf
plot = 2

# range of state files to read flow field data
iFirst =  1675000 # 570000
iLast  =  1675000
iStep  =     5000
iFiles = range(iFirst, iLast+iStep, iStep)
print('Compute eFlux (Gauss) and 1d axial correlations with streaks for', len(iFiles), 'snapshot(s):', iFiles[0], 'to', iFiles[-1])

# path to data files (do modify)
fpath = '../../outFiles/'

# read grid from first HDF5 file
fnam = fpath+'fields_pipe0002_'+'{:08d}'.format(iFirst)+'.h5'
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

# prepare arrays for statistics
acUz        = np.zeros(nz) # initialise auto-correlation for u'_z
acUzF       = np.zeros(nz) # initialise auto-correlation for u'_z Gauss filtered
acOmegaZ    = np.zeros(nz) # initialise auto-correlation for omega_z
acPi        = np.zeros(nz) # initialise auto-correlation for Pi
ccUzPi      = np.zeros(nz) # initialise cross-correlation for u'_z and eFlux
ccUzFPi     = np.zeros(nz) # initialise cross-correlation for u'_z filtered and eFlux
ccOmegaZPi  = np.zeros(nz) # initialise cross-correlation for omega_z nd eFlux
ccUzPip     = np.zeros(nz) # initialise cross-correlation for u'_z and eFlux > 0 fwd only
ccUzFPip    = np.zeros(nz) # initialise cross-correlation for u'_z filtered and eFlux > 0 fwd only
ccOmegaZPip = np.zeros(nz) # initialise cross-correlation for omega_z and eFlux > 0 fwd only
ccUzPin     = np.zeros(nz) # initialise cross-correlation for u'_z and eFlux < 0 bwd only
ccUzFPin    = np.zeros(nz) # initialise cross-correlation for u'_z filtered and eFlux < 0 bwd only
ccOmegaZPin = np.zeros(nz) # initialise cross-correlation for omega_z and eFlux < 0 bwd only
nt          = 0            # reset ensemble counter

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
    fnam = fpath+'fields_pipe0002_'+'{:08d}'.format(iFile)+'.h5'
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

    # fix wall-normal (radial) location
    k = 65
    print("Extracting 1d axial data sets at wall-normal location y+ =", (1-r[k])*ReTau)
    
    tcorr = timeit.default_timer()
    print('Computing 1d correlations... ', end='', flush=True)

    # loop over all azimuthal (theta) locations
    for l in range(nth):

        # extract 1d data sub-sets along axial line at constant wall distance
        uz1d     =    u_z[k, l, :] # data structure is (r, theta, z)
        uzF1d    =   u_zF[k, l, :]
        omegaZ1d = omegaZ[k, l, :]
        pi1d     =     pi[k, l, :]

        # isolate forward/backward flux events in the 1d velocity subset
        pip1d = np.where(pi1d > 0, pi1d, 0) # only positive, zero elsewhere
        pin1d = np.where(pi1d < 0, pi1d, 0) # only negative, zero elsewhere

        # compute correlations and sum up azimuthal (spatial) and temporal (ensemble) statistics
        import crossCorrelation as c 
        acUz        = acUz        + c.corr1d(uz1d,     uz1d)     # auto-correlations
        acUzF       = acUzF       + c.corr1d(uzF1d,    uzF1d)
        acOmegaZ    = acOmegaZ    + c.corr1d(omegaZ1d, omegaZ1d)
        acPi        = acPi        + c.corr1d(pi1d,     pi1d)
        ccUzPi      = ccUzPi      + c.corr1d(uz1d,     pi1d)     # cross-correlations full flux
        ccUzFPi     = ccUzFPi     + c.corr1d(uzF1d,    pi1d)
        ccOmegaZPi  = ccOmegaZPi  + c.corr1d(omegaZ1d, pi1d)
        ccUzPip     = ccUzPip     + c.corr1d(uz1d,     pip1d)    # cross-correlations forward flux
        ccUzFPip    = ccUzFPip    + c.corr1d(uzF1d,    pip1d)
        ccOmegaZPip = ccOmegaZPip + c.corr1d(omegaZ1d, pip1d)
        ccUzPin     = ccUzPin     + c.corr1d(uz1d,     pin1d)    # cross-correlations backward flux
        ccUzFPin    = ccUzFPin    + c.corr1d(uzF1d,    pin1d)
        ccOmegaZPin = ccOmegaZPin + c.corr1d(omegaZ1d, pin1d)

        # sum up first and second statistical moments in time and (homogeneous) theta z direction for normalisation
        uz1     = uz1     + np.sum(uz1d)
        uz2     = uz2     + np.sum(uz1d**2)
        uzF1    = uzF1    + np.sum(uzF1d)
        uzF2    = uzF2    + np.sum(uzF1d**2)
        omegaZ1 = omegaZ1 + np.sum(omegaZ1d)
        omegaZ2 = omegaZ2 + np.sum(omegaZ1d**2)
        pi1     = pi1     + np.sum(pi1d)
        pi2     = pi2     + np.sum(pi1d**2)
        pip1    = pip1    + np.sum(pip1d)
        pip2    = pip2    + np.sum(pip1d**2)
        pin1    = pin1    + np.sum(pin1d)
        pin2    = pin2    + np.sum(pin1d**2)

    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tcorr), 'seconds')

    # increase temporal/ensemble counter
    nt = nt + 1

# divide correlation statistics by total number of spatio-temporal samples
acUz        = acUz        / (nt*nth)
acUzF       = acUzF       / (nt*nth)
acOmegaZ    = acOmegaZ    / (nt*nth)
acPi        = acPi        / (nt*nth)
ccUzPi      = ccUzPi      / (nt*nth)
ccUzFPi     = ccUzFPi     / (nt*nth)
ccOmegaZPi  = ccOmegaZPi  / (nt*nth)
ccUzPip     = ccUzPip     / (nt*nth)
ccUzFPip    = ccUzFPip    / (nt*nth)
ccOmegaZPip = ccOmegaZPip / (nt*nth)
ccUzPin     = ccUzPin     / (nt*nth)
ccUzFPin    = ccUzFPin    / (nt*nth)
ccOmegaZPin = ccOmegaZPin / (nt*nth)

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
uzRms     = np.sqrt(uz2 - uz1**2)
uzFRms    = np.sqrt(uzF2 - uzF1**2)
omegaZRms = np.sqrt(omegaZ2 - omegaZ1**2)
piRms     = np.sqrt(pi2 - pi1**2)
pipRms    = np.sqrt(pip2 - pip1**2)
pinRms    = np.sqrt(pin2 - pin1**2)
print('uzMean', uz1)
print('uzRms', uzRms)

# normalise correlations with local RMS 
acUz        = acUz       / (uzRms     * uzRms)
acUzF       = acUzF      / (uzFRms    * uzFRms)
acOmegaZ    = acOmegaZ   / (omegaZRms * omegaZRms)
acPi        = acPi       / (piRms     * piRms)
ccUzPi      = ccUzPi     / (uzRms     * piRms)
ccUzFPi     = ccUzFPi    / (uzFRms    * piRms)
ccOmegaZPi  = ccOmegaZPi / (omegaZRms * piRms)
ccUzPip     = ccUzPip     / (uzRms     * pipRms)
ccUzFPip    = ccUzFPip    / (uzFRms    * pipRms)
ccOmegaZPip = ccOmegaZPip / (omegaZRms * pipRms)
ccUzPin     = ccUzPin     / (uzRms     * pinRms)
ccUzFPin    = ccUzFPin    / (uzFRms    * pinRms)
ccOmegaZPin = ccOmegaZPin / (omegaZRms * pinRms)

print('Total elapsed wall-clock time:', '{:3.1f}'.format(timeit.default_timer()-t0), 'seconds')

# compute centered axial separation/displacement (for nice plotting only)
DeltaZ  =   z - ( z[-1] -  z[0]) / 2.0

# write 1d correlations to ascii file
fnam = 'piCorrZStreaksGauss2d_pipe0002_'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
f = open(fnam, 'w')
f.write("# One-dimensional two-point correlations in axial (z) direction\n")
f.write("# At wall-normal location r = %f -> y+ = %f\n" % (r[k], (1.0-r[k])*ReTau))
f.write("# For the following flow variables:\n")
f.write("# + Streamwise velocity u'_z (High-speed and low-speed streaks)\n")
f.write("# + Filtered streamwise velocity u'_zF (Smoothed streaks)\n")
f.write("# + Axial vorticity component omega_z (Streamwise aligned vortices)\n")
f.write("# + Inter-scale energy flux Pi (full, only positive, only negative) across scale lambda\n")
f.write("# Flux and filtered quantities based on 2d Gauss kernel with:\n")
f.write("# + Azimuthal filter scale: lambdaTh+ = %f viscous units, lambdaTh = %f R\n" % (lambdaThp, lambdaTh))
f.write("# + Axial filter scale:     lambdaZ+  = %f viscous units, lambdaZ  = %f R\n" % (lambdaZp,  lambdaZ))
f.write("# Python post-processing on data set nsPipe/pipe0002 generated in a DNS using nsPipe\n")
f.write("# Temporal (ensemble) averaging over %d sample(s)\n" % (nt))
f.write("# First snapshot: %08d\n" % (iFirst))
f.write("# Last snapshot:  %08d\n" % (iLast))
f.write("# Additional spatial averaging in azimuthal (theta) direction over %d points\n" % (nth))
f.write("# 01st column: Axial separation DeltaZ in units of pipe radii (R), nz = %d points\n" % nz)
f.write("# 02nd column: Auto-correlation  u'_z    with u'_z\n")
f.write("# 03rd column: Auto-correlation  u'_zF   with u'_zF\n")
f.write("# 04th column: Auto-correlation  omega_z with omega_z\n")
f.write("# 05th column: Auto-correlation  Pi      with Pi\n")
f.write("# 06th column: Cross-correlation u'_z    with Pi\n")
f.write("# 07th column: Cross-correlation u'_zF   with Pi\n")
f.write("# 08th column: Cross-correlation omega_z with Pi\n")
f.write("# 09th column: Cross-correlation u'_z    with Pi > 0\n")
f.write("# 10th column: Cross-correlation u'_zF   with Pi > 0\n")
f.write("# 11th column: Cross-correlation omega_z with Pi > 0\n")
f.write("# 12th column: Cross-correlation u'_z    with Pi < 0\n")
f.write("# 13th column: Cross-correlation u'_zF   with Pi < 0\n")
f.write("# 14th column: Cross-correlation omega_z with Pi < 0\n")
for i in range(nz):
  f.write("%23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n" % (DeltaZ[i], acUz[i], acUzF[i], acOmegaZ[i], acPi[i], ccUzPi[i], ccUzFPi[i], ccOmegaZPi[i], ccUzPip[i], ccUzFPip[i], ccOmegaZPip[i], ccUzPin[i], ccUzFPin[i], ccOmegaZPin[i]))
f.close()
print('Written 1d correlations to file', fnam)

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
fig = plt.figure(num=None, figsize=mm2inch(134.0, 140.0), dpi=300) # , constrained_layout=False) 
#fig = plt.figure(num=None, dpi=100) # , constrained_layout=False)

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

# convert spatial separation from outer to inner units
DeltaZ = DeltaZ * ReTau

# plot axial auto-correlations
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax1.set_xlabel(r"$\Delta z^+$")
ax1.set_ylabel(r"$C$")
ax1.axhline(y=0.0, color=Grey)
ax1.axvline(x=0.0, color=Grey)
ax1.plot(DeltaZ, acUz,     color=Black,       linestyle='-', label=r"$C_{u^{\prime}_{z}u^{\prime}_{z}}$")
ax1.plot(DeltaZ, acUzF,    color=Vermillion,  linestyle='-', label=r"$C_{\overline{u^{\prime}_{z}}\overline{u^{\prime}_{z}}}$")
ax1.plot(DeltaZ, acOmegaZ, color=Blue,        linestyle='-', label=r"$C_{\omega_{z}\omega_{z}}$")
ax1.plot(DeltaZ, acPi,     color=BluishGreen, linestyle='-', label=r"$C_{\Pi\Pi }$")
ax1.legend(loc='best', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)
ax1.text(-1000.0, 0.75, "Gauss", ha="center", va="center")

# plot axial cross-correlation full flux
ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax2.set_xlabel(r"$\Delta z^+$")
ax2.axhline(y=0.0, color=Grey)
ax2.axvline(x=0.0, color=Grey)
ax2.plot(DeltaZ, ccUzPi,     color=Black,      linestyle='-', label=r"$C_{u^{\prime}_{z}\Pi}$")
ax2.plot(DeltaZ, ccUzFPi,    color=Vermillion, linestyle='-', label=r"$C_{\overline{u^{\prime}_{z}}\Pi}$")
ax2.plot(DeltaZ, ccOmegaZPi, color=Blue,       linestyle='-', label=r"$C_{\omega_{z}\Pi}$")
ax2.legend(loc='best', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot axial cross-correlation positive flux
ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax3.set_xlabel(r"$\Delta z^+$")
ax3.axhline(y=0.0, color=Grey)
ax3.axvline(x=0.0, color=Grey)
ax3.plot(DeltaZ, ccUzPip,     color=Black,      linestyle='-', label=r"$C_{u^{\prime}_{z}\Pi^{+}}$")
ax3.plot(DeltaZ, ccUzFPip,    color=Vermillion, linestyle='-', label=r"$C_{\overline{u^{\prime}_{z}}\Pi^{+}}$")
ax3.plot(DeltaZ, ccOmegaZPip, color=Blue,       linestyle='-', label=r"$C_{\omega_{z}\Pi^{+}}$")
ax3.legend(loc='best', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot axial cross-correlation negative flux
ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
ax4.set_xlabel(r"$\Delta z^+$")
ax4.axhline(y=0.0, color=Grey)
ax4.axvline(x=0.0, color=Grey)
ax4.plot(DeltaZ, ccUzPin,     color=Black,      linestyle='-', label=r"$C_{u^{\prime}_{z}\Pi^{+}}$")
ax4.plot(DeltaZ, ccUzFPin,    color=Vermillion, linestyle='-', label=r"$C_{\overline{u^{\prime}_{z}}\Pi^{+}}$")
ax4.plot(DeltaZ, ccOmegaZPin, color=Blue,       linestyle='-', label=r"$C_{\omega_{z}\Pi^{+}}$")
ax4.legend(loc='best', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

# plot mode interactive or pdf
if plot != 2:
 plt.tight_layout()
 plt.show()
else:
 fig.tight_layout()
 fnam = str.replace(fnam, '.dat', '.pdf')
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()

print('Done!')
