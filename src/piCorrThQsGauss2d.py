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
#           snapshot. Finally, compute one-dimensional two-point correlations
#           in azimuthal (theta) direction; auto-correlations for the Q events
#           (representing important features of the near-wall cycle) and for the
#           energy flux, cross-correlations for all of the Q events with the
#           energy flux. Do statistics over all axial (z) locations and all
#           snapshots, and write the resulting 1d correlations to a single ascii
#           file. Optionally, plot the results interactively or as pdf figure
#           file.
# Usage:    python piCorrThQsGauss2d.py
# Authors:  Daniel Feldmann, Mohammad Umair, Jan Chen
# Date:     28th March 2019
# Modified: 20th September 2019

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
print('Compute eFlux (Gauss) and 1d azimuthal correlations with Q events for', len(iFiles), 'snapshot(s):', iFiles[0], 'to', iFiles[-1])

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
acQ1   = np.zeros(nth) # initialise auto-correlation for Q1 outward interactions
acQ2   = np.zeros(nth) # initialise auto-correlation for Q2 ejection events
acQ3   = np.zeros(nth) # initialise auto-correlation for Q3 inward interactions
acQ4   = np.zeros(nth) # initialise auto-correlation for Q4 sweep events
acPi   = np.zeros(nth) # initialise auto-correlation for Pi
ccQ1Pi = np.zeros(nth) # initialise cross-correlation for Q1 eFlux
ccQ2Pi = np.zeros(nth) # initialise cross-correlation for Q2 eFlux
ccQ3Pi = np.zeros(nth) # initialise cross-correlation for Q3 eFlux
ccQ4Pi = np.zeros(nth) # initialise cross-correlation for Q4 eFlux
nt     = 0             # reset ensemble counter

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
    print("Extracting 1d azimuthal data sets at wall-normal location y+ =", (1-r[k])*ReTau)
    
    tcorr = timeit.default_timer()
    print('Extracting Q events and computing 1d correlations... ', end='', flush=True)

    # loop over all axial (z) locations
    for l in range(nz):

        # extract 1d data sub-sets along azimuthal line at constant wall distance
        ur1d = u_r[k, :, l]  # data structure is (r, theta, z)
        uz1d = u_z[k, :, l] 
        pi1d =  pi[k, :, l]

        # detect and extract Q events from the 1d volocity sub-set
        q1 = np.zeros(nth) #* 1.0
        q2 = np.zeros(nth) #* 2.0
        q3 = np.zeros(nth) #+ 1.0
        q4 = np.zeros(nth) #+ 2.0
        for i in range(nth):
         if (uz1d[i]>0) and (ur1d[i]<0): q1[i] = ur1d[i]*uz1d[i] # outward interaction: high-speed fluid away from wall
         if (uz1d[i]<0) and (ur1d[i]<0): q2[i] = ur1d[i]*uz1d[i] # ejection event:       low-speed fluid away from wall
         if (uz1d[i]<0) and (ur1d[i]>0): q3[i] = ur1d[i]*uz1d[i] # inward interaction:   low-speed fluid towards   wall
         if (uz1d[i]>0) and (ur1d[i]>0): q4[i] = ur1d[i]*uz1d[i] # sweep event:         high-speed fluid towards   wall
        ioi = q1 - q3 # unify inward interactions (Q3 being negativ) and outward interactions (Q1 being positive) in one array
        see = q2 - q4 # unify sweep events (Q4 being negativ) and ejection events (Q2 being positiv) in one array
    
        # compute correlations and sum up axial (spatial) and temporal (ensemble) statistics
        import crossCorrelation as c 
        acQ1   = acQ1   + c.corr1d(q1,   q1)   # auto-correlations
        acQ2   = acQ2   + c.corr1d(q2,   q2)
        acQ3   = acQ3   + c.corr1d(q3,   q3)
        acQ4   = acQ4   + c.corr1d(q4,   q4)
        acPi   = acPi   + c.corr1d(pi1d, pi1d)
        ccQ1Pi = ccQ1Pi + c.corr1d(q1,   pi1d) # cross-correlations
        ccQ2Pi = ccQ2Pi + c.corr1d(q2,   pi1d)
        ccQ3Pi = ccQ3Pi + c.corr1d(q3,   pi1d)
        ccQ4Pi = ccQ4Pi + c.corr1d(q4,   pi1d)

        # sum up first and second statistical moments in time and (homogeneous) theta and z direction for normalisation
        q11 = q11 + np.sum(q1)
        q12 = q12 + np.sum(q1**2)
        q21 = q21 + np.sum(q2)
        q22 = q22 + np.sum(q2**2)
        q31 = q31 + np.sum(q3)
        q32 = q32 + np.sum(q3**2)
        q41 = q41 + np.sum(q4)
        q42 = q42 + np.sum(q4**2)
        pi1 = pi1 + np.sum(pi1d)
        pi2 = pi2 + np.sum(pi1d**2)

    print('Time elapsed:', '{:3.1f}'.format(timeit.default_timer()-tcorr), 'seconds')

    # increase temporal/ensemble counter
    nt = nt + 1

# divide correlation statistics by total number of spatio-temporal samples
acQ1   = acQ1   / (nt*nz)
acQ2   = acQ2   / (nt*nz)
acQ3   = acQ3   / (nt*nz)
acQ4   = acQ4   / (nt*nz)
acPi   = acPi   / (nt*nz)
ccQ1Pi = ccQ1Pi / (nt*nz)
ccQ2Pi = ccQ2Pi / (nt*nz)
ccQ3Pi = ccQ3Pi / (nt*nz)
ccQ4Pi = ccQ4Pi / (nt*nz)

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

# compute centered azimuthal separation/displacement (for nice plotting only)
DeltaTh = (th - (th[-1] - th[0]) / 2.0) * r[k]

# write 1d correlations to ascii file
fnam = 'piCorrThQsGauss2d_pipe0002_'+'{:08d}'.format(iFirst)+'to'+'{:08d}'.format(iLast)+'nt'+'{:04d}'.format(nt)+'.dat'
f = open(fnam, 'w')
f.write("# One-dimensional two-point correlations in azimuthal (theta) direction\n")
f.write("# At wall-normal location r = %f -> y+ = %f\n" % (r[k], (1.0-r[k])*ReTau))
f.write("# For the following flow variables:\n")
f.write("# + Q1 outward interactions (u'_z > 0 and u'_r < 0)\n")
f.write("# + Q2 ejection events      (u'_z < 0 and u'_r < 0)\n")
f.write("# + Q3 inward interactions  (u'_z < 0 and u'_r > 0)\n")
f.write("# + Q4 sweep events         (u'_z > 0 and u'_r > 0)\n")
f.write("# + Inter-scale energy flux Pi across scale lambda\n")
f.write("# Note that u'_r < 0 represents motion away from the wall in a cylindrical co-ordinate system)\n")
f.write("# Flux based on filtered quantities using a 2d box kernel with:\n")
f.write("# + Azimuthal filter scale: lambdaTh+ = %f viscous units, lambdaTh = %f R\n" % (lambdaThp, lambdaTh))
f.write("# + Axial filter scale:     lambdaZ+  = %f viscous units, lambdaZ  = %f R\n" % (lambdaZp,  lambdaZ))
f.write("# Python post-processing on data set nsPipe/pipe0002 generated in a DNS using nsPipe\n")
f.write("# Temporal (ensemble) averaging over %d sample(s)\n" % (nt))
f.write("# Additional spatial averaging in axial (z) direction over %d points\n" % (nz))
f.write("# 01st column: Azimuthal separation DeltaTh in units of pipe radii (R), nth = %d points\n" % nth)
f.write("# 02rd column: Auto-correlation  Q1 with Q1\n")
f.write("# 03th column: Auto-correlation  Q2 with Q2\n")
f.write("# 04th column: Auto-correlation  Q3 with Q3\n")
f.write("# 05th column: Auto-correlation  Q4 with Q4\n")
f.write("# 06th column: Auto-correlation  Pi with Pi\n")
f.write("# 07th column: Cross-correlation Q1 with Pi\n")
f.write("# 08th column: Cross-correlation Q2 with Pi\n")
f.write("# 09th column: Cross-correlation Q3 with Pi\n")
f.write("# 10th column: Cross-correlation Q4 with Pi\n")
for i in range(nth):
 f.write("%23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n" % (DeltaTh[i], acQ1[i], acQ2[i], acQ3[i], acQ4[i], acPi[i], ccQ1Pi[i], ccQ2Pi[i], ccQ3Pi[i], ccQ4Pi[i]))
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
fig = plt.figure(num=None, figsize=mm2inch(134.0, 70.0), dpi=300) # , constrained_layout=False) 
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

# convert spatial separation from outer to inner unit#s
DeltaTh = DeltaTh * ReTau

# plot azimuthal auto-correlations
ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
ax1.set_xlabel(r"$\Delta\theta r^+$")
ax1.set_ylabel(r"$C$")
ax1.axhline(y=0.0, color=Grey)
ax1.axvline(x=0.0, color=Grey)
ax1.plot(DeltaTh, acQ1, color=Black,       linestyle='-', label=r"$C_{Q_{1}Q_{1}}$")
ax1.plot(DeltaTh, acQ2, color=Vermillion,  linestyle='-', label=r"$C_{Q_{2}Q_{2}}$")
ax1.plot(DeltaTh, acQ3, color=Blue,        linestyle='-', label=r"$C_{Q_{3}Q_{3}}$")
ax1.plot(DeltaTh, acQ4, color=BluishGreen, linestyle='-', label=r"$C_{Q_{4}Q_{4}}$")
ax1.plot(DeltaTh, acPi, color=Orange,      linestyle='-', label=r"$C_{\Pi\Pi}$")
ax1.legend(loc='best', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)
ax1.text(-200.0, 0.75, "Gauss", ha="center", va="center")

# plot azimuthal cross-correlation
ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
ax2.set_xlabel(r"$\Delta\theta r^+$")
ax2.axhline(y=0.0, color=Grey)
ax2.axvline(x=0.0, color=Grey)
ax2.plot(DeltaTh, ccQ1Pi, color=Black,       linestyle='-', label=r"$C_{Q_{1}\Pi}$")
ax2.plot(DeltaTh, ccQ2Pi, color=Vermillion,  linestyle='-', label=r"$C_{Q_{2}\Pi}$")
ax2.plot(DeltaTh, ccQ3Pi, color=Blue,        linestyle='-', label=r"$C_{Q_{3}\Pi}$")
ax2.plot(DeltaTh, ccQ4Pi, color=BluishGreen, linestyle='-', label=r"$C_{Q_{4}\Pi}$")
ax2.legend(loc='best', frameon=False, fancybox=False, facecolor=None, edgecolor=None, framealpha=None)

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
