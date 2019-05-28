#!/usr/bin/env python3
#====================================================================================
# Purpose:  Computes the cross-correlation coefficient between the axial fluctuating 
#           velocity (both filtered and unfiltered), eflux and axial vorticity  with 
#           the Fourier filtered interscale energy flux in axial direction. 
#           Reads HDF5 files from given number of snapshots.
#           Computes the fluctuating field by subtracting the average from the statistics
#           file. Computes the filtered interscale energy flux using 2D Fourier filter.
#           Plots and prints the output in ascii format.
# ----------------------------------------------------------------------------------
# IMPORTANT:Make sure the statistics file should correspond to the given number
#           of snapshots 
# ----------------------------------------------------------------------------------
# Usage:    python crosscorrFourierZ.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 03rd May 2019
# ===================================================================================
import sys
import os.path
import timeit
import math
import numpy as np
import h5py

#-------------------------------------------
# read mean velocity profiles from ascii file
autoZF = '../autocorrelations1D/theta/autoCorr1dFourierTh00570000to01265000nt0140.dat'
autoZG = '../autocorrelations1D/theta/autoCorr1dGaussTh00570000to01265000nt0140.dat'
autoZB = '../autocorrelations1D/theta/autoCorr1dBoxTh00570000to01265000nt0140.dat'

dz = np.loadtxt(autoZF)[:,0]
acUz = np.loadtxt(autoZF)[:,1]
acOmg= np.loadtxt(autoZF)[:,4]

acPiF  = np.loadtxt(autoZF)[:,3]
acPiG  = np.loadtxt(autoZG)[:,3]
acPiB  = np.loadtxt(autoZB)[:,3]

crossZF = 'theta/crossCorr1dFourierTh00570000to01265000nt0140.dat'
crossZG = 'theta/crossCorr1dGaussTh00570000to01265000nt0140.dat'
crossZB = 'theta/crossCorr1dBoxTh00570000to01265000nt0140.dat'
print('-----------------------------------------------------------------')
print('Reading auto-correlations from', autoZF)
print('Reading auto-correlations from', autoZG)
print('Reading auto-correlations from', autoZB)

print('Reading cross-correlations from', crossZF)
print('Reading cross-correlations from', crossZG)
print('Reading cross-correlations from', crossZB)
print('-----------------------------------------------------------------')

ccUzPiF   = np.loadtxt(crossZF)[:,1]
ccUzPiG   = np.loadtxt(crossZG)[:,1]
ccUzPiB   = np.loadtxt(crossZB)[:,1]

ccOmgPiF  = np.loadtxt(crossZF)[:,4]
ccOmgPiG  = np.loadtxt(crossZG)[:,4]
ccOmgPiB  = np.loadtxt(crossZB)[:,4]
#--------------------------------------------------
# write correlation coefficient in ascii file
#=================================================================================
# plot data as graph, (0) none, (1) interactive, (2) pdf
plot = 2
if plot not in [1, 2]: sys.exit() # skip everything below
import matplotlib.pyplot as plt
import matplotlib as mpl
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
#fig = plt.figure(num=None, figsize=mm2inch(210.0, 297.0), dpi=600)
#fig = plt.figure(num=None, figsize=mm2inch(160.0, 140.0), dpi=600)
#fig = plt.figure(num=None, figsize=mm2inch(90.0, 70.0), dpi=150)

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

#-----------------------------------------------------------------------------------

for case in range(0,3):

 if case == 0:
  fig = plt.figure(num=None, figsize=mm2inch(160.0, 140.0), dpi=600)
  ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
  ax1.set_xlabel(r"$r\Delta\theta^+$ in $R$")
  ax1.set_ylabel('Cross-Correlations')
  ax1.plot(dz*180.4, acUz,    color=Black,       linestyle='-', label=r"$C_{u^{\prime}_zu^{\prime}_z}$")
  ax1.plot(dz*180.4, acPiG,   color=Blue,        linestyle='-', label=r"$C_{\tilde{\pi}\tilde{\pi}}$")
  ax1.plot(dz*180.4, ccUzPiG, color=BluishGreen, linestyle='-', label=r"$C_{u^{\prime}_z\tilde{\pi}}$")
  ax1.legend(loc='best', ncol=4)
  ax1.axhline(0, color='black', lw=0.5, linestyle='--')
  ax1.axvline(0, color='black', lw=0.5, linestyle='--')
  ax1.legend(loc='best')
 
  if plot != 2:
   plt.tight_layout()
   plt.show()
  else:
   fig.tight_layout()
   fnam = 'crossCorrUzPiTh.pdf'
   plt.savefig(fnam)
   print('Written file', fnam)
   print('=============================================================================================')
   fig.clf()
   case = case + 1

 if case == 1:
  fig = plt.figure(num=None, figsize=mm2inch(160.0, 140.0), dpi=600)
  ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
  ax1.set_xlabel(r"$r\Delta\theta^+$ in $R$")
  ax1.set_ylabel('Cross-Correlations')
  ax1.plot(dz*180.4, acOmg,    color=Black,       linestyle='-', label=r"$C_{u^{\prime}_zu^{\prime}_z}$")
  ax1.plot(dz*180.4, acPiG,    color=Blue,        linestyle='-', label=r"$C_{\omega\omega}$")
  ax1.plot(dz*180.4, ccOmgPiG, color=BluishGreen, linestyle='-', label=r"$C_{\omega\tilde{\pi}}$")
  ax1.legend(loc='best', ncol=4)
  ax1.axhline(0, color='black', lw=0.5, linestyle='--')
  ax1.axvline(0, color='black', lw=0.5, linestyle='--')
  ax1.legend(loc='best')

  if plot != 2:
   plt.tight_layout()
   plt.show()
  else:
   fig.tight_layout()
   fnam = 'crossCorrOmgPiTh.pdf'
   plt.savefig(fnam)
   print('Written file', fnam)
   print('=============================================================================================')
   fig.clf()
   case = case + 1


 if case == 2:
  fig = plt.figure(num=None, figsize=mm2inch(160.0, 140.0), dpi=600)
  ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
  ax1.set_xlabel(r"$r\Delta\theta^+$ in $R$")
  ax1.set_ylabel(r"$C_{u^{\prime}_z\pi}$")
  ax1.plot(dz*180.4, ccUzPiF, color=Black,       linestyle='-', label='Fourier')
  ax1.plot(dz*180.4, ccUzPiG, color=Blue,        linestyle='-', label='Gauss')
  ax1.plot(dz*180.4, ccUzPiB, color=BluishGreen, linestyle='-', label='Box')
  ax1.legend(loc='best', ncol=4)
  ax1.axhline(0, color='black', lw=0.5, linestyle='--')
  ax1.axvline(0, color='black', lw=0.5, linestyle='--')
  ax1.legend(loc='best')

  if plot != 2:
   plt.tight_layout()
   plt.show()
  else:
   fig.tight_layout()
   fnam = 'crossCorrUzPiFGBTh.pdf'
   plt.savefig(fnam)
   print('Written file', fnam)
   print('=============================================================================================')
   fig.clf()
   case = case + 1

 if case == 3:
  fig = plt.figure(num=None, figsize=mm2inch(160.0, 140.0), dpi=600)
  ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
  ax1.set_xlabel(r"$r\Delta\theta^+$ in $R$")
  ax1.set_ylabel(r"$C_{\omega_z\pi}$")
  ax1.plot(dz*180.4, ccOmgPiF, color=Black,       linestyle='-', label='Fourier')
  ax1.plot(dz*180.4, ccOmgPiG, color=Blue,        linestyle='-', label='Gauss')
  ax1.plot(dz*180.4, ccOmgPiB, color=BluishGreen, linestyle='-', label='Box')
  ax1.legend(loc='best', ncol=4)
  ax1.axhline(0, color='black', lw=0.5, linestyle='--')
  ax1.axvline(0, color='black', lw=0.5, linestyle='--')
  ax1.legend(loc='best')

  if plot != 2:
   plt.tight_layout()
   plt.show()
  else:
   fig.tight_layout()
   fnam = 'crossCorrOmgPiFGBTh.pdf'
   plt.savefig(fnam)
   print('Written file', fnam)
   print('=============================================================================================')
   fig.clf()
   case = case + 1


   if case == 4:
    break





