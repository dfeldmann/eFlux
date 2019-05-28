#!/usr/bin/env python3
#====================================================================================
# Purpose:  Computes the auto-correlation coefficient between the axial fluctuating 
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
# Usage:    python autocorrFourierZ.py 
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
fnamZF = 'z/autoCorr1dFourierZ00570000to01265000nt0140.dat'
fnamZG = 'z/autoCorr1dGaussZ00570000to01265000nt0140.dat'
fnamZB = 'z/autoCorr1dBoxZ00570000to01265000nt0140.dat'
print('-----------------------------------------------------------------')
print('Reading auto-correlations from', fnamZF)
print('Reading auto-correlations from', fnamZG)
print('Reading auto-correlations from', fnamZB)

dz = np.loadtxt(fnamZF)[:,0]

acUz = np.loadtxt(fnamZF)[:,1]
acOmg= np.loadtxt(fnamZF)[:,4]

acUzZF  = np.loadtxt(fnamZF)[:,2]
acPiZF  = np.loadtxt(fnamZF)[:,3]

acUzZG  = np.loadtxt(fnamZG)[:,2]
acPiZG  = np.loadtxt(fnamZG)[:,3]

acUzZB  = np.loadtxt(fnamZB)[:,2]
acPiZB  = np.loadtxt(fnamZB)[:,3]
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
  ax1.set_xlabel(r"$z^+$ in $R$")
  ax1.set_ylabel(r"$C_{u^{\prime}_zu^{\prime}_z}$")
  ax1.plot(dz*180.4, acUz,   color=Black, linestyle='-', label='Unfiltered')
  ax1.plot(dz*180.4, acUzZF, color=Blue, linestyle='-', label='Fourier')
  ax1.plot(dz*180.4, acUzZG, color=Vermillion, linestyle='-', label='Gauss')
  ax1.plot(dz*180.4, acUzZB, color=BluishGreen, linestyle='-', label='Box')
  ax1.legend(loc='best', ncol=4)
  #plt.xlim(left=0,right=1800)
  ax1.axhline(0, color='black', lw=0.5, linestyle='--')
  ax1.axvline(0, color='black', lw=0.5, linestyle='--')
  ax1.legend(loc='best')
  #ax1.get_legend().remove()
  # plot mode interactive or pdf
 
 
  if plot != 2:
   plt.tight_layout()
   plt.show()
  else:
   fig.tight_layout()
   fnam = 'autoCorrUzZ.pdf'
   plt.savefig(fnam)
   print('Written file', fnam)
   print('=============================================================================================')
   fig.clf()
   case = case + 1

 if case == 1:
  fig = plt.figure(num=None, figsize=mm2inch(160.0, 140.0), dpi=600)
  ax2 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
  ax2.set_xlabel(r"$z^+$ in $R$")
  ax2.set_ylabel(r"$C_{\omega_z\omega_z}$")
  ax2.plot(dz*180.4, acOmg, color=BluishGreen, linestyle='-')
  ax2.legend(loc='best', ncol=4)
  #plt.xlim(left=0,right=1800)
  ax2.axhline(0, color='black', lw=0.5, linestyle='--')
  ax2.axvline(0, color='black', lw=0.5, linestyle='--')
  ax2.legend(loc='best')
  ax2.get_legend().remove()
  if plot != 2:
   plt.tight_layout()
   plt.show()
  else:
   fig.tight_layout()
   fnam = 'autoCorrOmgZ.pdf'
   plt.savefig(fnam)
   print('Written file', fnam)
   print('=============================================================================================')
   fig.clf()
   case = case + 1

 if case == 2:
  fig = plt.figure(num=None, figsize=mm2inch(160.0, 140.0), dpi=600)
  ax3 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
  ax3.set_xlabel(r"$z^+$ in $R$")
  ax3.set_ylabel(r"$C_{\tilde{\pi}\tilde{\pi}}$")
  ax3.plot(dz*180.4, acPiZF, color=Blue, linestyle='-', label='Fourier')
  ax3.plot(dz*180.4, acPiZG, color=Vermillion, linestyle='-', label='Gauss')
  ax3.plot(dz*180.4, acPiZB, color=BluishGreen, linestyle='-', label='Box')
  ax3.legend(loc='best', ncol=4)
  #plt.xlim(left=0,right=1800)
  ax3.axhline(0, color='black', lw=0.5, linestyle='--')
  ax3.axvline(0, color='black', lw=0.5, linestyle='--')
  ax3.legend(loc='best')

  if plot != 2:
   plt.tight_layout()
   plt.show()
  else:
   fig.tight_layout()
   fnam = 'autoCorrPiZ.pdf'
   plt.savefig(fnam)
   print('Written file', fnam)
   print('=============================================================================================')
   fig.clf()
   case = case + 1
   if case == 3:
    break





