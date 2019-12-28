#!/usr/bin/env python3
# Purpose:  Simple plot to define/visualise quadrant (Q) events in pipe flow,
#           i.e. in a cylindrical co-ordiante system
# Usage:    python qEventsPipe.py 
# Authors:  Daniel Feldmann
# Date:     22nd August 2019
# Modified: 22nd August 2019


print('Creating plot (using LaTeX)...')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
r"\usepackage[utf8x]{inputenc}",
r"\usepackage[T1]{fontenc}",
r"\usepackage[detect-all]{siunitx}",
r'\usepackage{amsmath, amstext, amssymb}',
r'\usepackage{xfrac}',
r'\usepackage{lmodern, palatino, eulervm}']
mpl.rcParams.update({'font.family': 'sans-serif'})
mpl.rcParams.update({'font.family': 'serif'})
mpl.rcParams.update({'font.size': 9})
mpl.rcParams.update({'lines.linewidth': 1.5})
mpl.rcParams.update({'axes.linewidth': 1.5})
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.minor.size'] = 1
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.size'] = 1
mpl.rcParams['ytick.minor.width'] = 1.5

# create figure suitable for A4 format
def mm2inch(*tupl):
 inch = 25.4
 if isinstance(tupl[0], tuple):
  return tuple(i/inch for i in tupl[0])
 else:
   return tuple(i/inch for i in tupl)
fig = plt.figure(num=None, figsize=mm2inch(80.0, 80.0), dpi=300)

# conservative colour palette appropriate for colour-blind (http://mkweb.bcgsc.ca/colorblind/)
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Black         = '#000000'

# modify box for filter name annotation
filterBox = dict(boxstyle="square, pad=0.3", fc='w', ec=Black, lw=1.5)

# main plot with everything off and centerd spine
ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
ax1.set_xlabel(r"$u_z$", x=0.90, rotation='horizontal')
ax1.set_xlim([-1.3, 1.3])
ax1.set_xticks([])
# ax1.set_ylabel(r"$u_r$", y=0.10, rotation='horizontal')
t0 = ax1.text(-0.22, 1.11, r"$u_r$")
ax1.set_ylim([1.3, -1.3])
ax1.set_yticks([])
ax1.set_aspect('equal')
ax1.tick_params(axis='both', direction='inout', length=6.0)
ax1.spines['left'].set_position('center')
ax1.spines['left'].set_color('none')
ax1.spines['right'].set_color('none')
ax1.spines['bottom'].set_position('center')
ax1.spines['bottom'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.spines['left'].set_smart_bounds(True)
ax1.spines['bottom'].set_smart_bounds(True)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

# handmade axes with arrow heads
ax1.arrow(-1.2, 0.0, 2.3, 0.0, head_width=0.08, head_length=0.18, linewidth=1.5, fc='k', ec='k')
ax1.arrow( 0.0,-1.2, 0.0, 2.3, head_width=0.08, head_length=0.18, linewidth=1.5, fc='k', ec='k')

# quadrants
t1 = ax1.text(+0.65, -0.65, r" \begin{center} $Q_1$\\ Outw. interact.\\ $u^\prime_z>0$\\ $u_r<0$\\ $u_r u^\prime_z<0$ \end{center}", ha="center", va="center", color='gray') #Blue)
t2 = ax1.text(-0.65, -0.65, r" \begin{center} $Q_2$\\ Ejection\\ $u^\prime_z<0$\\ $u_r<0$\\ $u_r u^\prime_z>0$ \end{center}",        ha="center", va="center", color=Blue)   # 'gray')
t3 = ax1.text(-0.65, +0.65, r" \begin{center} $Q_3$\\ Inw. interact.\\ $u^\prime_z<0$\\ $u_r>0$\\ $u_r u^\prime_z<0$ \end{center}",  ha="center", va="center", color='gray') # Vermillion)
t4 = ax1.text(+0.65, +0.65, r" \begin{center} $Q_4$\\ Sweep\\ $u^\prime_z>0$\\ $u_r>0$\\ $u_r u^\prime_z>0$ \end{center}",           ha="center", va="center", color=Vermillion) #'gray')

plot = 2
# plot mode interactive or pdf
if plot != 2:
 #plt.tight_layout()
 plt.show()
else:
 #fig.tight_layout()
 fnam = 'qEventsPipe.pdf'
 plt.savefig(fnam)
 print('Written file', fnam)
fig.clf()
