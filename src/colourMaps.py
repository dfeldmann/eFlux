# This very cool peace of Python code was taken from github.com/nesanders/colorblind-colormap
# and slightly modified. It provides colour blind friendly colour maps according to
# Wong, B. (2011) Nature Methods 8:441

import matplotlib
import matplotlib.pyplot as plt

CBcdict={
    #'Bl':(0,0,0),
    #'Or':(.9,.6,0),
    #'SB':(.35,.7,.9),
    #'bG':(0,.6,.5),
    #'Ye':(.95,.9,.25),
    #'Bu':(0,.45,.7),
    #'Ve':(.8,.4,0),
    #'rP':(.8,.6,.7),
    # updated acc. to http://mkweb.bcgsc.ca/colorblind/img/colorblindness.palettes.trivial.png
    'Bl':(0.00000, 0.00000, 0.000000), # Black            '#000000'
    'Or':(0.90196, 0.62353, 0.000000), # Orange           '#E69F00'
    'SB':(0.33725, 0.70588, 0.913730), # Sky Blue         '#56B4E9'
    'bG':(0.00000, 0.61961, 0.450980), # bluish Green     '#009E73'
    'Ye':(0.94118, 0.89412, 0.258820), # Yellow           '#F0E442'
    'Bu':(0.00000, 0.44706, 0.698040), # Blue             '#0072B2'
    'Ve':(0.83529, 0.36863, 0.000000), # Vermillion       '#D55E00' 
    'rP':(0.80000, 0.47451, 0.654900), # reddish Purple   '#CC79A7'
}

# Single color gradient maps
def lighter(colors):
 li=lambda x: x+.5*(1-x)
 return (li(colors[0]),li(colors[1]),li(colors[2]))

def darker(colors):
 return (.5*colors[0],.5*colors[1],.5*colors[2])

CBLDcm={}
for key in CBcdict:
 CBLDcm[key]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key,[lighter(CBcdict[key]),darker(CBcdict[key])])

# Two color gradient maps
CB2cm={}
for key in CBcdict:
 for key2 in CBcdict:
  if key!=key2: CB2cm[key+key2]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key+key2,[CBcdict[key],CBcdict[key2]])

# Two color gradient maps with white in the middle
CBWcm={}
for key in CBcdict:
 for key2 in CBcdict:
  if key!=key2: CBWcm[key+key2]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key+key2,[CBcdict[key],(1,1,1),CBcdict[key2]])

# Two color gradient maps with Black in the middle
CBBcm={}
for key in CBcdict:
 for key2 in CBcdict:
  if key!=key2: CBBcm[key+key2]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key+key2,[CBcdict[key],(0,0,0),CBcdict[key2]])

# Change default color cycle
# matplotlib.rcParams['axes.color_cycle'] = [CBcdict[c] for c in sorted(CBcdict.keys())]
# matplotlib.rcParams['axes.prop_cycle'] = [CBcdict[c] for c in sorted(CBcdict.keys())]
