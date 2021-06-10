###############################################################################
# Description	: Plotting functions
# Args          :
#     wdir = path to directory in which save data
# Author        : Julia Neme
# Email         : j.neme@unsw.edu.au
###############################################################################

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

###############################################################################

def set_rcParams():

    # Font
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.sans-serif'] = 'helvetica'
    # Font size
    plt.rcParams['font.size'] = 8
    # Axes
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    # Saving parameters
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    # Default colors for each model resolution
    clrs = {'1':'k', '025':'r', '01':'b'}

    return clrs

def custom_colormaps(name):

    import matplotlib.colors as mcolors
    import numpy as np

    if name == 'bathymetry':
        cmap_colors = ['#EDF3F3', '#D8E6E7', '#C5DADB', '#B1CDCE', '#A2BEC1',
                       '#95AEB5', '#899DA8', '#7C8C9C', '#707B8F', '#636A84',
                       '#565A77', '#4A4968']
        cmap = mcolors.LinearSegmentedColormap.from_list(name = None,
               colors = cmap_colors, N = 250, gamma = 1)

    elif name == 'ts_diags':
        cmap = mcolors.LinearSegmentedColormap.from_list(name = None,
               colors = plt.get_cmap('Blues')(np.linspace(.4, 1, 5)), N = 5)

    elif name == 'subsfc_tmax':
        cmap_colors =  ['#353992', '#7294C2', '#A5C4DD', '#F9FCCF', '#F2CF85',
                        '#CB533B']
        cmap = mcolors.LinearSegmentedColormap.from_list(name = None,
               colors = cmap_colors, N = 250, gamma = .8)

    return cmap

def map_weddell(size_x, size_y):

    fig = plt.figure(figsize = (size_x/25.4, size_y/25.4))
    axs = fig.add_subplot(projection = ccrs.Mercator(central_longitude = 5))
    axs.set_extent([-70, 80, -77, -50], crs = ccrs.PlateCarree())
    axs.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
                    edgecolor = 'black', facecolor = 'white'))
    axs.set_xticks([-70, -50, -30, -10, 10, 30, 50, 70],
                   crs = ccrs.PlateCarree())
    axs.set_yticks([-75, -70, -65, -60, -55], crs = ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label = True)
    lat_formatter = LatitudeFormatter()
    axs.xaxis.set_major_formatter(lon_formatter)
    axs.yaxis.set_major_formatter(lat_formatter)

    return fig, axs

def ts_diagram(dset):

    import gsw
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize = (95/25.4, 80/25.4))
    ax.plot([32.5, 35.25], [1.5, 1.5], color = 'k', linewidth = 1)
    ax.plot([34, 34, 35, 35], [-2.5, -1.9, -1.9, -2.5], color = 'k', 
            linewidth = 1)
    ax.plot([34.57, 34.57], [3, 1.5], color = 'k', linewidth = 1)
    ax.plot([34.57, 34.57, 35.25], [1.5, 0, 0], color = 'k', linewidth = 1)
    ax.plot([34.63, 34.63, 35.25], [0, -0.7, -0.7], color = 'k', linewidth = 1)
    ax.plot([34.63, 34.6, 34.6, 35.25], [-0.7, -0.7, -1.5, -1.5], color = 'k',
            linewidth = 1)
    ax.text(32.55, 1.55, 'ACC')
    ax.text(32.55, -1.75, 'SW')
    ax.text(34.05, -2.15, 'ISW')
    ax.text(35.2, 1.55, 'CDW', horizontalalignment = 'right')
    ax.text(35.2, 0.05, 'WDW', horizontalalignment = 'right')
    ax.text(35.2, -0.65, 'WSDW', horizontalalignment = 'right')
    ax.text(35.2, -1.45, 'WSBW', horizontalalignment = 'right')
    ax.set_xlim(32.5, 35.25)
    ax.set_ylim(-2.5, 3)
    ax.set_xlabel('Salinity [PSU]')
    ax.set_ylabel('$\\theta$ [$^{\circ}$C]')

    temp = np.arange(-3, 4, .1)
    salt = np.arange(31, 36, .01)
    temp_mesh, salt_mesh = np.meshgrid(temp, salt)
    sigma = gsw.sigma0(salt_mesh, temp_mesh)

    cc = ax.contour(salt_mesh, temp_mesh, sigma, 10, colors = 'k',
                    interpolation = 'none', linewidths = .7, zorder = 0)
    ax.clabel(cc, inline = True, fontsize = 7)

    depth = gsw.z_from_p(dset['pressure'], -65).values
    depth = np.transpose(depth*np.ones([len(dset['station']),
                                        len(dset['pressure'])]))
    depth_tag  = np.empty(np.shape(depth))
    depth_tag[(depth > -200)] = 0
    depth_tag[(depth <= -200) & (depth > -5000)] = 1
    depth_tag[(depth <= -500) & (depth > -1000)] = 2
    depth_tag[(depth <= -1000) & (depth > -3000)] = 3
    depth_tag[(depth <= -3000)] = 4
    depth_tag = depth_tag.astype('int')

    return fig, ax, depth_tag

def a12_section():

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize = (95/25.4, 65/25.4),
                           gridspec_kw = {'height_ratios':[0.5, 1],
                                          'hspace':.03})
    axs[0].set_ylim(-250, 0)
    axs[0].set_xticks([-67, -62, -57])
    axs[0].set_xticklabels([])
    axs[0].set_yticks([-250, -150, -50, 0])
    axs[0].set_yticklabels([250, 150, 50, 0])
    axs[1].set_ylabel('Pressure [dbar]', loc = 'top')
    axs[1].set_ylim(-6000, -250)
    axs[1].set_xticks([-67, -62, -57])
    axs[1].set_xticklabels(['67$^{\circ}$S', '62$^{\circ}$S', '57$^{\circ}$S'])
    axs[1].set_yticks([-6000, -4000, -2000])
    axs[1].set_yticklabels([6000, 4000, 2000])

    return fig, axs

def shiftedColorMap(cmap, min_val, max_val, name):

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val)
    midpoint = 1.0 - max_val/(max_val + abs(min_val))
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    reg_index = np.linspace(start, stop, 257)
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint = False), 
                             np.linspace(midpoint, 1, 129, endpoint = True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) 
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = mcolors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap = newcmap)

    return newcmap

def annual_cycles():

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    lgnd = [Line2D([0], [0], color = 'k', lw = 1, label = '1$^{\circ}$'),
            Line2D([0], [0], color = 'r', lw = 1, label = '0.25$^{\circ}$'),
            Line2D([0], [0], color = 'b', lw = 1, label = '0.1$^{\circ}$')]

    fig = plt.figure(figsize = (100/25.4, 100/25.4))
    axs = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]
    for ax, t in zip(axs, ['a)', 'b)', 'c)']):
        ax.set_xlim(0, 11)
        ax.set_xticks(np.arange(0, 12, 1))
        if ax == axs[-1]:
            ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 
                                'O', 'N', 'D'])
        else:
            ax.set_xticklabels([])
        ax.text(-0.15, 0.95, t, horizontalalignment = 'left', 
                transform = ax.transAxes)
    axs[1].legend(handles = lgnd, ncol = 3, bbox_to_anchor = (0.5, 2.2),
                  loc = 'lower center', frameon = False)
    axs[0].set_ylabel('Strength [Sv]', fontsize = 6)
    axs[1].set_ylabel('$| \\nabla \\times \\tau |$ \n [10$^{-7}$ N m$^{-3}$]',
                      fontsize = 6)
    axs[2].set_ylabel('$\\mathcal{B}$ \n [10$^{-8}$ m$^{2}$  s$^{-3}$]',
                      fontsize = 6)

    return fig, axs

def interannual_time_series():

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator

    lgnd = [Line2D([0], [0], color = 'k', lw = 1, label = '1$^{\circ}$'),
            Line2D([0], [0], color = 'r', lw = 1, label = '0.25$^{\circ}$'),
            Line2D([0], [0], color = 'b', lw = 1, label = '0.1$^{\circ}$'),
            mpatches.Patch(color = 'mistyrose', label = 'Strong events'),
            mpatches.Patch(color = 'lavender', label = 'Weak events')]

    fig = plt.figure(figsize = (190/25.4, 150/25.4))
    axs = [fig.add_subplot(511), fig.add_subplot(512), fig.add_subplot(513), 
           fig.add_subplot(514), fig.add_subplot(515)]
    plt.subplots_adjust(wspace = 0, hspace = 0)
    for i in range(0, 4):
        axs[i].spines['bottom'].set_visible(False)
        axs[i+1].spines['top'].set_visible(False)
        axs[i].set_xticklabels([])
        axs[i].tick_params(bottom = False)
    for ax, t in zip(axs, ['a)', 'b)', 'c)', 'd)', 'e)']):
        ax.text(0.01, 0.8, t, horizontalalignment = 'left', 
                transform = ax.transAxes, fontsize = 8)
    axs[0].legend(handles = lgnd, ncol = 5, bbox_to_anchor = (0.5, 1.02),
                  loc = 'lower center', frameon = False, fontsize = 8)
    axs[0].set_ylabel('Strength \n [Sv]', fontsize = 8)
    axs[1].set_ylabel('$| \\nabla \\times \\tau |$ \n [10$^{-7}$ N m$^{-3}$]',
                      fontsize = 8)
    axs[2].set_ylabel('$\\mathcal{B}$ \n [10$^{-8}$ m$^{2}$  s$^{-3}$]',
                      fontsize = 8)
    axs[3].set_ylabel('SAM \n index', fontsize = 8)
    axs[4].set_ylabel('EAS \n index', fontsize = 8)
    axs[4].set_xticks(np.arange(12*2, 61*12+12, 12*5))
    axs[4].set_xticklabels(np.arange(1960, 2020, 5))
    axs[4].xaxis.set_minor_locator(AutoMinorLocator(n = 5))

    return fig, axs
