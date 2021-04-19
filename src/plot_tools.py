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
    clrs = {'1':'brown', '025':'darkgoldenrod', '01':'navy'}
    alph = {'1':1, '025':1, '01':.7}

    return clrs, alph

def custom_colormaps(name):

    import matplotlib.colors as mcolors

    if name == 'bathymetry':
        cmap_colors = ['#EDF3F3', '#D8E6E7', '#C5DADB', '#B1CDCE', '#A2BEC1',
                       '#95AEB5', '#899DA8', '#7C8C9C', '#707B8F', '#636A84',
                       '#565A77', '#4A4968']
        cmap = mcolors.LinearSegmentedColormap.from_list(name = None,
               colors = cmap_colors, N = 250, gamma = 1)
    elif name == 'ts_diags':
        cmap = mcolors.LinearSegmentedColormap.from_list(name = None,
               colors = plt.get_cmap('Blues')(np.linspace(.4, 1, 5)), N = 5)

    return cmap

def map_weddell(size_x, size_y):

    """
    Args          :
         size_x, size_y = figure sizes in mm.
    """

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
    ax.plot([32.5, 35.25], [-1.9, -1.9], color = 'k', linewidth = 1)
    ax.plot([34.57, 34.57], [3, 1.5], color = 'k', linewidth = 1)
    ax.plot([34.57, 34.57, 35.25], [1.5, 0, 0], color = 'k', linewidth = 1)
    ax.plot([34.63, 34.63, 35.25], [0, -0.7, -0.7], color = 'k', linewidth = 1)
    ax.plot([34.63, 34.6, 34.6, 35.25], [-0.7, -0.7, -1.5, -1.5], color = 'k',
            linewidth = 1)
    ax.text(32.55, 1.55, 'ACC')
    ax.text(32.55, -1.75, 'SW')
    ax.text(32.55, -2.15, 'ISW')
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
                    interpolation = 'none', linewidths = .7, zorder = 0);
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
    fig, axs = plt.subplots(2, 1, figsize = (95/25.4, 75/25.4),
                           gridspec_kw = {'height_ratios':[0.5, 1],
                                          'hspace':.03})
    axs[0].set_ylim(-250, 0)
    axs[0].set_xticks([-67, -62, -57])
    axs[0].set_xticklabels([])
    axs[0].set_yticks([-250, 0])
    axs[0].set_yticklabels([250, 0])
    axs[1].set_ylim(-6000, -250)
    axs[1].set_xticks([-67, -62, -57])
    axs[1].set_xticklabels(['67$^{\circ}$S', '62$^{\circ}$S', '57$^{\circ}$S'])
    axs[1].set_yticks([-6000, -4000, -2000])
    axs[1].set_yticklabels([6000, 4000, 2000])

    return fig, axs
