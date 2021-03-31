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
    axs.set_yticks([-75, -70, -65, -60, -55]], crs = ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label = True)
    lat_formatter = LatitudeFormatter()
    axs.xaxis.set_major_formatter(lon_formatter)
    axs.yaxis.set_major_formatter(lat_formatter)

    return fig, axs

def curstom_colormaps(name):

    import matplotlib.colors as mcolors

    if name == 'bathymetry':
        cmap_colors = ['#EDF3F3', '#D8E6E7', '#C5DADB', '#B1CDCE', '#A2BEC1',
                       '#95AEB5', '#899DA8', '#7C8C9C', '#707B8F', '#636A84',
                       '#565A77', '#4A4968']
        cmap = mcolors.LinearSegmentedColormap.from_list(name = None,
               colors = cmap_colors, N = 250, gamma = 1)

    return cmap
