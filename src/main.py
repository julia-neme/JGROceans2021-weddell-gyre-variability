###############################################################################
# Description	:
# Args          :
#     wdir = path to directory in which save data
# Author        : Julia Neme
# Email         : j.neme@unsw.edu.au
###############################################################################

import cmocean
import dask.distributed as dsk
import plot_tools
import xarray as xr

clnt = dsk.Client()
keys = ['1', '025', '01']
wdir = input("Please directory where to save the data: ")

###############################################################################

def plot_figure_1():

    from matplotlib.lines import Line2D

    bath = xr.open_dataset(wdir+'/ocean_grid-01deg.nc')['ht']
    isobath_1000m = xr.open_dataset(wdir+'/isobath_1000m.nc')
    aice = xr.open_dataset(wdir+'/aice_m-monthly-1958_2018-01deg.nc')['aice_m']
    aice = aice.groupby('time.month').mean(dim = 'time')

    aice_obs_feb = g02202_february_aice() # NEED TO DEFINE
    aice_obs_sep = g02202_september_aice()

    hydr_lat, hydr_lon = hydr_lat_lon() # NEED TO DEFINE
    a12_lat = np.array([-69.        , -68.26315789, -67.52631579, -66.78947368,
                        -66.05263158, -65.31578947, -64.57894737, -63.84210526,
                        -63.10526316, -62.36842105, -61.63157895, -60.89473684,
                        -60.15789474, -59.42105263, -58.68421053, -57.94736842,
                        -57.21052632, -56.47368421, -55.73684211, -55.        ])
    a12_lon = np.zeros(np.shape(a12_lat))

    legend_elements = [
         Line2D([0], [0], ls = '', lw = 0.5, marker = 'o',
         markerfacecolor = 'white', markeredgecolor = 'k', markersize = 6,
         label = 'A12'),
         Line2D([0], [0], ls = '', lw = 0.5, marker = 'o',
         markerfacecolor = 'white', markeredgecolor = 'k', markeredgewidth = .2,
         markersize = 4,  label = 'Hydrography'),
         Line2D([0], [0], color = 'blue', lw = 1.5, label = 'February'),
         Line2D([0], [0], color = 'red', lw = 1.5, label = 'September')]
    bath_cmap = plot_tools.curstom_colormaps('bathymetry')
    
    fig, axs = plot_tools.map_weddell(190, 95)
    c = axs.contourf(bath['xt_ocean'], bath['yt_ocean'], bath, cmap = bath_cmap,
                      levels = np.arange(0, 6500, 500),
                      transform = ccrs.PlateCarree())
    axs.contour(aice['xt_ocean'], aice['yt_ocean'], aice.isel(month = 1),
                levels = [0.15], colors = ['blue'], linewidths = 1,
                transform = ccrs.PlateCarree())
    axs.contour(aice['xt_ocean'], aice['yt_ocean'], aice.isel(month = 9),
                levels = [0.15], colors = ['red'], linewidths = 1,
                transform = ccrs.PlateCarree())
    axs.plot(aice_obs_feb[0,:], aice_obs_feb[1,:], color = 'blue',
             linestyle = '--', linewidth = 1, transform = ccrs.PlateCarree())
    axs.plot(aice_obs_sep[0,:], aice_obs_sep[1,:], color = 'red',
             linestyle = '--', linewidth = 1, transform = ccrs.PlateCarree())
    axs.plot(isobath_1000m['x'], isobath_1000m['y'], color = 'k',
             linewidth = 1.2, transform = ccrs.PlateCarree())
    axs.plot([-60, 10, 10, -60, -60], [-75, -75, -57, -57, -75],
             color = 'orange', linewidth = 1, linestyle = '--',
             transform = ccrs.PlateCarree(), zorder = 1)
    axs.plot([-60, 50, 50, -60, -60], [-75, -75, -57, -57, -75],
             color = 'orange', linewidth = 1, transform = ccrs.PlateCarree(),
             zorder = 1)
    axs.scatter(a12_lon, a12_lat, s = 12, c = 'white', edgecolor = 'k',
                marker = 'o', transform = ccrs.PlateCarree(), zorder = 3)
    axs.plot(hydr_lon, hydr_lat, color = 'none', markersize = 2,
             markeredgecolor = 'k', markerfacecolor = 'None', marker = 'o',
             markeredgewidth = .2, transform = ccrs.PlateCarree(), zorder = 2)
    axs.legend(handles = legend_elements, ncol = 2, loc = 'lower center',
               bbox_to_anchor = (0.68, 0.03), frameon = False)

    plt.savefig(wdir+'figure_1.jpg')
