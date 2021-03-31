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

    plt.savefig(wdir+'/figure_1.jpg')

def plot_figure_2():

    eta_mod, eta_obs = load_sea_level(keys)
    gyre_boundary = get_gyre_boundary(keys, 'mean')
    isobath_1000m = xr.open_dataset(wdir+'/isobath_1000m.nc')

    fig, axs = plot_tools.map_weddell(190, 95)
    c = axs.contourf(eta_obs['xt_ocean'], eta_obs['yt_ocean'], eta_obs['MDT'],
                     levels = np.arange(-216, -130, 2),
                     cmap = cmocean.cm.tarn_r, extend = 'max',
                     transform = ccrs.PlateCarree())
    axs.plot(isobath_1000m['x'], isobath_1000m['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree())
    axs.text(0.98, 0.08, 'Observations', horizontalalignment = 'right',
             transform = axs.transAxes, bbox = dict(boxstyle = 'round',
             facecolor = 'white'));
    axs.text(-0.08, 0.93, 'a)', horizontalalignment = 'right',
             transform = axs.transAxes);
    cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink = .6)
    cbar.set_label('Sea level [cm]')
    plt.savefig(wdir+'/figure_2a.jpg')

    for k, t in zip(keys, ['b', 'c', 'd']):
        fig, axs = plot_tools.map_weddell(190, 95)
        c = axs.contourf(eta_mod[k]['xt_ocean'], eta_mod[k]['yt_ocean'],
                         eta_mod[k]-eta_obs, levels = np.arange(-30, 31, 1),
                         cmap = 'RdBu_r', extend = 'both',
                         transform = ccrs.PlateCarree())
        axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
                 transform = ccrs.PlateCarree())
        axs.plot(bndy[k][:,0], bndy[k][:,1], color = 'k', linestyle = 'solid',
                 linewidth = 1.5, transform = ccrs.PlateCarree())
        axs.text(0.98, 0.08, 'ACCESS-OM2-'+k, horizontalalignment = 'right',
                 transform = axs.transAxes,
                 bbox = dict(boxstyle = 'round', facecolor = 'white'));
        axs.text(-0.08, 0.93, t+')', horizontalalignment = 'right',
                 transform = axs.transAxes);
        cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink =.6)
        cbar.set_label('Sea level [cm]')
        plt.savefig(wdir+'/figure_2'+t+'.jpg')

def plot_figure_3():

    import matplotlib.dates as mdates

    eta_mod, eta_obs = load_sea_level(keys)
    gyre_boundary = get_gyre_boundary(keys, 'mean')
    isobath_1000m = xr.open_dataset(wdir+'/isobath_1000m.nc')

    rval = {}; pval = {}
    for k, t in zip(keys, ['a', 'b', 'c']):
        eta_obs['time'] = eta_mod[k]['time'].values
        rval[k], pval[k] = correlation_with_gstr(eta_obs['DOT'], eta_mod[k])

        fig, axs = plot_tools.map_weddell(190, 95)
        c = axs.contourf(rval[k]['xt_ocean'], rval[k]['yt_ocean'], rval[k],
                         levels = np.arange(-1, 1.1, .1), cmap = 'RdBu_r',
                         transform = ccrs.PlateCarree())
        axs.contourf(pval[k]['xt_ocean'], pval[k]['yt_ocean'],
                     pval[k].where(pval[k]<0.05), colors = ['none'],
                     hatches = ['xx'], transform = ccrs.PlateCarree())
        axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
                 transform = ccrs.PlateCarree())
        axs.plot(bndy[k][:,0], bndy[k][:,1], color = 'k', linestyle = 'solid',
                 linewidth = 1.5, transform = ccrs.PlateCarree())
        axs.text(0.98, 0.08, 'ACCESS-OM2-'+k, horizontalalignment = 'right',
                 transform = axs.transAxes,
                 bbox = dict(boxstyle = 'round', facecolor = 'white'));
        axs.text(-0.08, 0.93, t+')', horizontalalignment = 'right',
                 transform = axs.transAxes);
        cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink =.6)
        cbar.set_label('Correlation coefficient')
        plt.savefig(wdir+'/figure_3'+t+'.jpg')

    RMSE = {}
    for k in keys:
        mod = eta_mod[k].sel(xt_ocean = slice(-30, 30),
                             yt_ocean = slice(-70, -60))
        # Make same time axis (if not align fails)
        eta_obs['time'] = eta_mod[k]['time'].values
        obs = eta_obs['DOT'].sel(xt_ocean = slice(-30, 30),
                                 yt_ocean = slice(-70, -60))
        mod_a = mod - mod.mean(dim = 'time')
        obs_a = obs - obs.mean(dim = 'time')
        RMSE[k] = ((mod_a - obs_a)**2).mean(dim = ['xt_ocean', 'yt_ocean'])

    fig = plt.figure(figsize = (150/25.4, 65/25.4))
    axs = fig.add_subplot()
    for k in keys:
        axs.plot(RMSE[k]['time'], np.sqrt(RMSE[k]), color = clrs[k],
                 alpha = alph[k], linewidth = 1.5, label = 'ACCESS-OM2-'+k)
    axs.xaxis.set_minor_locator(mdates.MonthLocator())
    axs.set_ylabel('RMSE [cm]');
    plt.legend(frameon = False);
    axs.text(-0.05, 0.93, 'd)', horizontalalignment = 'right',
             transform = axs.transAxes);
    plt.savefig(wdir+'/figre_3d.jpg')
