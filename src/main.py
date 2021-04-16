###############################################################################
# Description	:
# Args          :
#     wdir = path to directory in which save data
# Author        : Julia Neme
# Email         : j.neme@unsw.edu.au
###############################################################################

import analysis_tools as at
import cartopy.crs as ccrs
import cmocean
import dask.distributed as dsk
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import plot_tools as pt
import xarray as xr
from glob import glob
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

clnt = dsk.Client()
keys = ['1', '025', '01']
wdir = input("Directory where to save and look for: ")

###############################################################################

def plot_figure_1():

    bath = xr.open_dataset(wdir+'/ocean_grid-01deg.nc')['ht']
    iso1 = xr.open_dataset(wdir+'/isobath_1000m.nc')
    aice = xr.open_dataset(wdir+'/aice_m-monthly-1958_2018-01deg.nc')['aice_m']
    aice = aice.groupby('time.month').mean(dim = 'time')
    aice_obs_feb = at.g02202_aice('feb')
    aice_obs_sep = at.g02202_aice('sep')
    hyd_lat, hyd_lon = at.hyd_lat_lon()
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
    bath_cmap = pt.custom_colormaps('bathymetry')

    fig, axs = pt.map_weddell(190, 95)
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
    axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1.2,
             transform = ccrs.PlateCarree())
    axs.plot([-60, 10, 10, -60, -60], [-75, -75, -57, -57, -75],
             color = 'orange', linewidth = 1, linestyle = '--',
             transform = ccrs.PlateCarree(), zorder = 1)
    axs.plot([-60, 50, 50, -60, -60], [-75, -75, -57, -57, -75],
             color = 'orange', linewidth = 1, transform = ccrs.PlateCarree(),
             zorder = 1)
    axs.plot(hyd_lon, hyd_lat, color = 'none', markersize = 2, marker = 'o',
             markeredgecolor = 'k', markerfacecolor = 'None',
             markeredgewidth = .2, transform = ccrs.PlateCarree(), zorder = 2)
    axs.scatter(a12_lon, a12_lat, s = 12, c = 'white', edgecolor = 'k',
                marker = 'o', transform = ccrs.PlateCarree(), zorder = 3)
    axs.legend(handles = legend_elements, ncol = 2, loc = 'lower center',
               bbox_to_anchor = (0.68, 0.03), frameon = False)
    plt.savefig(wdir+'/figure_1.jpg')

def plot_figure_2():

    etam, etao = at.load_sea_level(keys, wdir)
    for k in keys:
        etam[k] = etam[k].mean(dim = 'time')
    etao = etao['MDT']
    bndy = at.gyre_boundary(keys, wdir, 'mean')
    iso1 = xr.open_dataset(wdir+'/isobath_1000m.nc')

    fig, axs = pt.map_weddell(190, 95)
    c = axs.contourf(etao['xt_ocean'], etao['yt_ocean'], etao,
                     levels = np.arange(-216, -130, 2),
                     cmap = cmocean.cm.tarn_r, extend = 'max',
                     transform = ccrs.PlateCarree())
    axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
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
        fig, axs = pt.map_weddell(190, 95)
        c = axs.contourf(etam[k]['xt_ocean'], etam[k]['yt_ocean'],
                         etam[k]-etao, levels = np.arange(-30, 31, 1),
                         cmap = 'RdBu_r', extend = 'both',
                         transform = ccrs.PlateCarree())
        axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
                 transform = ccrs.PlateCarree())
        axs.plot(bndy[k][:,0], bndy[k][:,1], color = 'k', linestyle = 'solid',
                 linewidth = 1.5, transform = ccrs.PlateCarree())
        axs.text(0.98, 0.08, k+'$^{\circ}$', horizontalalignment = 'right',
                 transform = axs.transAxes,
                 bbox = dict(boxstyle = 'round', facecolor = 'white'));
        axs.text(-0.08, 0.93, t+')', horizontalalignment = 'right',
                 transform = axs.transAxes);
        cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink =.6)
        cbar.set_label('Sea level [cm]')
        plt.savefig(wdir+'/figure_2'+t+'.jpg')

def plot_figure_3():

    etam, etao = at.load_sea_level(keys, wdir)
    bndy = at.gyre_boundary(keys, wdir, 'mean')
    iso1 = xr.open_dataset(wdir+'/isobath_1000m.nc')
    rval = {}; pval = {}
    for k, t in zip(keys, ['a', 'b', 'c']):
        etao['time'] = etam[k]['time'].values
        rval[k], pval[k] = at.linear_correlation(etao['DOT'], etam[k])

        fig, axs = pt.map_weddell(190, 95)
        c = axs.contourf(rval[k]['xt_ocean'], rval[k]['yt_ocean'], rval[k],
                         levels = np.arange(-1, 1.1, .1), cmap = 'RdBu_r',
                         transform = ccrs.PlateCarree())
        axs.contourf(pval[k]['xt_ocean'], pval[k]['yt_ocean'],
                     pval[k].where(pval[k] < 0.05), colors = ['none'],
                     hatches = ['xx'], transform = ccrs.PlateCarree())
        axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
                 transform = ccrs.PlateCarree())
        axs.plot(bndy[k][:,0], bndy[k][:,1], color = 'k', linestyle = 'solid',
                 linewidth = 1.5, transform = ccrs.PlateCarree())
        axs.text(0.98, 0.08, k+'$^{\circ}$', horizontalalignment = 'right',
                 transform = axs.transAxes,
                 bbox = dict(boxstyle = 'round', facecolor = 'white'));
        axs.text(-0.08, 0.93, t+')', horizontalalignment = 'right',
                 transform = axs.transAxes);
        cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink =.6)
        cbar.set_label('Correlation coefficient')
        plt.savefig(wdir+'/figure_3'+t+'.jpg')

    RMSE = {}
    for k in keys:
        mod = etam[k].sel(xt_ocean = slice(-30, 30), yt_ocean = slice(-70, -60))
        etao['time'] = etam[k]['time'].values
        obs = etao['DOT'].sel(xt_ocean = slice(-30, 30),
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
    plt.savefig(wdir+'/figure_3d.jpg')

def plot_figure_4():

    #### TS diagrams
    ts_hydro = at.load_temp_salt_hydrography()
    hydr_cmap = pt.custom_colormaps('ts_diags')

    fig, ax, d_tag = pt.ts_diagram(ts_hydro)
    sc = ax.scatter(ts_hydro['salt'], ts_hydro['pot_temp'], c = d_tag,
                    s = 0.1, cmap = hydr_cmap);
    cbar = fig.colorbar(sc, ax = ax, orientation = 'vertical', shrink = .8,
                        format = mticker.ScalarFormatter());
    cbar.set_label('Depth [m]')
    cbar.ax.invert_yaxis()
    cbar.set_ticks([0, .8, 1.6, 2.4, 3.2])
    cbar.set_ticklabels([0, 200, 500, 1000, 3000])
    ax.text(0.03, 0.93, 'Observations', horizontalalignment = 'left',
            transform = ax.transAxes,
            bbox = dict(boxstyle = 'round', facecolor = 'white'));
    ax.text(-0.15, 0.95, 'a)', horizontalalignment = 'left',
            transform = ax.transAxes);
    plt.savefig(wdir+'/figure_4_a_ts.jpg')

    for k, t in zip(keys, ['b', 'c', 'd']):
        ts_model = at.load_temp_salt_model(k, wdir, ts_hydro)
        fig, ax, d_tag = pt.ts_diagram(ts_model)
        sc = ax.scatter(ts_model['salt'], ts_model['pot_temp'],
                        c = d_tag, s = 0.1, cmap = hydr_cmap);
        cbar = fig.colorbar(sc, ax = ax, orientation = 'vertical', shrink = .8,
                            format = mticker.ScalarFormatter());
        cbar.set_label('Depth [m]')
        cbar.ax.invert_yaxis()
        cbar.set_ticks([0, .8, 1.6, 2.4, 3.2])
        cbar.set_ticklabels([0, 200, 500, 1000, 3000])
        ax.text(0.03, 0.93, k+'$^{\circ}$', horizontalalignment = 'left',
                transform = ax.transAxes,
                bbox = dict(boxstyle = 'round', facecolor = 'white'));
        ax.text(-0.15, 0.95, t+')', horizontalalignment = 'left',
                transform = ax.transAxes);
        plt.savefig(wdir+'/figure_4_'+t+'_ts.jpg')

    #### A12
    a12_hydro = analysis_tools.a12_hydrography()
    a12_averg = a12_hydro.mean(dim = 'cruise')
    a12_averg_pot_rho = analysis_tools.potential_density(a12_averg)

###############################################################################

def main():

    clrs, alph = pt.set_rcParams()

    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()

if __name__ == "__main__":
    main()
