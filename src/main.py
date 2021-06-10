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
import matplotlib.ticker as mticker
import numpy as np
import os
import plot_tools as pt
import scipy.stats as st
import xarray as xr
from glob import glob
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

import warnings # ignore these warnings
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = RuntimeWarning)

clnt = dsk.Client()
clrs = pt.set_rcParams()
keys = ['1', '025', '01']
# Put working directory (with the data stored in a subdirectory data) 
wdir = ''

###############################################################################

def plot_figure_1():

    bath = xr.open_dataset(wdir+'/data/ocean_grid-01deg.nc')['ht']
    bath = bath.sel(xt_ocean = slice(-70, 80), yt_ocean = slice(None, -50))
    iso1 = xr.open_dataset(wdir+'/data/isobath_1000m.nc')
    aice = xr.open_dataset(wdir+'/data/aice_m-monthly-1958_2018-01deg.nc')['aice_m']
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
    psi_b = xr.open_dataset(wdir+'/data/psi_b-monthly-1958_2018-1deg.nc')['psi_b']
    psi_b = psi_b.mean(dim = 'time')
    bath_cmap = pt.custom_colormaps('bathymetry')

    legend_elements = [
     Line2D([0], [0], ls = '', lw = 0.5, marker = 'o',
            markerfacecolor = 'white', markeredgecolor = 'k', markersize = 6,
            label = 'A12'),
     Line2D([0], [0], ls = '', lw = 0.5, marker = 'o',
            markerfacecolor = 'white', markeredgecolor = 'k', 
            markeredgewidth = .2, markersize = 4,  label = 'Hyd. station'),
     Line2D([0], [0], color = 'k', lw = 1.2, label = '1000m isobath'),
     Line2D([0], [0], color = 'goldenrod', lw = 1.2, 
            label = 'Schematic circulation '),
     Line2D([0], [0], color = 'blue', lw = 1.5, 
            label = 'Feb. SIC (0.1$^\circ$)'),
     Line2D([0], [0], color = 'red', lw = 1.5, 
            label = 'Sep. SIC (0.1$^\circ$)'),
     Line2D([0], [0], color = 'blue', ls = '--', lw = 1.5, 
            label = 'Feb. SIC (NOAA)'),
     Line2D([0], [0], color = 'red', ls = '--', lw = 1.5, 
            label = 'Sep. SIC (NOAA)')]

    fig, axs = pt.map_weddell(190, 95)
    c = axs.contourf(bath['xt_ocean'], bath['yt_ocean'], bath, cmap = bath_cmap,
                    levels = np.arange(0, 6500, 500),
                    transform = ccrs.PlateCarree())
    axs.contour(psi_b['xu_ocean'], psi_b['yt_ocean'], psi_b, linestyles = ['-'],
                levels = [-20, -10], colors = ['goldenrod'], linewidths = 1.2,
                transform = ccrs.PlateCarree())
    axs.contour(aice['xt_ocean'], aice['yt_ocean'], aice.isel(month = 1),
                levels = [0.15], colors = ['blue'], linewidths = .8,
                transform = ccrs.PlateCarree())
    axs.contour(aice['xt_ocean'], aice['yt_ocean'], aice.isel(month = 9),
                levels = [0.15], colors = ['red'], linewidths = .8,
                transform = ccrs.PlateCarree())
    axs.plot(aice_obs_feb[0,:], aice_obs_feb[1,:], color = 'blue',
             linestyle = '--', linewidth = 1, transform = ccrs.PlateCarree())
    axs.plot(aice_obs_sep[0,:], aice_obs_sep[1,:], color = 'red',
             linestyle = '--', linewidth = 1, transform = ccrs.PlateCarree())
    axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1.2,
             transform = ccrs.PlateCarree())
    axs.plot([-60, 10, 10, -60, -60], [-75, -75, -57, -57, -75],
             color = 'white', linewidth = .8, linestyle = '--',
             transform = ccrs.PlateCarree(), zorder = 1)
    axs.plot([-60, 50, 50, -60, -60], [-75, -75, -57, -57, -75],
             color = 'white', linewidth = .8, transform = ccrs.PlateCarree(),
             zorder = 1)
    axs.plot(hyd_lon, hyd_lat, color = 'none', markersize = 2, marker = 'o',
             markeredgecolor = 'k', markerfacecolor = 'None',
             markeredgewidth = .2, transform = ccrs.PlateCarree(), zorder = 2)
    axs.scatter(a12_lon, a12_lat, s = 8, c = 'white', edgecolor = 'k',
                marker = 'o', transform = ccrs.PlateCarree(), zorder = 3)
    axs.legend(handles = legend_elements, ncol = 4, loc = 'lower center',
               bbox_to_anchor = (0.5, -0.35), frameon = False)
    cbar = fig.colorbar(c, ax = axs, orientation = 'vertical', shrink = .6)
    cbar.set_label('Depth [m]')
    plt.savefig(wdir+'/results/figure_1.jpg')

def plot_figure_2():

    etam, etao = at.load_sea_level(keys, wdir)
    for k in keys:
        etam[k] = etam[k].mean(dim = 'time')
    etao = etao['MDT']
    bndy = at.gyre_boundary(keys, wdir, 'mean')
    iso1 = xr.open_dataset(wdir+'/data/isobath_1000m.nc')

    fig, axs = pt.map_weddell(190, 95)
    c = axs.contourf(etao['xt_ocean'], etao['yt_ocean'], etao,
                     levels = np.arange(-216, -130, 2),
                     cmap = cmocean.cm.tarn_r, extend = 'max',
                     transform = ccrs.PlateCarree())
    axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree())
    axs.text(0.98, 0.08, 'Observations', horizontalalignment = 'right',
             transform = axs.transAxes, bbox = dict(boxstyle = 'round',
             facecolor = 'white'))
    axs.text(-0.08, 0.93, 'a)', horizontalalignment = 'right',
             transform = axs.transAxes)
    cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink = .6)
    cbar.set_label('Sea level [cm]')
    plt.savefig(wdir+'/results/figure_2a.jpg')

    for k, t, l in zip(keys, ['b', 'c', 'd'], ['1$^{\circ}$', '0.25$^{\circ}$', 
                                               '0.1$^{\circ}$']):
        fig, axs = pt.map_weddell(190, 95)
        c = axs.contourf(etam[k]['xt_ocean'], etam[k]['yt_ocean'],
                         etam[k]-etao, levels = np.arange(-30, 31, 1),
                         cmap = 'RdBu_r', extend = 'both',
                         transform = ccrs.PlateCarree())
        axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
                 transform = ccrs.PlateCarree())
        axs.plot(bndy[k][:,0], bndy[k][:,1], color = 'k', linestyle = 'solid',
                 linewidth = 1.5, transform = ccrs.PlateCarree())
        axs.text(0.98, 0.08, l, horizontalalignment = 'right',
                 transform = axs.transAxes,
                 bbox = dict(boxstyle = 'round', facecolor = 'white'))
        axs.text(-0.08, 0.93, t+')', horizontalalignment = 'right',
                 transform = axs.transAxes)
        cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink =.6)
        cbar.set_label('Sea level [cm]')
        plt.savefig(wdir+'/results/figure_2'+t+'.jpg')

def plot_figure_3():

    etam, etao = at.load_sea_level(keys, wdir)
    bndy = at.gyre_boundary(keys, wdir, 'mean')
    iso1 = xr.open_dataset(wdir+'/data/isobath_1000m.nc')
    rval = {}; pval = {}
    for k, t, l in zip(keys, ['a', 'b', 'c'], ['1$^{\circ}$', '0.25$^{\circ}$', 
                                               '0.1$^{\circ}$']):
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
        axs.text(0.98, 0.08, l, horizontalalignment = 'right',
                 transform = axs.transAxes,
                 bbox = dict(boxstyle = 'round', facecolor = 'white'))
        axs.text(-0.08, 0.93, t+')', horizontalalignment = 'right',
                 transform = axs.transAxes)
        cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink =.6)
        cbar.set_label('Correlation coefficient')
        plt.savefig(wdir+'/results/figure_3'+t+'.jpg')

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
    for k, l in zip(keys, ['1$^{\circ}$', '0.25$^{\circ}$', '0.1$^{\circ}$']):
        axs.plot(RMSE[k]['time'], np.sqrt(RMSE[k]), color = clrs[k],
                 linewidth = 1.5, label = l)
    axs.xaxis.set_minor_locator(mdates.MonthLocator())
    axs.set_ylabel('RMSE [cm]')
    plt.legend(frameon = False)
    axs.text(-0.05, 0.93, 'd)', horizontalalignment = 'right',
             transform = axs.transAxes)
    plt.savefig(wdir+'/results/figure_3d.jpg')

def plot_figure_4():

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
    plt.savefig(wdir+'/results/figure_4_a_ts.jpg')

    for k, t, l in zip(keys, ['b', 'c', 'd'], ['1$^{\circ}$', '0.25$^{\circ}$', 
                                               '0.1$^{\circ}$']):
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
        ax.text(0.03, 0.93, l, horizontalalignment = 'left',
                transform = ax.transAxes,
                bbox = dict(boxstyle = 'round', facecolor = 'white'));
        ax.text(-0.15, 0.95, t+')', horizontalalignment = 'left',
                transform = ax.transAxes);
        plt.savefig(wdir+'/results/figure_4_'+t+'_ts.jpg')

def plot_figure_5():

    ht = xr.open_dataset(wdir+'/raw_outputs/ocean_grid-01deg.nc')['ht']
    a12_hydro = at.a12_hydrography(wdir)
    a12_pot_rho_obs = at.a12_mean_pot_rho(a12_hydro, 'hydrography')
    ht = ht.sel(xt_ocean = a12_hydro['lon'], yt_ocean = a12_hydro['lat'],
                   method = 'nearest')

    fig, axs = pt.a12_section()
    c = axs[0].contourf(a12_hydro['lat'],
                       -a12_pot_rho_obs['pressure'][:251],
                        a12_pot_rho_obs[:251, :],
                        levels = np.arange(27.1, 27.92, .02), extend = 'both',
                        cmap = 'Spectral')
    axs[1].contourf(a12_hydro['lat'],
                   -a12_pot_rho_obs['pressure'][250:],
                    a12_pot_rho_obs[250:, :],
                    levels = np.arange(27.1, 27.92, .02), extend = 'both',
                    cmap = 'Spectral')
    plt.fill_between(a12_hydro['lat'], -a12_pot_rho_obs['pressure'][-1], -ht,
                    color = 'k')
    cbar = plt.colorbar(c, ax = axs[:], orientation = 'vertical')
    cbar.set_label('$\\sigma_{\\theta}$ [kg m$^{-3}$]')
    axs[0].text(-0.15, 0.95, 'a)', horizontalalignment = 'left',
                transform = axs[0].transAxes);
    axs[1].text(0.98, 0.08, 'Observations', horizontalalignment = 'right', 
                transform = axs[1].transAxes, 
                bbox = dict(boxstyle = 'round', facecolor = 'white'));
    plt.savefig(wdir+'/results/figure_4_a_rs.jpg')

    a12_model = at.a12_model(keys, a12_hydro, wdir)
    a12_pot_rho_mod = at.a12_mean_pot_rho(a12_model, 'model')

    for k, t, l in zip(keys, ['b', 'c', 'd'], ['1$^{\circ}$', '0.25$^{\circ}$', 
                                               '0.1$^{\circ}$']):
        fig, axs = pt.a12_section()
        c = axs[0].contourf(a12_hydro['lat'],
                            -a12_pot_rho_mod[k]['pressure'][:251],
                            (a12_pot_rho_mod[k] - a12_pot_rho_obs)[:251, :],
                            levels = np.arange(-.3, .31, .01), extend = 'both',
                            cmap = 'RdBu_r')
        axs[0].contour(a12_hydro['lat'],
                       -a12_pot_rho_mod[k]['pressure'][:251],
                       (a12_pot_rho_mod[k] - a12_pot_rho_obs)[:251, :],
                       levels = np.arange(-.3, .4, .1), colors = ['k'],
                       linewidths = [.5], linestyles = ['solid'])
        axs[1].contourf(a12_hydro['lat'],
                       -a12_pot_rho_mod[k]['pressure'][250:],
                       (a12_pot_rho_mod[k] - a12_pot_rho_obs)[250:, :],
                       levels = np.arange(-.3, .31, .01), extend = 'both',
                       cmap = 'RdBu_r')
        axs[1].contour(a12_hydro['lat'],
                       -a12_pot_rho_mod[k]['pressure'][250:],
                       (a12_pot_rho_mod[k] - a12_pot_rho_obs)[250:, :],
                       levels = np.arange(-.3, .4, .1), colors = ['k'],
                       linewidths = [.5], linestyles = ['solid'])
        plt.fill_between(a12_hydro['lat'], -a12_pot_rho_obs['pressure'][-1], -ht,
                         color = 'k')
        cbar = plt.colorbar(c, ax = axs[:], orientation = 'vertical')
        cbar.set_label('$\\sigma_{\\theta}$ [kg m$^{-3}$]')
        axs[0].text(-0.15, 0.95, t+')', horizontalalignment = 'left',
                    transform = axs[0].transAxes);
        axs[1].text(0.98, 0.08, l, horizontalalignment = 'right', 
                    transform = axs[1].transAxes, 
                    bbox = dict(boxstyle = 'round', facecolor = 'white'));
        plt.savefig(wdir+'/results/figure_4_'+t+'_rs.jpg')

def plot_figure_6():

    psi_mean = {}
    psi_sdev = {}
    pvor = {}
    for k in keys:
        psi = xr.open_dataset(wdir+'/data/psi_b-monthly-1958_2018-'+k+'deg.nc')['psi_b']
        psi_mean[k] = psi.mean(dim = 'time')
        psi_sdev[k] = psi.std(dim = 'time')
    pvor = at.potential_vorticity(keys, wdir)
    bndy = at.gyre_boundary(keys, wdir, 'mean')
    iso1 = xr.open_dataset(wdir+'/data/isobath_1000m.nc')
    pv_lvls = -np.flip(np.logspace(np.log10(0.0001), np.log10(0.04), num = 50))
    psi_cmap = pt.shiftedColorMap(cmocean.cm.curl, -40, 100.25, 'psi_cmap')

    for k, t, l in zip(keys, ['a', 'c', 'e'], ['1$^{\circ}$', '0.25$^{\circ}$', 
                                               '0.1$^{\circ}$']):
        fig, axs = pt.map_weddell(190, 95)
        cf = axs.contourf(psi_mean[k]['xu_ocean'], psi_mean[k]['yt_ocean'], 
                          psi_mean[k], levels = np.arange(-40, 100.25, 2.5), 
                          cmap = psi_cmap, extend = 'both', 
                          transform = ccrs.PlateCarree())
        ct = axs.contour(pvor[k]['xt_ocean'], pvor[k]['yt_ocean'], pvor[k]*1e6, 
                         levels = pv_lvls, colors = ['k'], linestyles = 'solid', 
                         linewidths = 0.6, transform = ccrs.PlateCarree())
        axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1.5, 
                 transform = ccrs.PlateCarree())
        axs.text(0.98, 0.08, l, horizontalalignment = 'right', 
                 transform = axs.transAxes, 
                 bbox = dict(boxstyle = 'round', facecolor = 'white'));
        axs.text(-0.08, 0.93, t+')', horizontalalignment = 'right', 
                 transform = axs.transAxes);
        cbar = fig.colorbar(cf, ax = axs, orientation = 'horizontal', 
                            shrink = .6)
        cbar.set_label('$\\psi$ [Sv]')
        plt.savefig(wdir+'/results/figure_6_'+t+'.jpg')

    for k, t, l in zip(keys, ['b', 'd', 'f'], ['1$^{\circ}$', '0.25$^{\circ}$', 
                                               '0.1$^{\circ}$']):
        fig, axs = pt.map_weddell(190, 95)
        cf = axs.contourf(psi_sdev[k]['xu_ocean'], psi_sdev[k]['yt_ocean'], 
                          psi_sdev[k], levels = np.arange(0, 32, 2), 
                          cmap = cmocean.cm.amp, extend = 'max', 
                          transform = ccrs.PlateCarree())
        ct = axs.contour(pvor[k]['xt_ocean'], pvor[k]['yt_ocean'], pvor[k]*1e6, 
                         levels = pv_lvls, colors = ['k'], linestyles = 'solid', 
                         linewidths = 0.6, transform = ccrs.PlateCarree())
        axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1.5, 
                 transform = ccrs.PlateCarree())
        axs.text(0.98, 0.08, l, horizontalalignment = 'right', 
                 transform = axs.transAxes, 
                 bbox = dict(boxstyle = 'round', facecolor = 'white'));
        axs.text(-0.08, 0.93, t+')', horizontalalignment = 'right', 
                 transform = axs.transAxes);
        cbar = fig.colorbar(cf, ax = axs, orientation = 'horizontal', 
                            shrink = .6)
        cbar.set_label('$\\psi$ [Sv]')
        plt.savefig(wdir+'/results/figure_6_'+t+'.jpg')

def plot_figure_7():

    subsfc_tmax = {}
    for k in keys:
        subsfc_tmax[k] = xr.open_dataset(wdir+'/data/sub_sfc_tmax-mean-1958_2018-'+k+'deg.nc')['sub_sfc_tmax']
    iso1 = xr.open_dataset(wdir+'/data/isobath_1000m.nc')
    bndy = at.gyre_boundary(keys, wdir, 'mean')
    temp_cmap = pt.custom_colormaps('subsfc_tmax')
    
    lat = xr.open_dataset(wdir+'/data/WeddellGyre_OM_Period2001to2005_potTpSal.nc')['Latitude (DegN)']
    lon = xr.open_dataset(wdir+'/data/WeddellGyre_OM_Period2001to2005_potTpSal.nc')['Longitude (DegE)']
    reeve = xr.open_dataset(wdir+'/data/Reeve-pottemp-mean.nc')
    subsfc_tmax_obs = reeve['Potential Temperature (DegC)'].max(dim = 'level')

    fig, axs = pt.map_weddell(190, 95)
    cf = axs.contourf(lon, lat, subsfc_tmax_obs.transpose(),
                      levels = np.arange(-2, 4.25, .25), cmap = temp_cmap, 
                      extend = 'both', transform = ccrs.PlateCarree())
    axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree())
    axs.text(0.98, 0.08, 'Reeve et al. 2016', horizontalalignment = 'right',
             transform = axs.transAxes, bbox = dict(boxstyle = 'round', 
             facecolor = 'white'));
    axs.text(-0.08, 0.93, 'a)', horizontalalignment = 'right',
             transform = axs.transAxes);
    cbar = fig.colorbar(cf, ax = axs, orientation = 'horizontal',
                       shrink = .6)
    cbar.set_ticks(np.arange(-2, 5, 1))
    cbar.set_label('$\\theta$ [$^{\circ}$C]')
    plt.savefig(wdir+'/results/figure_7_a.jpg')

    for k, t, l in zip(keys, ['b', 'c', 'd'], ['1$^{\circ}$', '0.25$^{\circ}$', 
                                               '0.1$^{\circ}$']):
        fig, axs = pt.map_weddell(190, 95)
        cf = axs.contourf(subsfc_tmax[k]['xt_ocean'],
                          subsfc_tmax[k]['yt_ocean'],
                          subsfc_tmax[k]-273.15,
                          levels = np.arange(-2, 4.25, .25), cmap = temp_cmap, 
                          extend = 'both', transform = ccrs.PlateCarree())
        axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
                 transform = ccrs.PlateCarree())
        axs.plot(bndy[k][:,0], bndy[k][:,1], color = 'k', linestyle = 'solid',
                 linewidth = 1.5, transform = ccrs.PlateCarree())
        axs.text(0.98, 0.08, l, horizontalalignment = 'right',
                 transform = axs.transAxes, bbox = dict(boxstyle = 'round', 
                 facecolor = 'white'));
        axs.text(-0.08, 0.93, t+')', horizontalalignment = 'right',
                 transform = axs.transAxes);
        cbar = fig.colorbar(cf, ax = axs, orientation = 'horizontal',
                           shrink = .6)
        cbar.set_ticks(np.arange(-2, 5, 1))
        cbar.set_label('$\\theta$ [$^{\circ}$C]')
        plt.savefig(wdir+'/results/figure_7_'+t+'.jpg')

def plot_figure_8():

    gstr = at.gyre_strength(keys, wdir, 'seasonal')
    scrl = at.wind_stress_curl(keys, wdir, 'seasonal')
    bflu = at.buoyancy_flux(keys, wdir, 'seasonal')

    for k in keys:
        print('\n Correlations of gyre strength with surface stress curl')
        r = st.linregress(gstr[k]['gstr'], -scrl[k]).rvalue
        edof_a = at.edof(gstr[k]['gstr'])
        edof_b = at.edof(scrl[k])
        tstat = r * np.sqrt(np.mean([edof_a, edof_b]))/np.sqrt(1 - r**2)
        p = st.t.sf(np.abs(tstat), np.mean([edof_a, edof_b]))*2
        if p <= 0.05:
            print(k+'deg: '+ str(r) + ' (significant)')
        else:
            print(k+'deg: '+ str(r) + ' (not-significant)')

        print('\n Correlations of gyre strength with buoyancy_fluxes')
        r = st.linregress(gstr[k]['gstr'], -bflu[k]).rvalue
        edof_a = at.edof(gstr[k]['gstr'])
        edof_b = at.edof(bflu[k])
        tstat = r * np.sqrt(np.mean([edof_a, edof_b]))/np.sqrt(1 - r**2)
        p = st.t.sf(np.abs(tstat), np.mean([edof_a, edof_b]))*2
        if p <= 0.05:
            print(k+'deg: '+ str(r) + ' (significant)')
        else:
            print(k+'deg: '+ str(r) + ' (not-significant)')

    fig, axs = pt.annual_cycles()
    for k in keys:
        axs[2].axhline(y = 0, color = 'k', linewidth = 0.5, linestyle = '--')
        axs[0].plot(np.arange(0, 12, 1), -gstr[k]['gstr'], color = clrs[k], 
                    linewidth = 1)
        axs[1].plot(np.arange(0, 12, 1), -1e7*scrl[k], color = clrs[k], 
                    linewidth = 1)
        axs[2].plot(np.arange(0, 12, 1), 1e8*bflu[k], color = clrs[k], 
                    linewidth = 1)
    axs[0].text(0.96, 0.85, 'Gyre strength', 
                horizontalalignment = 'right',
                transform = axs[0].transAxes);
    axs[1].text(0.96, 0.12, 'Surface stress curl', 
                horizontalalignment = 'right',
                transform = axs[1].transAxes);
    axs[2].text(0.96, 0.85, 'Surface buoyancy flux', 
                horizontalalignment = 'right',
                transform = axs[2].transAxes);
    plt.savefig(wdir+'/results/figure_8.jpg')

def plot_figure_9():

    psib = at.psi_b_seasonal([keys[-1]], wdir)
    psib_djf = psib['01']['psi_b'].isel(month = [11, 0, 1]).mean(dim = 'month')
    psib_jja = psib['01']['psi_b'].isel(month = [5, 6, 7]).mean(dim = 'month')
    bflu = at.buoyancy_flux_seasonal([keys[-1]], wdir)
    bflu_djf = bflu['01'].isel(month = [11, 0, 1]).mean(dim = 'month')
    bflu_jja = bflu['01'].isel(month = [5, 6, 7]).mean(dim = 'month')
    slp = at.slp_seasonal(wdir)
    slp_djf = slp.isel(month = [11, 0, 1]).mean(dim = 'month')
    slp_jja = slp.isel(month = [5, 6, 7]).mean(dim = 'month')
    iso1 = xr.open_dataset(wdir+'/data/isobath_1000m.nc')
    bndy = at.gyre_boundary([keys[-1]], wdir, 'seasonal')

    fig, axs = pt.map_weddell(190, 95)
    c = axs.contourf(psib_djf['xu_ocean'], psib_djf['yt_ocean'], psib_djf,
                     levels = np.arange(-10, 11, 1), cmap = 'RdBu_r',
                     extend = 'both', transform = ccrs.PlateCarree())
    axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree())
    axs.plot(bndy[0][:, 0], bndy[0][:, 1], color = 'k', linewidth =1.5,
             transform = ccrs.PlateCarree())
    axs.text(0.98, 0.08, 'DJF', horizontalalignment = 'right',
             transform = axs.transAxes,
             bbox = dict(boxstyle = 'round', facecolor = 'white'));
    axs.text(-0.08, 0.93, 'a)', horizontalalignment = 'right',
             transform = axs.transAxes);
    cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink = .6)
    cbar.set_label('Transport [Sv]')
    plt.savefig(wdir+'/results/figure_9_a.jpg')

    fig, axs = pt.map_weddell(190, 95)
    c = axs.contourf(psib_jja['xu_ocean'], psib_jja['yt_ocean'], psib_jja,
                     levels = np.arange(-10, 11, 1), cmap = 'RdBu_r',
                     extend = 'both', transform = ccrs.PlateCarree())
    axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree())
    axs.plot(bndy[1][:, 0], bndy[1][:, 1], color = 'k', linewidth =1.5,
             transform = ccrs.PlateCarree())
    axs.text(0.98, 0.08, 'JJA', horizontalalignment = 'right',
             transform = axs.transAxes,
             bbox = dict(boxstyle = 'round', facecolor = 'white'));
    axs.text(-0.08, 0.93, 'c)', horizontalalignment = 'right',
             transform = axs.transAxes);
    cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink = .6)
    cbar.set_label('Transport [Sv]')
    plt.savefig(wdir+'/results/figure_9_c.jpg')

    fig, axs = pt.map_weddell(190, 95)
    c = axs.contourf(bflu_djf['xt_ocean'], bflu_djf['yt_ocean'], 1e8*bflu_djf,
                     levels = np.arange(-5, 5.5, .5), cmap = 'RdBu_r',
                     extend = 'both', transform = ccrs.PlateCarree(),
                     zorder = -2)
    axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree(), zorder = -1)
    axs.plot(bndy[0][:,0], bndy[0][:,1], color = 'k', linewidth = 1.5,
             transform = ccrs.PlateCarree(), zorder = -1)
    cc = axs.contour(slp_djf['lon'], slp_djf['lat'], slp_djf/100,
                     colors = ['white'], levels = np.arange(984, 1002, 2),
                     linewidths = [1.5], transform = ccrs.PlateCarree(),
                     zorder = 0)
    axs.clabel(cc, inline = True, fmt = '%1.0f')
    axs.text(0.98, 0.08, 'DJF', horizontalalignment = 'right',
             transform = axs.transAxes,
             bbox = dict(boxstyle = 'round', facecolor = 'white'));
    axs.text(-0.08, 0.93, 'b)', horizontalalignment = 'right',
             transform = axs.transAxes);
    cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink = .6)
    cbar.set_label('$\\mathcal{B}$ [10$^{-8}$ m$^{2}$  s$^{-3}$]')
    plt.savefig(wdir+'/results/figure_9_b.jpg')

    fig, axs = pt.map_weddell(190, 95)
    c = axs.contourf(bflu_jja['xt_ocean'], bflu_jja['yt_ocean'], 1e8*bflu_jja,
                     levels = np.arange(-5, 5.5, .5), cmap = 'RdBu_r',
                     extend = 'both', transform = ccrs.PlateCarree(),
                     zorder = -2)
    axs.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree(), zorder = -1)
    axs.plot(bndy[1][:,0], bndy[1][:,1], color = 'k', linewidth = 1.5,
             transform = ccrs.PlateCarree(), zorder = -1)
    cc = axs.contour(slp_jja['lon'], slp_jja['lat'], slp_jja/100,
                     colors = ['white'], levels = np.arange(984, 1002, 2),
                     linewidths = [1.5], transform = ccrs.PlateCarree(),
                     zorder = 0)
    axs.clabel(cc, inline = True, fmt = '%1.0f')
    axs.text(0.98, 0.08, 'JJA', horizontalalignment = 'right',
             transform = axs.transAxes,
             bbox = dict(boxstyle = 'round', facecolor = 'white'));
    axs.text(-0.08, 0.93, 'd)', horizontalalignment = 'right',
             transform = axs.transAxes);
    cbar = fig.colorbar(c, ax = axs, orientation = 'horizontal', shrink = .6)
    cbar.set_label('$\\mathcal{B}$ [10$^{-8}$ m$^{2}$  s$^{-3}$]')
    plt.savefig(wdir+'/results/figure_9_d.jpg')

def plot_figure_10():

    gstr = at.gyre_strength(keys, wdir, 'interannual')
    scrl = at.wind_stress_curl(keys, wdir, 'interannual')
    bflu = at.buoyancy_flux(keys, wdir, 'interannual')
    sam = at.sam_index(wdir)
    eas = at.eas_index(wdir)

    for k in keys:
        gstr[k] = gstr[k].rolling(time = 12, center = True).mean()
        scrl[k] = scrl[k].rolling(time = 12, center = True).mean()
        bflu[k] = bflu[k].rolling(time = 12, center = True).mean()
    sam = sam.rolling(time = 12, center = True).mean()
    eas = eas.rolling(time = 12, center = True).mean()

    for k in keys:
        print('\n Correlations of gyre strength with surface stress curl')
        r = st.linregress(gstr[k]['gstr'][6:-5], scrl[k][6:-5]).rvalue
        edof_a = at.edof(gstr[k]['gstr'][6:-5])
        edof_b = at.edof(scrl[k][6:-5])
        tstat = r * np.sqrt(np.mean([edof_a, edof_b]))/np.sqrt(1 - r**2)
        p = st.t.sf(np.abs(tstat), np.mean([edof_a, edof_b]))*2
        if p <= 0.05:
            print(k+'deg: '+ str(r) + ' (significant)')
        else:
            print(k+'deg: '+ str(r) + ' (not-significant)')

        print('\n Correlations of gyre strength with buoyancy fluxes')
        r = st.linregress(-gstr[k]['gstr'][6:-5], bflu[k][6:-5]).rvalue
        edof_a = at.edof(-gstr[k]['gstr'][6:-5])
        edof_b = at.edof(bflu[k][6:-5])
        tstat = r * np.sqrt(np.mean([edof_a, edof_b]))/np.sqrt(1 - r**2)
        p = st.t.sf(np.abs(tstat), np.mean([edof_a, edof_b]))*2
        if p <= 0.05:
            print(k+'deg: '+ str(r) + ' (significant)')
        else:
            print(k+'deg: '+ str(r) + ' (not-significant)')

        print('\n Correlations of gyre strength with SAM')
        r = st.linregress(-gstr[k]['gstr'][6:-5], sam[6:-5]).rvalue
        edof_a = at.edof(-gstr[k]['gstr'][6:-5])
        edof_b = at.edof(sam[6:-5])
        tstat = r * np.sqrt(np.mean([edof_a, edof_b]))/np.sqrt(1 - r**2)
        p = st.t.sf(np.abs(tstat), np.mean([edof_a, edof_b]))*2
        if p <= 0.05:
            print(k+'deg: '+ str(r) + ' (significant)')
        else:
            print(k+'deg: '+ str(r) + ' (not-significant)')

        print('\n Correlations of gyre strength with EAS')
        r = st.linregress(-gstr[k]['gstr'][6:-5], eas[6:-5]).rvalue
        edof_a = at.edof(-gstr[k]['gstr'][6:-5])
        edof_b = at.edof(eas[6:-5])
        tstat = r * np.sqrt(np.mean([edof_a, edof_b]))/np.sqrt(1 - r**2)
        p = st.t.sf(np.abs(tstat), np.mean([edof_a, edof_b]))*2
        if p <= 0.05:
            print(k+'deg: '+ str(r) + ' (significant)')
        else:
            print(k+'deg: '+ str(r) + ' (not-significant)')

    evnt = at.get_events(gstr['01'], wdir)
    t = np.arange(0, 732, 1)

    fig, axs = pt.interannual_time_series()
    for ax in axs:
        ax.axhline(y = 0, color = 'k', linewidth = 0.5, linestyle = '--')
        ax.axhline(y = 0, color = 'k', linewidth = 0.5, linestyle = '--')
        for i in range(0, len(evnt[0][0])):
            ax.axvspan(t[evnt[0][0][i]], t[evnt[0][1][i]], color = 'mistyrose')
        for i in range(0, len(evnt[1][0])):
            ax.axvspan(t[evnt[1][0][i]], t[evnt[1][1][i]], color = 'lavender')
    for k in keys:
        axs[0].plot(t, -gstr[k]['gstr'], color = clrs[k], linewidth = .8)
        axs[1].plot(t, -1e7*scrl[k], color = clrs[k], linewidth = .8)
        axs[2].plot(t, 1e8*bflu[k], color = clrs[k], linewidth = .8)
    axs[3].plot(t, sam, color = 'k', linewidth = .8)
    axs[4].plot(t, eas, color = 'k', linewidth = .8)
    plt.savefig(wdir+'/results/figure_10.jpg')

def plot_figure_11():

    iso1 = xr.open_dataset(wdir+'/data/isobath_1000m.nc')
    gstr = at.gyre_strength(keys, wdir, 'interannual')
    gstr = gstr['01'].rolling(time = 12, center = True).mean()
    evnt = at.get_events(gstr, wdir)

    psib_cmp = at.composites(evnt, 'psi_b', wdir)
    slp_cmp = at.composites(evnt, 'psl', wdir)
    slp_cmp = slp_cmp.sel(lon = slice(-70, 80), lat = slice(None, -50))
    
    fig, ax = pt.map_weddell(190, 95)
    cf = ax.contourf(psib_cmp['xu_ocean'], psib_cmp['yt_ocean'], psib_cmp,
                     levels = np.arange(-10, 11, 1), cmap = 'RdBu_r',
                     extend = 'both', transform = ccrs.PlateCarree(),
                     zorder = 0)
    ax.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree())
    cc = ax.contour(slp_cmp['lon'], slp_cmp['lat'], slp_cmp/100,
                    colors = ['white'], transform = ccrs.PlateCarree(),
                    zorder = 1)
    ax.clabel(cc, inline = True, fmt = '%1.1f', fontsize = 8)
    cbar = fig.colorbar(cf, ax = ax, orientation = 'horizontal', shrink =.6)
    cbar.set_label('Transport [Sv]', fontsize = 8)
    ax.text(-0.08, 0.93, 'a)', horizontalalignment = 'right', 
            transform = ax.transAxes);
    plt.savefig(wdir+'/results/figure_11_a.jpg')
    
    bflu_cmp = at.composites(evnt, 'buoyancy_flux', wdir)
    lr_bflu = at.composites_correlations(gstr['gstr'], 'bflux', wdir)
    
    fig, ax = pt.map_weddell(190, 95)
    cf = ax.contourf(bflu_cmp['xt_ocean'], bflu_cmp['yt_ocean'], bflu_cmp*1e8,
                    levels = np.arange(-1, 1.1, .1), cmap = 'RdBu_r',
                    extend = 'both', transform = ccrs.PlateCarree())
    ax.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree())
    ax.contourf(lr_bflu['xt_ocean'], lr_bflu['yt_ocean'],
                lr_bflu.where(lr_bflu < 0.1), colors = ['none'],
                hatches = ['xxx'], transform = ccrs.PlateCarree())
    cbar = fig.colorbar(cf, ax = ax, orientation = 'horizontal', shrink =.6)
    cbar.set_label('$\\mathcal{B}$ [10$^{-8}$ m$^{2}$  s$^{-3}$]')
    ax.text(-0.08, 0.93, 'b)', horizontalalignment = 'right', 
            transform = ax.transAxes);
    plt.savefig(wdir+'/results/figure_11_b.jpg')
        
    aice_cmp = at.composites(evnt, 'aice', wdir)
    lr_aice = at.composites_correlations(gstr['gstr'], 'aice', wdir)
    
    fig, ax = pt.map_weddell(190, 95)
    cf = ax.contourf(aice_cmp['xt_ocean'], aice_cmp['yt_ocean'], aice_cmp*100,
                     levels = np.arange(-.1, .11, .01)*100, cmap = 'RdBu_r',
                     extend = 'both', transform = ccrs.PlateCarree())
    ax.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree())
    ax.contourf(lr_aice['xt_ocean'], lr_aice['yt_ocean'],
                lr_aice.where(lr_aice < 0.1), colors = ['none'],
                hatches = ['xxx'], transform = ccrs.PlateCarree())
    cbar = fig.colorbar(cf, ax = ax, orientation = 'horizontal', shrink =.6)
    cbar.set_label('Sea ice concentration [$\\%$]')
    ax.text(-0.08, 0.93, 'd)', horizontalalignment = 'right', 
            transform = ax.transAxes);
    plt.savefig(wdir+'/results/figure_11_d.jpg')
    
    tmax_cmp = at.composites(evnt, 'tmax', wdir)
    lr_tmax = at.composites_correlations(gstr['gstr'], 'tmax', wdir)

    fig, ax = pt.map_weddell(190, 95)
    cf = ax.contourf(tmax_cmp['xt_ocean'], tmax_cmp['yt_ocean'], tmax_cmp,
                     levels = np.arange(-.5, .55, .05), cmap = 'RdBu_r',
                     extend = 'both', transform = ccrs.PlateCarree())
    ax.plot(iso1['x'], iso1['y'], color = 'k', linewidth = 1,
             transform = ccrs.PlateCarree())
    ax.contourf(lr_tmax['xt_ocean'], lr_tmax['yt_ocean'],
                lr_tmax.where(lr_tmax<0.1), colors = ['none'],
                hatches = ['xxx'], transform = ccrs.PlateCarree())
    cbar = fig.colorbar(cf, ax = ax, orientation = 'horizontal', shrink =.6)
    cbar.set_label('$\\theta$ [$^{\circ}$C]')
    ax.text(-0.08, 0.93, 'c)', horizontalalignment = 'right', 
            transform = ax.transAxes);
    plt.savefig(wdir+'/results/figure_11_c.jpg')

###############################################################################

def main():

    clrs = pt.set_rcParams()

    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    plot_figure_7()
    plot_figure_8()
    plot_figure_9()
    plot_figure_10()
    plot_figure_11()

if __name__ == "__main__":
    main()
