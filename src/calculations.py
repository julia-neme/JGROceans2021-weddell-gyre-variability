###############################################################################
# Description	: This script uses raw output from ACCESS-OM2 model to perform offline calculations and saves the output to wdir.
# Args        :
#     wdir = path to directory in which save data
# Author      : Julia Neme
# Email       : j.neme@unsw.edu.au
###############################################################################

import dask.distributed as dask
import gsw
import numpy as np
import xarray as xr

clnt = dsk.Client()
keys = ['1', '025', '01']
wdir = input("Please directory where to save the data: ")

###############################################################################

def ice_coords_corrections(key):

    """
    Corrects CICE data time stamps and grid to match the MOM's.
    """

    aice = xr.open_dataset(wdir+'/aice_m-monthly-1958_2018-'+key+'deg-global.nc')
    time = xr.open_dataset(wdir+'/net_sfc_heating-monthly-1958_2018-'+key+'deg.nc')['time']
    grid = xr.open_dataset(wdir+'data/raw_outputs/ocean_grid-'+key+'deg.nc')['ht']

    aice['time'] = time['time'].values
    aice.coords['ni'] = grid['xt_ocean'].values
    aice.coords['nj'] = grid['yt_ocean'][:len(aice['nj'])].values
    aice = aice.rename(({'ni':'xt_ocean', 'nj':'yt_ocean'}))

    aice.sel(xt_ocean = slice(-70, 80), yt_ocean = slice(-80, -50)).to_netcdf(wdir+'/aice_m-monthly-1958_2018-'+key+'deg.nc')
    os.remove(wdir+'/aice_m-monthly-1958_2018-'+key+'deg-global.nc')

def barotropic_streamfunction(key):

    psi_b = xr.open_dataset(wdir+'/tx_trans_int_z-monthly-1958_2018-'+key+'deg.nc')['tx_trans_int_z'].cumsum(dim = 'yt_ocean')/(1035*1e6)

    dset = xr.DataArray(psi_b, name = 'psi_b')
    dset.to_netcdf(wdir+'/psi_b-monthly-1958_2018-'+key+'deg.nc', mode = 'w')
    dset.close()

def buoyancy_flux(key):

    net_sfc_heating = xr.open_dataset(wdir+'/net_sfc_heating-monthly-1958_2018-'+key+'deg.nc')['net_sfc_heating']
    frazil_3d_int_z = xr.open_dataset(wdir+'/frazil_3d_int_z-monthly-1958_2018-'+key+'deg.nc')['frazil_3d_int_z']
    pme_river = xr.open_dataset(wdir+'/pme_river-monthly-1958_2018-'+key+'deg.nc')['pme_river']
    temp = xr.open_dataset(wdir+'/temp-monthly-1958_2018-'+key+'deg.nc')['temp'].isel(st_ocean = 0)-273.15
    salt = xr.open_dataset(wdir+'/salt-monthly-1958_2018-'+key+'deg.nc')['salt'].isel(st_ocean = 0)

    salt_abs = gsw.SA_from_SP(salt, 0, salt['xt_ocean'], salt['yt_ocean'])
    alpha = gsw.alpha(salt_abs, temp, 0)
    rho = gsw.rho(salt_abs, temp, 0)
    beta = gsw.beta(salt_abs, temp, 0)

    bflux_from_heat = (9.8*alpha)/(3850*rho)*(net_sfc_heating + frazil_3d_int_z)
    bflux_from_fwat = 9.8*beta*(pme_river/1000)*salt_abs
    bflux = bflux_from_heat + bflux_from_fwat

    dset = xr.DataArray(bflux, name = 'buoyancy_flux')
    dset.to_netcdf(wdir+'/buoyancy_flux-monthly-1958_2018-'+key+'deg.nc', mode = 'w')
    dset.close()

def sfc_stress_curl(key):

    tau_x = xr.open_dataset(wdir+'data/raw_outputs/tau_x-monthly-1958_2018-'+key+'deg.nc')['tau_x']
    tau_y = xr.open_dataset(wdir+'data/raw_outputs/tau_y-monthly-1958_2018-'+key+'deg.nc')['tau_y']

    R = 6371e3
    f = 2*7.292e-5*np.sin(tau_x['yu_ocean']*np.pi/180)
    deg_to_m_lon = np.pi/180*R*np.cos(np.deg2rad(tau_x['yu_ocean']))
    deg_to_m_lat = np.pi/180*R

    sfc_stress_curl = tau_y.differentiate('xu_ocean')/deg_to_m_lon - tau_x.differentiate('yu_ocean')/deg_to_m_lat

    dset = xr.DataArray(sfc_stress_curl, name = 'sfc_stress_curl')
    dset.to_netcdf(wdir+'/sfc_stress_curl-monthly-1958_2018-'+key+'deg.nc', mode = 'w')
    dset.close()

def sub_sfc_tmax(key):

    pot_temp = xr.open_dataset(wdir+'/pot_temp-mean-1958_2018-'+key+'deg.nc')['pot_temp']

    sub_sfc_tmax = pot_temp.max(dim = ['st_ocean'])

    dset = xr.DataArray(sub_sfc_tmax, name = 'sub_sfc_tmax')
    dset.to_netcdf(wdir+'/sub_sfc_tmax-monthly-1958_2018-'+key+'deg.nc', mode = 'w')
    dset.close()

###############################################################################

def main():
    for k in keys:
        ice_coords_corrections(k)
        barotropic_streamfunction(k)
        buoyancy_flux(k)
        sfc_stress_curl(k)
        sub_sfc_tmax(k)

if __name__ == "__main__":
    main()
