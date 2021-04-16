###############################################################################
# Description	: Analysis functions
# Args          :
#     wdir = path to directory in which save data
# Author        : Julia Neme
# Email         : j.neme@unsw.edu.au
###############################################################################


###############################################################################

def g02202_aice(month):

    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import xarray as xr
    from glob import glob

    # from http://nsidc.org/data/G02202
    path = '/g/data3/hh5/tmp/cosima/observations/NOAA/G02202_V3'
    fils = glob(os.path.join(path, 'south/monthly/*.nc'))
    drop = ['projection', 'seaice_conc_monthly_cdr',
            'stdev_of_seaice_conc_monthly_cdr',
            'melt_onset_day_seaice_conc_monthly_cdr',
            'qa_of_seaice_conc_monthly_cdr', 'goddard_nt_seaice_conc_monthly',
            'goddard_bt_seaice_conc_monthly']
    aice = xr.open_mfdataset(fils, drop_variables = drop)
    aice = aice['goddard_merged_seaice_conc_monthly']
    aice = aice.groupby('time.month').mean(dim = 'time')

    if month == 'feb':
        cset = plt.contour(aice['longitude'], aice['latitude'],
                           aice.isel(month = 1), levels = [0.15])
        siex = np.array([cset.allsegs[0][5][:,0], cset.allsegs[0][5][:,1]])
        plt.close('all')
        srt = siex[0,:].argsort()
        siex = np.array([siex[0, srt], siex[1, srt]])
    elif month == 'sep':
        cset = plt.contour(aice['longitude'], aice['latitude'],
                           aice.isel(month = 9), levels = [0.15])
        siex = np.array([cset.allsegs[0][3][:,0], cset.allsegs[0][3][:,1]])
        plt.close('all')
        srt = siex[0,:].argsort()
        siex = np.array([siex[0, srt], siex[1, srt]])

    return siex

def hyd_lat_lon():

    import os
    import xarray as xr
    from glob import glob
    from joblib import Parallel, delayed

    def lat_parallel(file_name):
        lat = xr.open_dataset(file_name)['latitude']
        return lat
    def lon_parallel(file_name):
        lon = xr.open_dataset(file_name)['longitude']
        return lon

    fils = [y for x in os.walk('/scratch/e14/jn8053/cchdo_hydrography/')
            for y in glob(os.path.join(x[0], '*.nc'))]
    lat = Parallel(n_jobs = -1)(delayed(lat_parallel)(file) for file in fils)
    lon = Parallel(n_jobs = -1)(delayed(lon_parallel)(file) for file in fils)

    return lat, lon

def load_sea_level(keys, wdir):

    import xarray as xr

    """
    Returns sea level from obs and model interpolated to the obs grid, with the
    artificial global annual cycle removed and an offset applied.
    """

    obs = xr.open_dataset(wdir+'/CS2_combined_Southern_Ocean_2011-2016_regridded.nc')
    obs = obs.rename(({'X':'xt_ocean', 'Y':'yt_ocean', 'date':'time'}))
    obs = obs.sortby('yt_ocean', ascending = True)
    obs['_lon_adj'] = xr.where(obs['xt_ocean'] > 80, obs['xt_ocean'] - 360,
                               obs['xt_ocean'])
    obs = obs.swap_dims({'xt_ocean': '_lon_adj'})
    obs = obs.sel(**{'_lon_adj': sorted(obs._lon_adj)}).drop('xt_ocean'))
    obs = obs.rename({'_lon_adj': 'xt_ocean'})

    mod = {}
    for k in keys:
        mod[k] = xr.open_dataset(wdir+'/sea_level-monthly-2011_2016-'+k+'deg.nc')['sea_level']
        mod[k] = mod[k].sel(time = slice('2011-01-01', '2017-01-01'))
        mod[k] = mod[k]*100
        # Filter an artifical global sea level annual cycle
        etag = xr.open_dataset(wdir+'/eta_global-monthly-1958_2018-'+k+'deg.nc')['eta_global'].squeeze()
        etag = etag.groupby('time.month').mean(dim = 'time')*100
        mod[k] = mod[k].groupby('time.month') - etag
        # Offset
        msk = (obs['MDT'] != 0)
        mod[k] = mod[k].interp(xt_ocean = obs['MDT']['xt_ocean'],
                               yt_ocean = obs['MDT']['yt_ocean'])
        mean_mod = mod[k].where(msk == True).mean()
        mean_obs = obs['MDT'].where(msk == True).mean()
        offs = mean_obs - mean_mod
        mod[k] = (mod[k] + offs).sel(xt_ocean = slice(-70, 80),
                                     yt_ocean = slice(-80, -50))
    obs = obs.sel(xt_ocean = slice(-70, 80), yt_ocean = slice(-80, -50))

    return mod, obs

def gyre_boundary(keys, wdir, field):

    import matplotlib.pyplot as plt
    import xarray as xr

    bndy = {}
    for k in keys:
        psib = xr.open_dataset(wdir+'/psi_b-monthly-1958_2018-'+k+'deg.nc')
        if field == 'mean':
            psib = psib['psi_b'].mean(dim = 'time')
            cset = plt.contour(psib['xu_ocean'], psib['yt_ocean'], psib,
                               levels = [-12])
            if k == '01':
                bndy[k] = cset.allsegs[0][1]
            else:
                bndy[k] = cset.allsegs[0][0]
            plt.close('all')

    return bndy

def edof(A):
    """
    Calculates effective degrees of freedom using the integral timescale as per
    Emery and Thompson's book.
    """
    import numpy as np
    from scipy import integrate

    if len(np.shape(A)) == 1:
        N = len(A)
        x = A - A.mean(dim = 'time')
        c = np.correlate(x, x, 'full')
        c = c[N-1:]/(N-1-np.arange(0, N, 1))
        n = 0
        while (c[n] > 0) and (n < N/2):
            n = n+1
        T = integrate.trapz(c[:n])/c[0]
        edof = N/(2*T)
    else:
        N = len(A['time'])
        A = A - A.mean(dim = 'time')
        p = np.where(A.isel(time = 0) != np.isnan(A.isel(time = 0)))
        edof = np.empty(np.shape(A.isel(time = 0)))*np.nan
        for i in range(0, len(A['xt_ocean'])):
            for j in range(0, len(A['yt_ocean'])):
                if any(np.isnan(A.isel(xt_ocean = i, yt_ocean = j))):
                    pass
                else:
                    x = A.isel(xt_ocean = i, yt_ocean = j)
                    c = np.correlate(x, x, 'full')
                    c = c[N-1:]/(N-1-np.arange(0, N, 1))
                    n = 0
                    while (c[n] > 0) and (n < N/2):
                        n = n+1
                    T = integrate.trapz(c[:n])/c[0]
                    if np.isinf(N/(2*T)):
                        pass
                    else:
                        edof[j, i] = N/(2*T)
        edof = np.nanmean(edof)

    return edof

def linear_correlation(A, B, *, lagB = False):

    """
    Calculates correlation coefficient from linear regression and associated
    pvalues according to a t-student. Input can be 1D or 2D.
    """
    import numpy as np
    import xarray as xr
    from scipy.stats import t

    edof_A = edof(A)
    edof_B = edof(B)
    edof_m = np.nanmean([edof_A, edof_B])

    A, B = xr.align(A, B)
    if lagB:
        B = B.shift(time = lagB).dropna('time', how = 'all')
        A, B = xr.align(A, B)
    n = len(A['time'])
    Am = A.mean(dim = 'time')
    Bm = B.mean(dim = 'time')
    As = A.std(dim = 'time')
    Bs = B.std(dim = 'time')
    covr = np.sum((A - Am)*(B - Bm), axis = 0) / (n-1)
    rval = covr/(As * Bs)
    t_s = rval * np.sqrt(edof_m)/np.sqrt(1 - rval**2)
    pval = t.sf(np.abs(t_s), edof_m)*2
    pval = xr.DataArray(pval, dims = rval.dims, coords = rval.coords)

    return rval, pval

def load_temp_salt_hydrography():

    import gsw
    import numpy as np
    import os
    import xarray as xr
    from glob import glob
    from joblib import Parallel, delayed

    def temp_parallel(file_name):
        prof = xr.open_dataset(file_name)
        if 'CTDTMP' in prof.data_vars:
            temp_name = 'CTDTMP'
            salt_name = 'CTDSAL'
        else:
            temp_name = 'temperature'
            salt_name = 'salinity'
        temp = prof[temp_name]
        p_unq, p_unq_idx = np.unique(prof['pressure'], return_index = True)
        temp = temp[p_unq_idx].interp(pressure = np.arange(0, 6000, 10))
        salt = prof[salt_name]
        salt = salt[p_unq_idx].interp(pressure = np.arange(0, 6000, 10))
        salt_abs = gsw.SA_from_SP(salt, np.arange(0, 6000, 10),
                                  prof['longitude'].values,
                                  prof['latitude'].values)
        pot_temp = gsw.pt0_from_t(salt_abs, temp, np.arange(0, 6000, 10))
        return pot_temp.values
    def salt_parallel(file_name):
        prof = xr.open_dataset(file_name)
        if 'CTDSAL' in prof:
            salt_name = 'CTDSAL'
        else:
            salt_name = 'salinity'
        salt = prof[salt_name]
        p_unq, p_unq_idx = np.unique(prof['pressure'], return_index = True)
        salt = salt[p_unq_idx].interp(pressure = np.arange(0, 6000, 10))
        return salt.values
    def time_parallel(file_name):
        prof = xr.open_dataset(file_name)
        year = prof['time.year'].item()
        mnth = prof['time.month'].item()
        return year, mnth
    def lat_parallel(file_name):
        lat = xr.open_dataset(file_name)['latitude']
        return lat.values
    def lon_parallel(file_name):
        lon = xr.open_dataset(file_name)['longitude']
        return lon.values

    # HAVE TO SEE HOW I MAKE THESE AVAILABLE
    fils = [y for x in os.walk('/scratch/e14/jn8053/cchdo_hydrography/')
             for y in glob(os.path.join(x[0], '*.nc'))]
    pott = Parallel(n_jobs = -1)(delayed(temp_parallel)(file) for file in fils)
    salt = Parallel(n_jobs = -1)(delayed(salt_parallel)(file) for file in fils)
    yrmh = Parallel(n_jobs = -1)(delayed(time_parallel)(file) for file in fils)
    yrmh = np.array(yrmh)
    lat = Parallel(n_jobs = -1)(delayed(lat_parallel)(file) for file in fils)
    lat = np.squeeze(np.array(lat))
    lon = Parallel(n_jobs = -1)(delayed(lon_parallel)(file) for file in fils)
    lon = np.squeeze(np.array(lon))

    pot_temp_xr = xr.DataArray(np.transpose(pott), name = 'pot_temp',
                               dims = ['pressure', 'station'],
                               coords = {'pressure':np.arange(0, 6000, 10),
                                         'station':np.arange(0, len(lon), 1)})
    salt_xr = xr.DataArray(np.transpose(salt), name = 'salt',
                           dims = ['pressure', 'station'],
                           coords = {'pressure':np.arange(0, 6000, 10),
                                     'station':np.arange(0, len(lon), 1)})
    year_xr = xr.DataArray(yrmh[:, 0], name = 'year', dims = ['station'],
                          coords = {'station':np.arange(0, len(lon), 1)})
    mth_xr = xr.DataArray(yrmh[:, 1], name = 'month', dims = ['station'],
                          coords = {'station':np.arange(0, len(lon), 1)})
    lon_xr = xr.DataArray(lon, name = 'lon', dims = ['station'],
                          coords = {'station':np.arange(0, len(lon), 1)})
    lat_xr = xr.DataArray(lat, name = 'lat', dims = ['station'],
                          coords = {'station':np.arange(0, len(lon), 1)})
    ts_hydro = xr.merge([pot_temp_xr, salt_xr, year_xr, mth_xr, lon_xr, lat_xr])

    return ts_hydro

def load_temp_salt_model(key, wdir, ts_hydro):

    import gsw
    import numpy as np
    import xarray as xr
    # CORRECT HOW WE LOAD..
    if key == '01':
        temp = xr.open_dataset(wdir+'/pot_temp_hydrography-'+key+'deg.nc')['pot_temp'] - 273.15
        salt = xr.open_dataset(wdir+'/salt_hydrography-'+key+'deg.nc')['salt']
    else:
        temp = xr.open_dataset(wdir+'/pot_temp-monthly-1958_2018-'+key+'deg.nc')['pot_temp'] - 273.15
        salt = xr.open_dataset(wdir+'/salt-monthly-1958_2018-'+key+'deg.nc')['salt']
    yrmh = np.array([ts_hydro['year'].values, ts_hydro['month'].values])
    unqe = np.unique(yrmh, axis = 1)
    temp_times = temp.sel(time = str(unqe[0,0])+'-'+f'{unqe[1,0]:02}',
                          method = 'nearest')
    salt_times = salt.sel(time = str(unqe[0,0])+'-'+f'{unqe[1,0]:02}',
                          method = 'nearest')
    idx = np.where((np.transpose(yrmh) == unqe[:,0]).all(axis = 1))[0]
    lat_model = xr.DataArray(ts_hydro['lat'][idx], dims = 'station')
    lon_model = xr.DataArray(ts_hydro['lon'][idx], dims = 'station')
    temp_model = temp_times.sel(xt_ocean = lon_model, yt_ocean = lat_model,
                                method = 'nearest').squeeze()
    salt_model = salt_times.sel(xt_ocean = lon_model, yt_ocean = lat_model,
                                method = 'nearest').squeeze()
    for i in range(1, len(unqe[0,:])):
        temp_times = temp.sel(time = str(unqe[0,i])+'-'+f'{unqe[1,i]:02}',
                              method = 'nearest')
        salt_times = salt.sel(time = str(unqe[0,i])+'-'+f'{unqe[1,i]:02}',
                              method = 'nearest')
        idx = np.where((np.transpose(yrmh) == unqe[:,i]).all(axis = 1))[0]
        lat_model = xr.DataArray(ts_hydro['lat'][idx], dims = 'station')
        lon_model = xr.DataArray(ts_hydro['lon'][idx], dims = 'station')
        temp_times = temp_times.sel(xt_ocean = lon_model, yt_ocean = lat_model,
                                    method = 'nearest').squeeze()
        salt_times = salt_times.sel(xt_ocean = lon_model, yt_ocean = lat_model,
                                    method = 'nearest').squeeze()
        temp_model = xr.concat([temp_model, temp_times], dim = 'station')
        salt_model = xr.concat([salt_model, salt_times], dim = 'station')

    p = gsw.p_from_z(-temp_model['st_ocean'], -65)
    temp_model['st_ocean'] = p
    temp_model = temp_model.rename({'st_ocean':'pressure'})
    temp_model = temp_model.interp(pressure = np.arange(0, 6000, 10))
    temp_model = temp_model.sortby('station')
    salt_model['st_ocean'] = p
    salt_model = salt_model.rename({'st_ocean':'pressure'})
    salt_model = salt_model.interp(pressure = np.arange(0, 6000, 10))
    salt_model = salt_model.sortby('station')
    salt_model = salt_model.where(ts_hydro['salt'] != np.nan)
    ts_model = xr.merge([temp_model, salt_model])

    return ts_model

def a12_hydrography():

    import gsw
    import numpy as np
    import xarray as xr
    from metpy.interpolate import interpolate_to_points
    from glob import glob
    from joblib import Parallel, delayed

    def load_stations(directory):
        prof = {}; n = 0;
        file_names = [y for x in os.walk(''+directory+'/')
                      for y in glob(os.path.join(x[0], '*.nc'))]
        file_names = np.sort(file_names)
        prof[n] = xr.open_dataset(file_names[0])
        n += 1
        if 'CTDTMP' in station[0].data_vars:
            temp_name = 'CTDTMP'
            salt_name = 'CDTSAL'
         else:
            temp_name = 'temperature'
            salt_name = 'salinity'
        prof[n] = prof[n].rename({temp_name:'temp', salt_name:'salt'})
        for file in file_names[1:]:
            prof[n] = xr.open_dataset(file)
            prof[n] = prof[n].rename({temp_name:'temp', salt_name:'salt'})
            n += 1
        return prof

    def interp_to_pressure(station, p):
        N = len(station)
        rho = np.empty([len(p), N])*np.nan
        lon = np.empty(N)*np.nan
        lat = np.empty(N)*np.nan
        tim = [station[0]['time'].values]
        for st in range(0, N, 1):
            lon[st] = station[st]['longitude']
            lat[st] = station[st]['latitude']
            temp = station[st]['temp'].dropna(dim = 'pressure')
            temp = temp.interp(pressure = p, method = 'nearest')
            salt = station[st]['salt'].dropna(dim = 'pressure')
            salt = salt.interp(pressure = p, method = 'nearest')
            salt_abs = gsw.SA_from_SP(salt, p, lon[st], lat[st])
            temp_cst = gsw.CT_from_t(salt_abs, temp, p)
            rho[:, st] = gsw.sigma0(salt_abs, temp_cst)
            if st != 0:
                tim.append(station[st]['time'].values)
        idx_shallow = []
        for st in range(0, N, 1):
            depth = ht.sel(xt_ocean = station[st]['longitude'],
                           yt_ocean = station[st]['latitude'],
                           method = 'nearest')
            count_nans = np.isnan(rho[:, st]).sum()
            # Depth of deepest station
            z = gsw.z_from_p(station[st]['pressure'][-1],
                             station[st]['latitude'])
            # Distance of deepest station from the bottom
            distance_to_bottom = depth - z.values
            if count_nans > 0.8*distance_to_bottom:
                idx_shallow.append(st)
        rho = np.delete(rho, idx_shallow, axis = 1)
        lon = np.delete(lon, idx_shallow)
        lat = np.delete(lat, idx_shallow)
        tim = np.delete(tim, idx_shallow)

        rho_array = xr.DataArray(rho, name = 'pot_rho',
                                 dims = ['pressure', 'station'],
                                 coords = {'pressure':p,
                                 'station':np.arange(0, len(lon), 1)})
        rho_array.expand_dims({'lon':lon, 'lat':lat, 'time':tim})
        lon_array = xr.DataArray(lon, name = 'lon', dims = ['station'],
                                 coords = {'station':np.arange(0, len(lon), 1)})
        lat_array = xr.DataArray(lat, name = 'lat', dims = ['station'],
                                 coords = {'station':np.arange(0, len(lon), 1)})
        t_array = xr.DataArray(tim, name = 'time', dims = ['station'],
                               coords = {'station':np.arange(0, len(lon), 1)})
        dset = xr.merge([rho_array, lon_array, lat_array, t_array])
        dset = dset.sortby('lat')
        return dset

    def project_to_a12(dset, idx_a12):

        def projection_all_depths(dset, st, pr, x, y, xi, yi):
            interp_to_a12 = interpolate_to_points((x, y),
                                                  dset['pot_rho'][pr, st].values,
                                                  (xi, yi),
                                                  interp_type = 'nearest')
            return interp_to_a12

        xori = dset['lon'][idx_a12]
        yori = dset['lat'][idx_a12]
        v = np.array([0, -55+69])
        u = np.transpose(np.array([xori, yori + 69]))
        pr = [np.dot(u, v)/(np.sum(v**2))*v[0],
              np.dot(u, v)/(np.sum(v**2))*v[1] - 69]
        xi = pr[0]
        yi = pr[1]
        # Remove points that are > 0.7 degrees away from the transect
        dist = np.sqrt((xori-xi)**2 + (yori-yi)**2)
        if np.shape(np.where(dist.values > 0.5)[0] != 0):
            xi = np.delete(xi, np.where(dist.values > 0.5)[0])
            yi = np.delete(yi, np.where(dist.values > 0.5)[0])
            xori = np.delete(xori, np.where(dist.values > 0.5)[0])
            yori = np.delete(yori, np.where(dist.values > 0.5)[0])
            idx_a12 = np.delete(idx_a12, np.where(dist.values > 0.5)[0])

        rho_a12 = Parallel(n_jobs = -1)(delayed(projection_all_depths)(dset,
                        idx_a12, p, xori, yori, xi, yi)
                        for pr in range(len(dset['pressure'])))
        rho_a12 = np.squeeze(rho_a12)

        return rho_a12

    def interp_to_standard_a12(dset):

        def distance_coordinate_A12(x, y):
            dist = [0]
            xini = -0
            yini = -50
            for i in range(1, len(x)):
                dist.append(np.sqrt((x[i] - xini)**2 + (y[i] - yini)**2))
            dist = np.squeeze(dist)
            return dist

        xf = np.zeros(20)
        yf = np.linspace(-69, -55, 20)
        dist_coords = distance_coordinate_A12(Xi[0], Xi[1])
        dist_interp = distance_coordinate_A12(Xf[0], Xf[1])
        dist_unique, dist_unique_idx = np.unique(dist_coords,
                                                 return_index = True)

        rho_xr = xr.DataArray(dset[:, dist_unique_idx], name = 'pot_rho',
                              dims = ['pressure', 'distance'],
                              coords = {'pressure':p, 'distance':dist_unique})
        rho_xr = rho_xr.interp(distance = dist_interp, method = 'nearest')
        # Change dimension distance for a station number
        rho_xr['distance'] = np.arange(0, len(Xf[0]), 1)
        rho_xr = rho_xr.rename({'distance':'station'})
        x_xarray = xr.DataArray(Xf[0], name = 'lon', dims = ['station'],
                                coords = {'station':s_xarray['station']})
        y_xarray = xr.DataArray(Xf[1], name = 'lat', dims = ['station'],
                                coords = {'station':s_xarray['station']})
        rho_final = xr.merge([t_xarray, x_xarray, y_xarray])
        return rho_final

    dirs = ['a12_nc_ctd', 's04a_nc_ctd', 'sr04_e_nc_ctd', 'a12_1999a_nc_ctd',
            '06AQ20050122_nc_ctd', '06AQ20071128_nc_ctd', '06AQ20080210_nc_ctd',
            '06AQ20101128', '06AQ20141202_nc_ctd']
    ht = xr.open_dataset(wdir+'/ocean_grid-01deg.nc')['ht']
    ht = ht.sel(xt_ocean = slice(-30, 30), yt_ocean = slice(-80, 50))
    p = np.arange(0, 6000, 10)

    temp_a12_st = {}; salt_a12_st = {}
    for d in dirs:
        station = load_stations(d)
        station_interp = interpolate_to_pressure(station, p)
        idx_A12 = np.where((station_interp['lon'] <= 2) &
                           (station_interp['lon'] >= -2) &
                           (station_interp['lat'] > -75) &
                           (station_interp['lat'] <= -55))[0]
        rho_a12 = project_to_a12(dset, idx_A12)
        rho_a12_st[d] = interp_to_standard_a12(rho_a12)
        if d == dirs[0]:
            rho_a12_repeat = rho_a12_st[d]
        else:
            rho_a12_repeat = xr.concat([rho_a12_repeat, rho_a12_st[d]],
                                        dim = 'cruise')

    return rho_a12_repeat
