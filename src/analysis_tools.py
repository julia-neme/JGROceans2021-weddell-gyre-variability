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

def a12_hydrography(wdir):

    import gsw
    import numpy as np
    import xarray as xr
    from metpy.interpolate import interpolate_to_points
    from glob import glob
    from joblib import Parallel, delayed

    def load_stations(directory):
        prof = {}; n = 0;
        file_names = [y for x in os.walk('/scratch/e14/jn8053/cchdo_hydrography/'+directory+'/')
                      for y in glob(os.path.join(x[0], '*.nc'))]
        file_names = np.sort(file_names)
        prof[n] = xr.open_dataset(file_names[0])
        if 'CTDTMP' in prof[0].data_vars:
            temp_name = 'CTDTMP'
            salt_name = 'CTDSAL'
        else:
            temp_name = 'temperature'
            salt_name = 'salinity'
        prof[n] = prof[n].rename({temp_name:'temp', salt_name:'salt'})
        n += 1
        for file in file_names[1:]:
            prof[n] = xr.open_dataset(file)
            if 'CTDTMP' in prof[n].data_vars:
                temp_name = 'CTDTMP'
                salt_name = 'CTDSAL'
            else:
                temp_name = 'temperature'
                salt_name = 'salinity'
            prof[n] = prof[n].rename({temp_name:'temp', salt_name:'salt'})
            n += 1
        return prof

    def interpolate_to_pressure(profiles, var_name):

        def get_shallow_stations(profiles, var, ht):
            """
            Drops stations shallower than 1000m from the array
            """
            N = len(profiles)
            idx = []
            for st in range(0, N, 1):
                dpth = ht.sel(xt_ocean = profiles[st]['longitude'],
                              yt_ocean = profiles[st]['latitude'],
                              method = 'nearest')
                nans = np.isnan(var[:, st]).sum()
                # Depth of deepest station
                z = gsw.z_from_p(profiles[st]['pressure'][-1],
                                 profiles[st]['latitude'])
                # Distance of deepest station from the bottom
                dist = dpth - z.values
                if nans > 0.8*dist:
                    idx.append(st)
            return idx

        N = len(profiles)
        prs = np.arange(0, 6001, 1)
        var = np.empty([len(prs), N])*np.nan
        lon = np.empty(N)*np.nan
        lat = np.empty(N)*np.nan
        tim = [profiles[0]['time'].values]
        for st in range(0, N, 1):
            prf = profiles[st][var_name].dropna(dim = 'pressure')
            var[:, st] = prf.interp(pressure = prs, method = 'nearest')
            lon[st] = profiles[st]['longitude']
            lat[st] = profiles[st]['latitude']
            if st != 0:
                tim.append(profiles[st]['time'].values)

        idx = get_shallow_stations(profiles, var, ht)
        var = np.delete(var, idx, axis = 1)
        lon = np.delete(lon, idx)
        lat = np.delete(lat, idx)
        tim = np.delete(tim, idx)

        var_xr = xr.DataArray(var, name = var_name, dims = ['pressure', 'station'],
                              coords = {'pressure':prs,
                                        'station':np.arange(0, len(lon), 1)})
        var_xr.expand_dims({'lon':lon, 'lat':lat, 'time':tim})
        lon_xr = xr.DataArray(lon, name = 'lon', dims = ['station'],
                              coords = {'station':np.arange(0, len(lon), 1)})
        lat_xr = xr.DataArray(lat, name = 'lat', dims = ['station'],
                              coords = {'station':np.arange(0, len(lon), 1)})
        tim_xr = xr.DataArray(tim, name = 'time', dims = ['station'],
                              coords = {'station':np.arange(0, len(lon), 1)})
        dset = xr.merge([var_xr, lon_xr, lat_xr, tim_xr])

        return dset

    def a12_repeat_section(dset, var):

        def project_u_on_v(u, vi, vf):

            v = np.array([vf[0]-vi[0], vf[1]-vi[1]])
            u = np.transpose(np.array([u[0]-vi[0], u[1]-vi[1]]))
            u_proj = [np.dot(u, v)/(np.sum(v**2))*v[0]+vi[0], np.dot(u, v)/(np.sum(v**2))*v[1]+vi[1]]
            return u_proj

        def distance_from_start_A12(x, y):
            dist = [0]
            xini = -0
            yini = -50
            for i in range(1, len(x)):
                dist.append(np.sqrt((x[i] - xini)**2 + (y[i] - yini)**2))
            dist = np.squeeze(dist)

            return dist

        def projection_onto_A12(dset, var, st, p, x, y, xi, yi):

            interpolated = interpolate_to_points((x, y), dset[var][p, st].values, (xi, yi), interp_type = 'nearest')

            return interpolated

        def interpolation_to_standard_A12(var_array, var_name, pressure, Xi, Xf):

            dist_coords = distance_from_start_A12(Xi[0], Xi[1])
            dist_interp = distance_from_start_A12(Xf[0], Xf[1])
            # Drop duplicates
            dist_unique, dist_unique_idx = np.unique(dist_coords, return_index = True)

            # Create an xarray with distance from start as coordinate. We will interpolate in this dimension
            var_xarray = xr.DataArray(var_array[:, dist_unique_idx], name = var_name, dims = ['pressure', 'distance'], coords = {'pressure':pressure, 'distance':dist_unique})
            var_interp_xarray = var_xarray.interp(distance = dist_interp, method = 'nearest')
            # Change dimension distance for a station number
            var_interp_xarray['distance'] = np.arange(0, len(Xf[0]), 1)
            var_interp_xarray = var_interp_xarray.rename({'distance':'station'})
            xf_xarray = xr.DataArray(Xf[0], name = 'lon', dims = ['station'], coords = {'station':var_interp_xarray['station']})
            yf_xarray = xr.DataArray(Xf[1], name = 'lat', dims = ['station'], coords = {'station':var_interp_xarray['station']})
            dset_output = xr.merge([var_interp_xarray, xf_xarray, yf_xarray])

            return dset_output

        dset = dset.sortby('lat')
        idx_A12 = np.where((dset['lon']<=2) & (dset['lon']>=-2) &
                           (dset['lat']>-75) & (dset['lat']<=-55))[0]
        # Projections onto SR04
        pr = project_u_on_v((dset['lon'][idx_A12], dset['lat'][idx_A12]),
                            ((0, -69)), (-0, -55))
        xi = pr[0]
        yi = pr[1]
        # Original points
        xori = dset['lon'][idx_A12]
        yori = dset['lat'][idx_A12]
        # Remove points that are > 0.7 degrees away from the transect
        dist = np.sqrt((xori-xi)**2 + (yori-yi)**2)
        if np.shape(np.where(dist.values > 0.5)[0] != 0):
            xi = np.delete(xi, np.where(dist.values > 0.5)[0])
            yi = np.delete(yi, np.where(dist.values > 0.5)[0])
            xori = np.delete(xori, np.where(dist.values > 0.5)[0])
            yori = np.delete(yori, np.where(dist.values > 0.5)[0])
            idx_A12 = np.delete(idx_A12, np.where(dist.values > 0.5)[0])

        # Project to A12 (Xori onto Xi)
        dset_on_A12 = Parallel(n_jobs = -1)(delayed(projection_onto_A12)(dset, var, idx_A12, p, xori, yori, xi, yi) for p in range(len(dset['pressure'])))
        dset_on_A12 = np.squeeze(dset_on_A12)

        # Define a standard A12
        xf = np.zeros(20)
        yf = np.linspace(-69, -55, 20)

        var_interp_xarray = interpolation_to_standard_A12(dset_on_A12, var, dset['pressure'], (xi, yi), (xf, yf))
        var_uninterp_xarray = xr.DataArray(dset[var][:, idx_A12], name = var, dims = dset.dims, coords = {'pressure':dset['pressure'], 'station':idx_A12})
        var_uninterp_xarray = xr.merge([var_uninterp_xarray, dset['lon'][idx_A12], dset['lat'][idx_A12], dset['time'][idx_A12]])

        return var_interp_xarray

    direc = ['a12_nc_ctd', 's04a_nc_ctd', 'sr04_e_nc_ctd', 'a12_1999a_nc_ctd', '06AQ20050122_nc_ctd', '06AQ20071128_nc_ctd', '06AQ20080210_nc_ctd', '06AQ20101128', '06AQ20141202_nc_ctd']
    ht = xr.open_dataset(wdir+'/raw_outputs/ht-01deg.nc')['ht']
    ht = ht.sel(xt_ocean = slice(-30, 30), yt_ocean = slice(-80, 50))
    for d in direc:
        prof = load_stations(d)
        temp = interpolate_to_pressure(prof, 'temp')
        salt = interpolate_to_pressure(prof, 'salt')
        temp_interp = a12_repeat_section(temp, 'temp')
        salt_interp = a12_repeat_section(salt, 'salt')
        if d == direc[0]:
            temp_A12 = temp_interp
            salt_A12 = salt_interp
        else:
            temp_A12 = xr.concat([temp_A12, temp_interp], dim = 'cruise')
            salt_A12 = xr.concat([salt_A12, salt_interp], dim = 'cruise')

    hydro = xr.merge([temp_A12.mean(dim = 'cruise'),
                      salt_A12.mean(dim = 'cruise')])
    return hydro

def a12_mean_pot_rho(dset, hydro_or_model):

    import gsw
    import xarray as xr

    salt = dset['salt']
    pres = dset['pressure']
    lon = dset['lon']
    lat = dset['lat']
    salt_abs = gsw.SA_from_SP(salt, pres, lon, lat)
    if hydro_or_model == 'hydrography':
        temp_csv = gsw.CT_from_t(salt_abs, dset['temp'], pres)
    elif hydro_or_model == 'model':
        temp_csv = gsw.CT_from_pt(salt_abs, dset['pot_temp'])
    pot_rho = gsw.sigma0(salt_abs, temp_csv)
    pot_rho = xr.DataArray(pot_rho, name = 'pot_rho')
    return pot_rho

def a12_model(keys, hydro, wdir):

    import gsw
    import numpy as np
    import xarray as xr

    time_select = ['1992-06-15T00:00:00.000000000',
                   '1992-07-15T00:00:00.000000000',
                   '1996-03-15T00:00:00.000000000',
                   '1996-04-15T00:00:00.000000000',
                   '1996-05-15T00:00:00.000000000',
                   '1998-05-15T00:00:00.000000000',
                   '1999-01-15T00:00:00.000000000',
                   '1999-02-15T00:00:00.000000000',
                   '1999-03-15T00:00:00.000000000',
                   '2005-02-15T00:00:00.000000000',
                   '2005-03-15T00:00:00.000000000',
                   '2007-12-15T00:00:00.000000000',
                   '2008-01-15T00:00:00.000000000',
                   '2008-02-15T00:00:00.000000000',
                   '2008-03-15T00:00:00.000000000',
                   '2010-12-15T00:00:00.000000000',
                   '2011-01-15T00:00:00.000000000',
                   '2014-12-15T00:00:00.000000000',
                   '2015-01-15T00:00:00.000000000']
    model = {};
    for k in keys[0:1]:

        temp_access = xr.open_dataset(wdir+'/raw_outputs/pot_temp-monthly-A12-'+k+'deg.nc')['pot_temp'] - 273.15
        temp_t = temp_access.sel(time = time_select[0], method = 'nearest')
        for t in time_select[1:]:
            T = temp_access.sel(time = t, method = 'nearest')
            temp_t = xr.concat([temp_t, T], dim = 'time')
        temp_t = temp_t.mean(dim = 'time')
        temp = temp_t.interp(xt_ocean = hydro['lon'], yt_ocean = hydro['lat'])
        p = gsw.p_from_z(-temp['st_ocean'], -65)
        temp['st_ocean'] = p.values
        temp = temp.rename({'st_ocean':'pressure'})
        temp = temp.interp(pressure = np.arange(0, 6001, 1))

        salt_access = xr.open_dataset(wdir+'/raw_outputs/salt-monthly-A12-'+k+'deg.nc')['salt']
        salt_t = salt_access.sel(time = time_select[0],  method = 'nearest')
        for t in time_select[1:]:
            S = salt_access.sel(time = t, method = 'nearest')
            salt_t = xr.concat([salt_t, S], dim = 'time')
        salt_t = salt_t.mean(dim = 'time')
        salt = salt_t.interp(xt_ocean = hydro['lon'],
                                yt_ocean = hydro['lat'])
        p = gsw.p_from_z(-salt['st_ocean'], -65)
        salt['st_ocean'] = p.values
        salt = salt.rename({'st_ocean':'pressure'})
        salt = salt.interp(pressure = np.arange(0, 6001, 1))

        model[k] = xr.merge([temp, salt])

    return model

def potential_vorticity(keys, wdir):
    import numpy as np
    import xarray as xr
    pvor = {}
    for k in keys:
        ht = xr.open_dataset(wdir+'ocean_grid-'+k+'deg.nc')['ht'].sel(xt_ocean = slice(-70, 80), yt_ocean = slice(-80, -50))
        pvor[k] = 2*7.292e-5*np.sin(ht['yt_ocean']*np.pi/180)/ht

    pvor['1'] = pvor['1'].rolling(xt_ocean = 2).mean().rolling(yt_ocean = 2).mean()
    pvor['025'] = pvor['025'].rolling(xt_ocean = 8).mean().rolling(yt_ocean = 8).mean()
    pvor['01'] = pvor['01'].rolling(xt_ocean = 20).mean().rolling(yt_ocean = 20).mean()
    return pvor
