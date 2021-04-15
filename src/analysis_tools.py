###############################################################################
# Description	: Analysis functions
# Args          :
#     wdir = path to directory in which save data
# Author        : Julia Neme
# Email         : j.neme@unsw.edu.au
###############################################################################


###############################################################################

def g02202_aice(month):

    import glob
    import os
    # from http://nsidc.org/data/G02202
    path = '/g/data3/hh5/tmp/cosima/observations/NOAA/G02202_V3'
    files = glob(os.path.join(path, 'south/monthly/*.nc'))
    drop = ['projection', 'seaice_conc_monthly_cdr',
            'stdev_of_seaice_conc_monthly_cdr',
            'melt_onset_day_seaice_conc_monthly_cdr',
            'qa_of_seaice_conc_monthly_cdr', 'goddard_nt_seaice_conc_monthly', 'goddard_bt_seaice_conc_monthly']
    aice = xr.open_mfdataset(files, drop_variables = drop)

    aice = aice['goddard_merged_seaice_conc_monthly']
    aice = aice.groupby('time.month').mean(dim = 'time')
    if month == 'feb':

    elif month == 'sep':

def edof():
    """
    Calculates effective degrees of freedom using the integral timescale as per
    Emery and Thompson's book.
    """
    import numpy as np
    from scipy import integrate

    if len(np.shape(A)) == 1:

        N = len(A)
        N1 = N-1
        x = A - A.mean(dim = 'time')
        c = np.correlate(x, x, 'full')
        c = c[N1:]/(N-1-np.arange(0, N1+1, 1))
        n = 0
        while (c[n] > 0) and (n < N/2):
            n = n+1
        T = integrate.trapz(c[:n])/c[0]
        edof = N/(2*T)

    else:

        N = len(A['time'])
        N1 = N-1
        A = A - A.mean(dim = 'time')

        points_ocean = np.where(A.isel(time = 0) != np.isnan(A.isel(time = 0)))
        random_x = np.random.randint(0, len(A['xt_ocean']), size = 100)
        random_y = np.random.randint(0, len(A['yt_ocean']), size = 100)

        edof = np.empty(np.shape(A.isel(time = 0)))*np.nan
        for i in range(0, len(A['xt_ocean'])):
            for j in range(0, len(A['yt_ocean'])):
                if any(np.isnan(A.isel(xt_ocean = i, yt_ocean = j))):
                    pass
                else:
                    x = A.isel(xt_ocean = i, yt_ocean = j)
                    c = np.correlate(x, x, 'full')
                    c = c[N1:]/(N-1-np.arange(0, N1+1, 1))
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
    edof_m = np.nanmax([edof_A, edof_B])

    A, B = xr.align(A, B)
    if lagB:
        B = B.shift(time = lagB).dropna('time', how = 'all')
        A, B = xr.align(A, B)
    n = len(A['time'])
    Am = A.mean(dim = 'time')
    Bm = B.mean(dim = 'time')
    As = A.std(dim = 'time')
    Bs = B.std(dim = 'time')
    covarnc = np.sum((A - Am)*(B - Bm), axis = 0) / (n-1)
    rvalues = covarnc/(As * Bs)
    t_s = rvalues * np.sqrt(edof_m)/np.sqrt(1 - rvalues**2)
    pvalues = t.sf(np.abs(t_s), edof_m)*2
    pvalues = xr.DataArray(pvalues, dims = rvalues.dims, coords = rvalues.coords)

    return rvalues, pvalues

def load_sea_level(keys):

    """
    Returns sea level from obs and model interpolated to the obs grid, with the
    artificial global annual cycle removed and an offset applied.
    """

    obs = xr.open_dataset(wdir+'/CS2_combined_Southern_Ocean_2011-2016_regridded.nc')
    obs = obs.rename(({'X':'xt_ocean', 'Y':'yt_ocean', 'date':'time'}))
    # Flip latitude
    obs = obs.sortby('yt_ocean', ascending = True)
    # Correct longitude
    obs['_lon_adj'] = xr.where(obs['xt_ocean'] > 80, obs['xt_ocean'] - 360,
                               obs['xt_ocean'])
    obs = (obs.swap_dims({'xt_ocean': '_lon_adj'})
              .sel(**{'_long_adj': sorted(obs._lon_adj)}).drop('xt_ocean'))
    obs = obs.sel(**{'_lon_adj': sorted(obs._lon_adj)}).drop('xt_ocean'))
    eta_obs = obs.rename({'_lon_adj': 'xt_ocean'})

    eta_mod = {}
    for k in keys:
        eta_mod[k] = xr.open_dataset(wdir+'/sea_level-monthly-1958_2018-'+k+'deg.nc')['sea_level']
        eta_mod[k] = eta_mod[k].sel(time = slice('2011-01-01', '2017-01-01'))
        eta_mod[k] = eta_mod[k]*100

        # Filter an artifical global sea level annual cycle
        eta_glo = xr.open_dataset(wdir+'/eta_global-monthly-1958_2018-'+k+'deg.nc')['eta_global'].squeeze()
        eta_mod[k] = eta_mod[k].groupby('time.month') -
                        eta_glo.groupby('time.month').mean(dim = 'time')

        # Offset
        mask = (eta_obs['MDT'] != 0)
        eta_interp = eta_mod[k].interp(xt_ocean = eta_obs['MDT']['xt_ocean'],
                                       yt_ocean = eta_obs['MDT']['yt_ocean'])

        mean_mod = eta_interp.where(mask == True).mean()
        mean_obs = eta_obs['MDT'].where(mask == True).mean()
        offset = mean_obs - mean_mod
        eta_mod[k] = (η_in + offs).sel(xt_ocean = slice(-70, 80),
                                       yt_ocean = slice(-80, -50))
    eta_obs = eta_obs.sel(xt_ocean = slice(-70, 80), yt_ocean = slice(-80, -50))

    return eta_mod, eta_obs

def get_gyre_boundary(keys, which_field):

    import matplotlib.pyplot as plt

    bndy = {}
    for k in keys:
        psi_b = xr.open_dataset(wdir+'/psi_b-monthly-1958_2018-'+k+'deg.nc')

        if which_field == 'mean':
            psi_b = psi_b['psi_b'].mean(dim = 'time')
            cset = plt.contour(psi_b['xu_ocean'], psi_b['yt_ocean'], psi_b,
                               levels = [-12])
            if k == '01':
                bndy[k] = cset.allsegs[0][3]
            else:
                bndy[k] = cset.allsegs[0][0]
            plt.close('all')

    return bndy

def load_temp_salt_hydrography():

    import gsw
    import numpy as np
    import os
    from glob import glob

    # HAVE TO SEE HOW I MAKE THESE AVAILABLE
    files = [y for x in os.walk('/scratch/e14/jn8053/cchdo_hydrography/')
             for y in glob(os.path.join(x[0], '*.nc'))]
    station = {}; n = 0;
    for file in files:
        station[n] = xr.open_dataset(file)
        n += 1
    N = len(station)

    p = np.arange(0, 6000, 10)
    PT = np.empty([len(p), N])*np.nan # Potential temperature
    PS = np.empty([len(p), N])*np.nan # Practical salinity
    lon = np.empty(N)
    lat = np.empty(N)
    for i in range(0, N):
        if 'CTDSAL' in profiles[i].data_vars:

            # Drop duplicates
            p_unique, p_unique_idx = np.unique(station[i]['pressure'],
                                               return_index = True)
            PS[:, i] = station[i]['CTDSAL'][p_unique_idx].interp(pressure = p)
            temp = station[i]['CTDTMP'][p_unique_idx].interp(pressure = p)
            AS = gsw.SA_from_SP(PS[:,i], p, station[i]['longitude'].values,
                                station[i]['latitude'].values)
            PT[:, i] = gsw.pt0_from_t(AS, temp, p)

        elif 'salinity' in profiles[i].data_vars:

            p_unique, p_unique_idx = np.unique(profiles[i]['pressure'],
                                               return_index = True)
            PS[:, i] = station[i]['salinity'][p_unique_idx].interp(pressure = p)
            temp = station[i]['temperature'][p_unique_idx].interp(pressure = p)
            AS = gsw.SA_from_SP(PS[:,i], p, station[i]['longitude'].values,
                                station[i]['latitude'].values)
            PT[:, i] = gsw.pt0_from_t(AS, temp, p)

        lon[i] = station[i]['longitude'].values
        lat[i] = station[i]['latitude'].values

    PS_xarray = xr.DataArray(PS, name = 'salinity',
                             dims = ['pressure', 'station'],
                             coords = {'pressure':p,
                                       'station':np.arange(0, len(lon), 1)})
    PT_xarray = xr.DataArray(PT, name = 'pot_temp',
                             dims = ['pressure', 'station'],
                             coords = {'pressure':p,
                                       'station':np.arange(0, len(lon), 1)})
    lon_array = xr.DataArray(lon, name = 'lon', dims = ['station'],
                             coords = {'station':np.arange(0, len(lon), 1)})
    lat_array = xr.DataArray(lat, name = 'lat', dims = ['station'],
                             coords = {'station':np.arange(0, len(lon), 1)})
    ts_hydro = xr.merge([PS_xarray, PT_xarray, lon_array, lat_array])

    return ts_hydro

def load_temp_salt_model(key):

    # CORRECT HOW WE LOAD..
    for k in keys:
        if k == '01':
            temp = xr.open_dataset(wdir+'/pot_temp_hydrography-'+k+'deg.nc')['pot_temp'] - 273.15
            salt = xr.open_dataset(wdir+'data/raw_outputs/salt_hydrography-'+k+'deg.nc')['salt']
        else:
            temp = xr.open_dataset(wdir+'data/raw_outputs/pot_temp-monthly-1958_2018-'+k+'deg.nc')['pot_temp'] - 273.15
            salt = xr.open_dataset(wdir+'data/raw_outputs/salt-monthly-1958_2018-'+k+'deg.nc')['salt']

    files = [y for x in os.walk('/scratch/e14/jn8053/cchdo_hydrography/')
             for y in glob(os.path.join(x[0], '*.nc'))]
    station = {}; n = 0;
    for file in files:
        station[n] = xr.open_dataset(file)
        n += 1
    N = len(station)

    # Get years + month of hydrography profiles
    ym = [[station[0]['time.year'].item(), station[0]['time.month'].item()]]
    for i in range(1, N):
        ym.append([station[i]['time.year'].item(),
                   station[i]['time.month'].item()])
    ym = np.array(ym)

    # Get years + month that don't repeat and grab that month from the model
    unique = np.unique(ym, axis = 0)
    t_unique = temp.sel(time = str(unique_times[0, 0])+'-'+f'{unique[0, 1]:02}',
                        method = 'nearest')
    s_unique = salt.sel(time = str(unique_times[0, 0])+'-'+f'{unique[0, 1]:02}',
                        method = 'nearest')

    # Get all the lat/lon from profiles from that year + month and select the
    # nearest gridpoint from the model
    idx = np.where((ym == unique[0, :]).all(axis = 1))[0]
    lat_model = xr.DataArray(lat[idx], dims = 'station')
    lon_model = xr.DataArray(lon[idx], dims = 'station')
    temp_model = t_unique.sel(xt_ocean = lon_model, yt_ocean = lat_model,
                              method = 'nearest').squeeze()
    salt_model = s_unique.sel(xt_ocean = lon_model, yt_ocean = lat_model,
                              method = 'nearest').squeeze()

    # Iterate over the rest of the dates!
    for i in range(1, len(unique[:, 0])):
        t_unique = temp.sel(time = str(unique[i, 0])+'-'+f'{unique[i, 1]:02}',
                            method = 'nearest')
        s_unique = salt.sel(time = str(unique[i, 0])+'-'+f'{unique[i, 1]:02}',
                            method = 'nearest')
        idx = np.where((ym == unique[i, :]).all(axis = 1))[0]
        lat_model = xr.DataArray(lat[idx], dims = 'station')
        lon_model = xr.DataArray(lon[idx], dims = 'station')
        temp_model = xr.concat([temp_model, t_unique.sel(xt_ocean = lon_model,
                                yt_ocean = lat_model,
                                method = 'nearest').squeeze()], dim = 'station')
        salt_model = xr.concat([salt_model, s_uinque.sel(xt_ocean = lon_model,
                                yt_ocean = lat_model,
                                method = 'nearest').squeeze()], dim = 'station')


    # Switch depth for pressure and interpolate

    p = gsw.p_from_z(-temp_model['st_ocean'], -65)
    temp_model['st_ocean'] = p
    temp_model = temp_model.rename({'st_ocean':'pressure'})
    temp_model = temp_model.interp(pressure = np.arange(0, 6000, 10))
    temp_model = temp_model.sortby('station')
    salt_model['st_ocean'] = p
    salt_model = salt_model.rename({'st_ocean':'pressure'})
    salt_model = salt_model.interp(pressure = np.arange(0, 6000, 10))
    salt_model = salt_model.sortby('station')

    ts_model = xr.merge([temp_model, salt_model])
    ts_model['pot_temp'] = ts_model['pot_temp'].where(ts_hydro['pot_temp'] != np.nan)
    ts_model['salt'] = ts_model['salt'].where(ts_hydro['salinity'] != np.nan)

    return ts_model

def a12_hydrography():

    import gsw
    from metpy.interpolate import interpolate_to_points
    from joblib import Parallel, delayed

    def load_stations(directory):
        import glob
        station = {}; n = 0;
        file_names = [y for x in os.walk(''+directory+'/')
                     for y in glob.glob(os.path.join(x[0], '*.nc'))]
        file_names = np.sort(file_names)
        station[n] = xr.open_dataset(file_names[0])
        n += 1
        if 'CTDTMP' in station[0].data_vars:
            temp_name = 'CTDTMP'
            salt_name = 'CDTSAL'
        else:
            temp_name = 'temperature'
            salt_name = 'salinity'
        station[n] = station[n].rename({temp_name: 'temp', salt_name = 'salt'})
        for file in file_names[1:]:
            station[n] = xr.open_dataset(file)
            station[n] = station[n].rename({temp_name: 'temp',
                                            salt_name = 'salt'})
            n += 1
        return station

    def interp_to_pressure(stations):
        N = len(stations)
        temp = np.empty([len(p_interp), N])*np.nan
        salt = np.empty([len(p_interp), N])*np.nan
        lon = np.empty(N)*np.nan
        lat = np.empty(N)*np.nan
        tim = [station[0]['time'].values]
        for st in range(0, N, 1):
            temp[:, st] = stations[st]['temp'].dropna(dim = 'pressure')
            temp[:, st] = temp[:, st].interp(pressure = p, method = 'nearest')
            salt[:, st] = stations[st]['salt'].dropna(dim = 'pressure')
            salt[:, st] = salt[:, st].interp(pressure = p, method = 'nearest')
            lon[st] = stations[st]['longitude']
            lat[st] = stations[st]['latitude']
            if st != 0:
                tim.append(stations[st]['time'].values)
        idx_shallow = []
        for st in range(0, N, 1):
            depth = ht.sel(xt_ocean = stations[st]['longitude'],
                           yt_ocean = stations[st]['latitude'],
                           method = 'nearest')
            count_nans = np.isnan(salt[:, st]).sum()
            # Depth of deepest station
            z = gsw.z_from_p(stations[st]['pressure'][-1],
                             stations[st]['latitude'])
            # Distance of deepest station from the bottom
            distance_to_bottom = depth - z.values
            if count_nans > 0.8*distance_to_bottom:
                idx_shallow.append(st)
        temp = np.delete(temp, idx_shallow, axis = 1)
        salt = np.delete(salt, idx_shallow, axis = 1)
        lon = np.delete(lon, idx_shallow)
        lat = np.delete(lat, idx_shallow)
        tim = np.delete(tim, idx_shallow)

        temp_array = xr.DataArray(temp, name = 'temp',
                                  dims = ['pressure', 'station'],
                                  coords = {'pressure':p,
                                  'station':np.arange(0, len(lon), 1)})
        temp_array.expand_dims({'lon':lon, 'lat':lat, 'time':tim})
        salt_array = xr.DataArray(salt, name = 'salt',
                                  dims = ['pressure', 'station'],
                                  coords = {'pressure':p,
                                  'station':np.arange(0, len(lon), 1)})
        salt_array.expand_dims({'lon':lon, 'lat':lat, 'time':tim})
        lon_array = xr.DataArray(lon, name = 'lon', dims = ['station'],
                                 coords = {'station':np.arange(0, len(lon), 1)})
        lat_array = xr.DataArray(lat, name = 'lat', dims = ['station'],
                                 coords = {'station':np.arange(0, len(lon), 1)})
        t_array = xr.DataArray(tim, name = 'time', dims = ['station'],
                               coords = {'station':np.arange(0, len(lon), 1)})
        dset = xr.merge([temp_array, salt_array lon_array, lat_array, t_array])
        dset = dset.sortby('lat')
        return dset

    def project_to_a12(dset, idx_a12):

        def projection_all_depths(dset, var, st, pr, x, y, xi, yi):
            interp_to_a12 = interpolate_to_points((x, y),
                                                  dset[var][pr, st].values,
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

        temp_a12 = Parallel(n_jobs = -1)(delayed(projection_all_depths)(dset,
                        'temp', idx_a12, p, xori, yori, xi, yi)
                        for pr in range(len(dset['pressure'])))
        temp_a12 = np.squeeze(temp_a12)

        salt_a12 = Parallel(n_jobs = -1)(delayed(projection_all_depths)(dset,
                        'salt', idx_a12, p, xori, yori, xi, yi)
                        for pr in range(len(dset['pressure'])))
        salt_a12 = np.squeeze(salt_a12)

        return temp_a12, salt_a12

    def interp_to_standard_a12(t, s):

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

        t_xarray = xr.DataArray(t[:, dist_unique_idx], name = 'temp',
                                dims = ['pressure', 'distance'],
                                coords = {'pressure':p, 'distance':dist_unique})
        t_xarray = t_xarray.interp(distance = dist_interp, method = 'nearest')
        s_xarray = xr.DataArray(s[:, dist_unique_idx], name = 'salt',
                                dims = ['pressure', 'distance'],
                                coords = {'pressure':p, 'distance':dist_unique})
        s_xarray = s_xarray.interp(distance = dist_interp, method = 'nearest')
        # Change dimension distance for a station number
        t_xarray['distance'] = np.arange(0, len(Xf[0]), 1)
        t_xarray = t_xarray.rename({'distance':'station'})
        s_xarray['distance'] = np.arange(0, len(Xf[0]), 1)
        s_xarray = s_xarray.rename({'distance':'station'})
        x_xarray = xr.DataArray(Xf[0], name = 'lon', dims = ['station'],
                                coords = {'station':s_xarray['station']})
        y_xarray = xr.DataArray(Xf[1], name = 'lat', dims = ['station'],
                                coords = {'station':s_xarray['station']})
        t_final = xr.merge([t_xarray, x_xarray, y_xarray])
        s_final = xr.merge([s_xarray, x_xarray, y_xarray])
        return t_final, s_final

    dirs = ['a12_nc_ctd', 's04a_nc_ctd', 'sr04_e_nc_ctd', 'a12_1999a_nc_ctd',
            '06AQ20050122_nc_ctd', '06AQ20071128_nc_ctd', '06AQ20080210_nc_ctd',
            '06AQ20101128', '06AQ20141202_nc_ctd']
    ht = xr.open_dataset(wdir+'/ocean_grid-01deg.nc')['ht']
    ht = ht.sel(xt_ocean = slice(-30, 30), yt_ocean = slice(-80, 50))
    p = np.arange(0, 6001, 1)

    temp_a12_st = {}; salt_a12_st = {}
    for d in dirs:
        station = load_stations(d)
        station_interp = interpolate_to_pressure(station, p)

        idx_A12 = np.where((station_interp['lon'] <= 2) &
                           (station_interp['lon'] >= -2) &
                           (station_interp['lat'] > -75) &
                           (station_interp['lat'] <= -55))[0]

        temp_a12, salt_a12 = project_to_a12(dset, idx_A12)
        temp_a12_st[d], salt_a12_st[d] = interp_to_standard_a12(temp_a12,
                                                                salt_a12)

        if d == dirs[0]:
            temp_a12_repeat = temp_a12_st[d]
            salt_a12_repeat = salt_a12_st[d]
        else:
            temp_a12_repeat = xr.concat([temp_a12_repeat, temp_a12_st[d]],
                                        dim = 'cruise')
            salt_a12_repeat = xr.concat([salt_a12_repeat, salt_a12_st[d]],
                                        dim = 'cruise')

    return temp_a12_repeat, salt_a12_repeat

def potential_density(dset):
    from gsw import SA_from_SP
    from gsw import CT_from_t
    from gsw import sigma0
    SA = SA_from_SP(dset['salt'], dset['pressure'], dset['lon'], dset['lat'])
    CT =  CT_from_t(SA, dset['temp'], dset['pressure'])
    pot_rho =  sigma0(SA, CT)
    return pot_rho
