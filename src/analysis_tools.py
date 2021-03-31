###############################################################################
# Description	: Analysis functions
# Args          :
#     wdir = path to directory in which save data
# Author        : Julia Neme
# Email         : j.neme@unsw.edu.au
###############################################################################

import xarray as xr

###############################################################################

def edof():
    """
    Calculates effective degrees of freedom using the integral timescale as per
    Emery and Thompson's book.
    """
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
