# 3rd party imports
import numpy as np
from pyrfu.plot import plot_line, plot_spectr
import matplotlib.pyplot as plt

from pyrfu import pyrf, mms

# Local imports
from ne2fpe import ne2fpe

# Fitting formula from Khotyaintsev 2021 using RPW
def _vsc2ne_fit(ne, vsc):
    # Fit 1st order polynomial to electron number density 'ne' and spacecraft potential 'vsc'
    # according to empirical relationship (Khotyaintsev et al. 2021)
    # ne        = N0*exp(-Vsc / beta)
    # log(ne)   = log(N0) - Vsc / beta
    # y         = a + b*x

    log_ne = np.log(ne)
    fit_results = np.polyfit(vsc, log_ne, deg=1, cov=True)
    b, a = fit_results[0]
    covar = fit_results[1]

    # Convert coefficients to fit exponential form of equation
    N0_fit = np.exp(a)
    beta_fit = -1/b

    return N0_fit, beta_fit, covar


def _calibrate_density(ne_data, vsc):
    converged = False
    itr = 0
    vsc_tofit = vsc
    while not converged:
  
        N0, beta, covar = _vsc2ne_fit(ne_data, vsc_tofit)
        print(N0, beta)

        ne_fit = N0 * np.exp(-vsc_tofit / beta)
        residuals = np.log(ne_fit.data / ne_data.data)
        
        # Select points
        idxs_to_keep = residuals < 0.5
        ne_data = ne_data[idxs_to_keep]
        vsc_tofit = vsc_tofit[idxs_to_keep] 
        # Check convergence for iterations beyond first
        if itr == 0:
            N0_old = N0
            beta_old = beta
        elif itr > 15:
            print('\tCould not converge.')
            break
        else:
            N0_conv = abs(N0 - N0_old)/N0_old < 1e-3
            beta_conv = abs(beta - beta_old)/beta_old < 1e-3
            if N0_conv & beta_conv:
                converged = True

        N0_old = N0
        beta_old = beta
        itr += 1

    print(f'\tConverged after {itr} iterations')
    ne_final = ne_data
    N0_final = N0
    beta_final = beta
    vsc_final = vsc_tofit
    return ne_final, vsc_final, N0_final, beta_final


def fpeaks_scpot2ne(f_peaks, vsc):
    # Convert fpe [Hz] --> ne [cm^-3]
    ne_data = ne2fpe(f_peaks, inverse=True)

    # Resample
    vsc = pyrf.resample(vsc, ne_data.time, f_s=pyrf.calc_fs(ne_data))

    # Ensure time series don't contain inf
    valid_idxs = np.isfinite(vsc.data) & np.isfinite(ne_data.data)
    vsc = vsc[valid_idxs]
    ne_data = ne_data[valid_idxs]
    
    # Calibrate density from fpe peaks against scpot
    ne, vsc, N0, beta = _calibrate_density(ne_data, vsc)

    return ne, vsc, N0, beta