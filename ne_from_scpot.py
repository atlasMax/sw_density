import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import constants, optimize
from pyrfu import mms, pyrf

# Fitting formula from Khotyaintsev 2021 using RPW
def _n_e_func(psp, N0, beta):
    return N0 * np.exp(psp / beta)



# Updated fit_ne function
def fit_ne(vsc_data, f_pe_peaks, std_thresh=2.5, beta_bounds=(0, 10.0), N0_bounds=(0, 5e5), weights=None, method=2, coeffs=None):
    # Convert from frequency to number density
    w_pe_peaks = 2 * np.pi * f_pe_peaks
    ep0 = constants.epsilon_0
    m_e = constants.electron_mass
    q_e = constants.elementary_charge
    n_e_peaks = w_pe_peaks**2 * ep0 * m_e / q_e**2
    n_e_peaks *= 1e-6  # Convert to cm^-3

    if vsc_data.size != n_e_peaks.size:
        print("\tResampling SCpot to peaks")
        n_e_times = n_e_peaks.time
        vsc_tofit = vsc_data.sel(time=n_e_times, method='nearest')
    else:
        vsc_tofit = vsc_data
    print("COEFFS",coeffs)
    vsc_tofit *= -1  # Ensure correct sign
    if coeffs is None:
        method =1
    else:
        a, b, sigma_a, sigma_b = coeffs
        
        psp_mean = np.mean(-vsc_tofit.data)
        beta_fallback = a*psp_mean + b
    try:
        if method == 1:
            print('Method 1')
            # Initial fit with bounds
            fit_results = optimize.curve_fit(
                _n_e_func, vsc_tofit, n_e_peaks, bounds=([N0_bounds[0], beta_bounds[0]], [N0_bounds[1], beta_bounds[1]])
            )
            N0, beta = fit_results[0]
        elif method ==2:
            # IF BETA BAD, ONLY FIT N
            # No bounds, check beta value
            fit_results = np.polyfit(vsc_tofit, np.log(n_e_peaks), deg=1, cov=True)
            b, a = fit_results[0]
            N0 = np.exp(a)
            beta = 1/b
            # fit_results = optimize.curve_fit(_n_e_func, vsc_tofit, n_e_peaks)#,  sigma=weights, absolute_sigma=True)
            # N0, beta = fit_results[0] 
            print(f'\t2 BETA: {beta}')
            fit_covar1 = fit_results[1]
            # If beta bad, take it from stats and only fit N
            beta_diff = abs(beta - beta_fallback)
            print('BDIFF:',beta_diff)
            if beta_diff >= 1.4:
 
                psp_median = np.median(-vsc_tofit.data)
                print(f'\tUsing mean of Vsc to fit: {psp_median}')
                # beta = 0.43 * psp_median - 0.05
                beta = beta_fallback
                print(f'\t2 BETA AFTER: {beta}')
                fit_results = optimize.curve_fit(
                    lambda psp_, N0_: _n_e_func(psp_, N0_, beta), vsc_tofit, n_e_peaks
                )
                N0  = fit_results[0][0] 
                fit_covar1 = fit_results[1]
        elif method ==3:
            # ONLY FIT N
            print('\tMETHOD 3')
            print(f'\tUsing mean of Vsc to fit, BETA: {beta_fallback}')
            # beta = 0.43 * psp_median - 0.05
            beta = beta_fallback
            print(f'\t3 BETA: {beta}')
            # fit_results = optimize.curve_fit(
            #     lambda psp_, N0_: _n_e_func(psp_, N0_, beta), vsc_tofit, n_e_peaks
            # )
            a = np.mean(np.log(n_e_peaks.data) - vsc_tofit.data/beta)
            N0 = np.exp(a)
            print(20*'+', beta,N0)
            
            fit_covar1 = 0
        # Determine outliers
        n_e_fit = _n_e_func(vsc_tofit, N0, beta)
        residuals = n_e_fit.data - n_e_peaks.data
        std = (residuals - np.median(residuals)) / np.std(residuals)
        outliers_idxs = np.abs(std) > std_thresh

        # Refit without outliers
        n_e_nout = n_e_peaks[~outliers_idxs]
        psp_nout = vsc_tofit[~outliers_idxs]
        # if weights is None:
        #     fit_results = optimize.curve_fit(
        #         _n_e_func, psp_nout, n_e_nout, bounds=([N0_bounds[0], beta_bounds[0]], [N0_bounds[1], beta_bounds[1]]))
        # else:
        #     fit_results = optimize.curve_fit(
        #         _n_e_func, psp_nout, n_e_nout, bounds=([N0_bounds[0], beta_bounds[0]], [N0_bounds[1], beta_bounds[1]]),
        #         sigma=weights[~outliers_idxs], absolute_sigma=True
        # )
        
      
        if method == 1:
            N0_fit, beta_fit = fit_results[0]
            fit_covar = fit_results[1]
        elif method >= 2:
            N0_fit, beta_fit = N0, beta
            fit_covar = fit_covar1
 

        # Final fit
        n_e_fit = _n_e_func(vsc_tofit, N0_fit, beta_fit)

        return vsc_tofit, n_e_fit, outliers_idxs, N0_fit, beta_fit, fit_covar

    except (RuntimeError, ValueError) as e:
        print(f"fit_ne : Curve fitting failed with error: {e}")
        return None, None, None, None, None, None
