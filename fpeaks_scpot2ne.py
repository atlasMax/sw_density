# 3rd party imports
import numpy as np
from pyrfu import pyrf
# Local imports
from ne2fpe import ne2fpe

# Fitting formula from Khotyaintsev 2021 using RPW
def _vsc2ne_fit(ne, vsc, weighted=None):
    # Fit 1st order polynomial to electron number density 'ne' and spacecraft potential 'vsc'
    # according to empirical relationship (Khotyaintsev et al. 2021)
    # ne        = N0*exp(-Vsc / beta)
    # log(ne)   = log(N0) - Vsc / beta
    # y         = a + b*x
    if len(ne) < 10 or len(vsc) < 10 or len(ne) != len(vsc):
        return None, None, None, 2  # exit_code 2
    log_ne = np.log(ne)

    if weighted is not None:
        # logging.info('\tWEIGHTED FIT')
        weights_freq = log_ne.data**1 / np.max(log_ne.data)**1
        # weights_freq = np.exp((log_ne.data / np.max(log_ne.data))**2)
        
        rolling_std = log_ne.rolling(time=10, center=True).std()
        # Replace NaNs with max value to down-weight uncertain regions 
        rolling_std = rolling_std.fillna(rolling_std.max())
        weights_stability = 1 / (rolling_std.data + 1e-5)
        weights_stability /= weights_stability.max()
        
        weighted = weights_freq * weights_stability
        if len(weighted) != len(vsc):  # redundant but safe
            raise ValueError("Weight array shape mismatch.")
    try:
        fit_results = np.polyfit(vsc, log_ne, deg=1, cov=True, w=weighted)
    except (np.linalg.LinAlgError, ValueError) as e:
        exit_code = 2
        print('LINALG ERROR', e)
        return None, None, None, exit_code
    b, a = fit_results[0]
    covar = fit_results[1]

    # Convert coefficients to fit exponential form of equation
    N0_fit = np.exp(a)
    beta_fit = -1/b
    exit_code = 0
    if beta_fit <= 0 or N0_fit >= 1000:
        # logging.info(f'WARNING: (N, beta) = ({N0_fit:.2f}, {beta_fit:.2f}), using fallback from <vsc>')
        # Resort to beta relationship to <vsc>
        vsc_avg = np.mean(vsc_o.data)
        beta_approx = 0.6 * vsc_avg - 0.7
        beta_fit = beta_approx
        
        # N0 given by mean
        log_N0_fit = np.mean(log_ne.data + vsc.data / beta_approx)
        N0_fit = np.exp(log_N0_fit)
        exit_code = 1

    return N0_fit, beta_fit, covar, exit_code


def _calibrate_density(ne_data, vsc):
    converged = False
    itr = 0
    vsc_tofit = vsc
    while not converged:
        if len(vsc_tofit) < 10:
            return None, None, None, None
        N0, beta, covar, exit_code = _vsc2ne_fit(ne_data, vsc_tofit)
        # logging.info(N0, beta)
        if exit_code == 2 or beta is None:
            return None, None, None, None
        ne_fit = N0 * np.exp(-vsc_tofit / beta)
        
        # Check convergence for iterations beyond first
        if itr == 0:
            N0_old = N0
            beta_old = beta
        elif itr > 50:
            return None, None, None, None
        else:
            N0_conv = abs(N0 - N0_old)/N0_old < 1e-3
            beta_conv = abs(beta - beta_old)/beta_old < 1e-3
            if N0_conv & beta_conv:
                converged = True
                # Once converged, perform final fit to remove points based on std
                # ne_fit = N0 * np.exp(-vsc_tofit / beta)
                
                residuals = np.log(ne_fit.data / ne_data.data)
                idxs_to_keep = np.abs(residuals) < 2*np.std(residuals) 
                if np.sum(idxs_to_keep) < 5:
                    return None, None, None, None
                N0, beta, covar, exit_code = _vsc2ne_fit(ne_data[idxs_to_keep], vsc_tofit[idxs_to_keep])#, weighted=True)
                # logging.info('Final fit: ', N0, beta)
                        
                
        # Select points
        if exit_code == 0:
            # Normal
            # Residuals in log-scale to treat under/over estimation equally
            residuals = np.log(ne_fit.data / ne_data.data)
            
            idxs_to_keep = residuals < 0.3 # standard
            
        elif exit_code == 1:
            # Refit with weight toward higher freq points
            N0, beta, covar, exit_code = _vsc2ne_fit(ne_data, vsc_tofit, weighted=True)
            ne_fit = N0 * np.exp(-vsc_tofit / beta)
            
            # Residuals in log-scale to treat under/over estimation equally
            residuals = np.log(ne_fit.data / ne_data.data)
            
            idxs_to_keep = residuals < 0.3 # stanard

         
            
        # Update 
        ne_data = ne_data[idxs_to_keep]
        vsc_tofit = vsc_tofit[idxs_to_keep] 
        
        N0_old = N0
        beta_old = beta
        itr += 1
        
    
    if converged:
        # logging.info(f'\tConverged after {itr} iterations')
        N0_final = N0
        beta_final = beta
        ne_final = ne_data
        vsc_final = vsc_tofit
        
        return ne_final, vsc_final, N0_final, beta_final


def fpeaks_scpot2ne(f_peaks, vsc):
    # Convert fpe [Hz] --> ne [cm^-3]
    ne_data = ne2fpe(f_peaks, inverse=True)

    # Resample
    global vsc_o
    vsc = pyrf.resample(vsc, ne_data.time, f_s=pyrf.calc_fs(ne_data))
    vsc_o = vsc
    # Ensure time series don't contain inf
    valid_idxs = np.isfinite(vsc.data) & np.isfinite(ne_data.data)
    vsc = vsc[valid_idxs]
    ne_data = ne_data[valid_idxs]
    
    # Calibrate density from fpe peaks against scpot
    ne, vsc, N0, beta = _calibrate_density(ne_data, vsc)
    return ne, vsc, N0, beta