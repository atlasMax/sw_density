
from pyrfu import mms, pyrf
import numpy as np
import xarray as xr
from scipy import constants, signal
from scipy.ndimage import uniform_filter1d,gaussian_filter1d


def _weighted_median(values, weights):
    sorted_idx = np.argsort(values)
    values, weights = values[sorted_idx], weights[sorted_idx]
    cumsum = np.cumsum(weights)
    return values[np.searchsorted(cumsum, 0.5 * cumsum[-1])]


def epsd_peakfinder(epsd_data, power_threshold, frequency_lims=[5.88800e3, 8e4]):
    
    freq_axis = epsd_data.E_Freq.data
    time_axis = epsd_data.time.data
    
    # Preallocate results, one frequency value per time point (NaN by default)
    peaks_f = np.full(time_axis.size, np.nan)
    
    # Max/min frequency values and indices
    freq_min, freq_max = frequency_lims
    fmin_idx = np.argmin(np.abs(freq_axis - freq_min))
    fmax_idx = np.argmin(np.abs(freq_axis - freq_max))
    
    print(f'Frequency range between [{freq_min}, {freq_max}] at indices [{fmin_idx}, {fmax_idx}]')
    
    
    # Remove duplicates (Pyrfu bug when data is read across midnight)
    if len(np.unique(epsd_data.time.data)) != len(epsd_data.time.data):
        epsd_data = epsd_data.drop_duplicates(dim='time')
        time_axis = epsd_data.time.data
        peaks_f = np.full(time_axis.size, np.nan)
    
    # Filter spectral data
    epsd_cut = epsd_data - epsd_data.median(dim="time") # Subtract median to remove presistent noise
    epsd_cut = epsd_cut.where(epsd_cut >= power_threshold, 0) # Only keep points with sufficient amplitude
    epsd_cut = epsd_cut[:, fmin_idx:fmax_idx] # Keep data in freq. range

    # Normalize spectra: broadcast max normalization across time
    # max_vals = epsd_cut.max(dim="E_Freq")
    # epsd_slice = epsd_sub / max_vals

    # Normalize at every time
    spectra_norm = np.divide(epsd_cut, np.max(epsd_cut, axis=1)).data
    # Iterate over time points
    for tidx in range(time_axis.size):  
        # Slice the data at a single time point to obtain single spectrum
        spectrum = spectra_norm[tidx, :]
        spectrum = np.where(spectrum > 0, spectrum, 0)  # Zero out negative values
    
        # Find peaks
        peaks, _ = signal.find_peaks(spectrum, width=(1,3), threshold=0.8)
        if len(peaks) > 0:
            # Default to highest frequency peak if multiple found
            best_peak_idx = peaks[np.argmax(freq_axis[fmin_idx + peaks])]
            peaks_f[tidx] = freq_axis[fmin_idx + best_peak_idx]  # Store result
            
    if len(peaks_f) < 1:
        return epsd_cut, None, None
    
    # Convert to time series
    peaks_f_ts = pyrf.ts_scalar(time_axis, peaks_f)
    peaks_f_ts = peaks_f_ts[~np.isnan(peaks_f)]

    # Normalize weights to favor high frequencies
    weights = (peaks_f_ts.data)**2 / np.max(peaks_f_ts.data) 
    median_peak = _weighted_median(peaks_f_ts.data, weights)
        
    # Only keep peaks within certain range of weighted median
    peaks_f_ts_final = peaks_f_ts.where(abs(peaks_f_ts - median_peak) < 1*np.std(peaks_f_ts))
    
    
    return epsd_cut, peaks_f_ts_final, median_peak

