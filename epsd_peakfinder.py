
from pyrfu import mms, pyrf
import numpy as np
import xarray as xr
from scipy import constants, signal
from scipy.ndimage import uniform_filter1d,gaussian_filter1d


def epsd_peakfinder(epsd_data, power_threshold, frequency_lims=[5.88800e3, 1e5]):
    
    freq_axis = epsd_data.E_Freq.data
    time_axis = epsd_data.time.data
    
    # Preallocate results, one frequency value per time point (NaN by default)
    peaks_f = np.full(time_axis.size, np.nan)
    
    # Max/min frequency values and indices
    freq_min, freq_max = frequency_lims
    fmin_idx = np.argmin(np.abs(freq_axis - freq_min))
    fmax_idx = np.argmin(np.abs(freq_axis - freq_max))
    
    print(f'Frequency range between [{freq_min}, {freq_max}] at indices [{fmin_idx}, {fmax_idx}]')
    
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
        
        # spectrum = gaussian_filter1d(spectrum, sigma=1)  # Adjust sigma to control smoothness
        # if tidx < 10:
        #     print(spectrum)
 
        # Detect peaks
        peaks, _ = signal.find_peaks(spectrum, 
                                     width=(1,3), 
                                     threshold=1.0
                                     )
        if peaks.size > 0:
            # Default to highest frequency peak
            # best_peak_idx = peaks[-1]
            best_peak_idx = peaks[np.argmax(freq_axis[fmin_idx + peaks])]
 
            # print(f'Found peaks {peaks} at tidx')
            peaks_f[tidx] = freq_axis[fmin_idx + best_peak_idx]  # Store result

    # Convert to time series
    peaks_f_ts = pyrf.ts_scalar(time_axis, peaks_f)
    peaks_f_ts_nonan = peaks_f_ts[~np.isnan(peaks_f)]
    return epsd_cut, peaks_f_ts_nonan

