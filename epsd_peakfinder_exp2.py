
from pyrfu import mms, pyrf
import numpy as np
import xarray as xr
from scipy import constants, signal

def epsd_peakfinder(epsd_data, fmin=6200, fmax=None, reference_line=None):
    # Ensure fmin is compatible with the time coordinate
    if isinstance(fmin, xr.DataArray):
        assert 'time' in fmin.dims, "'fmin' as a time series must have a 'time' dimension."
        assert len(fmin.time) == len(epsd_data.time), "'fmin' time series must match the length of 'epsd_data' time."

    # Subtract median from entire dataset (faster)
    epsd_sub = epsd_data - epsd_data.median(dim="E_Freq")

    # Normalize spectra: broadcast max normalization across time
    max_vals = epsd_sub.max(dim="E_Freq")
    epsd_slice = epsd_sub / max_vals

    # Convert to NumPy for peak finding
    spectra = epsd_slice.data.T  # Shape: (freq, time)
    freq_axis = epsd_data.E_Freq.data

    # Preallocate results
    peaks_f = np.full(epsd_data.time.size, np.nan)
    # Vectorized peak finding for all time steps
    for tidx in range(spectra.shape[1]):  # Iterate over time (efficient)
        
        # Get time-specific fmin and fmax
        fmin_t = fmin if np.isscalar(fmin) else fmin[tidx].data
        fmax_t = fmax if np.isscalar(fmax) or fmax is None else fmax[tidx].data

        # Determine frequency slice indices
        fmin_idx = np.argmin(np.abs(freq_axis - fmin_t))
        fmax_idx = len(freq_axis) if fmax_t is None else np.argmin(np.abs(freq_axis - fmax_t))

        # Slice the spectrum
        slice_ = spectra[fmin_idx:fmax_idx, tidx]
        slice_ = np.where(slice_ > 0, slice_, 0)  # Zero out negative values

        # Detect peaks
        peaks, props = signal.find_peaks(slice_, width=(1, 3), prominence=(0.25, 1), distance=5)
        if peaks.size > 0:
            # Select best peak based on reference line or default to highest frequency
            if reference_line is not None:
                
                f_ref = reference_line[tidx].data
                distances = np.abs(freq_axis[fmin_idx + peaks] - f_ref)
                best_peak_idx = peaks[np.argmin(distances)]
            else:
                best_peak_idx = peaks[-1]  # Default to highest frequency peak

            peaks_f[tidx] = freq_axis[fmin_idx + best_peak_idx]  # Store result

    # Convert to time series
    peaks_f_ts = pyrf.ts_scalar(epsd_data.time.data, peaks_f)
    peaks_f_ts_nonan = peaks_f_ts[~np.isnan(peaks_f)]
    return peaks_f_ts_nonan

