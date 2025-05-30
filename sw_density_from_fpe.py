import numpy as np
from fpeaks_scpot2ne import fpeaks_scpot2ne   
from pyrfu import mms, pyrf
import sys
import os
import xarray as xr
import csv
def sw_density_from_fpe(tint, ic, sw_mode, plot_result = 0, progress_track=None, write_to=None):
    """_summary_

    Args:
        tint (list): _description_
        ic (int): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """    
    logstatus = ''
    if progress_track is not None:
        index, tot = progress_track
        logstatus = f'[{index+1}/{tot}]\t{ic}\t\t{tint}\t'
 
        

    # Check if tint longer than 10 mins
    tintlen_sec = (np.datetime64(tint[1]) - np.datetime64(tint[0])) / np.timedelta64(1, 's')
    if tintlen_sec // 60 < 10:
        # raise RuntimeWarning(f'Tint too short: {tintlen_sec // 60:.2f} min')
        _print_output(logstatus+'T')
        return
    
    try:
        # Get spacecraft potential, 20 sec medians (~every S/C spin)
        vsc_ = mms.get_data("v_edp_fast_l2", tint, ic)
        vsc = vsc_.drop_duplicates(dim='time') # Pyrfu bug when reading data spanning midnight (00:00)
        if len(vsc) < 10:
            _print_output(logstatus+'DV')
            return
        vsc = vsc.resample(time="20s").median()
    except FileNotFoundError as e:
        _print_output(logstatus+'DV')
        return

    # Get electric field power spectral density (EPSD) product from electric double probes
    try:
        epsd_data = mms.db_get_ts(f'mms{ic}_dsp_fast_l2_epsd',f'mms{ic}_dsp_epsd_omni', tint)
    except FileNotFoundError as e:
        _print_output(logstatus+'DE')
        return

    # Apply cut-offs in frequency and peak power. Filter data and get maximum spectral power amplitude at each time point
    power_threshold = 1e-6
    # Dynamic determination of fmin
    f_min = 6000 + (20000 - 6000) * np.exp(-np.median(vsc) / 5)
    f_peaks_ts, epsd_data_filt = _epsd_maxpeaks(epsd_data, power_threshold, freq_lims = [f_min, 8e4])

    if f_peaks_ts is None:
        _print_output(logstatus+'N')
        return
    else:
        # Obtain electron number density based on fpe peaks. Calibrate against scpot iteratively.
        ne, vsc_fitted, N0, beta = fpeaks_scpot2ne(f_peaks_ts, vsc)
        
        # If calibration fails, return ne as None
        if ne is None:
            _print_output(logstatus+'X')
            return None
        
        if write_to is not None:
            ne_fit = N0*np.exp(-vsc / beta)
            ne_fit_ts = xr.DataArray(
                data=ne_fit.data,  
                coords={"time": ne_fit.time.data},
                dims=["time"],
                name="n_e",
                attrs={'units': 'cm^-3', 
                    'beta': beta,
                    'N0' : N0,
                    'ic' : ic,
                    'tint' : tint,
                    'sw mode' : sw_mode
                }
            )
            if write_to is not None:
                _print_output(logstatus+'Y')
            # Write time series data to .cdf files
            _write_to_cdf(write_to, tint, ic, ne_fit_ts)

            # Write fitting statistic to statistics .csv file
            vsc_mean = np.nanmean(vsc_fitted.data)
            _write_to_csv(write_to, tint, beta, N0, ic, sw_mode, vsc_mean)
        if plot_result > 0:
            figure = _plot_results(ne, vsc_fitted, N0, beta, epsd_data_filt, vsc, tint, ic, plot_result)
            figure.savefig('tst.png', dpi=300)
        return ne, vsc_fitted, N0, beta, epsd_data_filt, vsc


def _print_output(string):
    print(string)

def _preprocess_epsd_data(epsd_data, power_threshold, freq_lims):
    # Returned cleaned and filtered spectral data
    
    # Convert spectral power from (V/m)^{2}/Hz -> (mV/m)^{2}/Hz
    epsd_data.data *= 1e6
    epsd_data.attrs['UNITS'] = '(mV/m)^{2}/Hz'

    freq_axis = epsd_data.E_Freq.data

    # Max/min frequency values and indices
    freq_min, freq_max = freq_lims
    fmin_idx = np.argmin(np.abs(freq_axis - freq_min))
    fmax_idx = np.argmin(np.abs(freq_axis - freq_max))

    # Remove duplicates (Pyrfu bug when data is read across midnight)
    if len(np.unique(epsd_data.time.data)) != len(epsd_data.time.data):
        epsd_data = epsd_data.drop_duplicates(dim='time')
        # time_axis = epsd_data.time.data
        # peaks_f = np.full(time_axis.size, np.nan)

    # Filter spectral data
    epsd_cut = epsd_data - epsd_data.median(dim="time") # Subtract median to remove presistent noise
    
    # Add back attrs
    epsd_cut.attrs = epsd_data.attrs
    
    epsd_cut = epsd_cut.where(epsd_cut >= power_threshold, 0) # Only keep points with sufficient amplitude
    epsd_cut = epsd_cut[:, fmin_idx:fmax_idx] # Keep data in freq. range

    return epsd_cut, freq_axis, fmin_idx


def _epsd_maxpeaks(epsd_data, power_threshold, freq_lims: list = [6e3, 8e4]):
    
    # Preprocess and filter peaks
    epsd_data_filt, freq_axis, fmin_idx = _preprocess_epsd_data(epsd_data, power_threshold, freq_lims)

    # Preallocate results, one frequency value per time point (NaN by default)
    time_axis = epsd_data_filt.time.data
    f_peaks = np.full(time_axis.size, np.nan)
    
    for tidx in range(time_axis.size):
        spectrum = epsd_data_filt[tidx, :].data
        spectrum = np.where(spectrum > 0, spectrum, 0)  # Zero out negative values
        
        peak_index = np.argmax(spectrum)
        if spectrum[peak_index] > 0:
            peak_f = freq_axis[fmin_idx + peak_index]
            f_peaks[tidx] = peak_f
    
    if len(f_peaks) > 100:
        # Convert to time series
        peaks_f_ts = pyrf.ts_scalar(time_axis, f_peaks)
        peaks_f_ts_nonan = peaks_f_ts[~np.isnan(f_peaks)]
        return peaks_f_ts_nonan, epsd_data_filt
    else:
        return None, epsd_data_filt


def _plot_results(ne, vsc_fitted, N0, beta, epsd_data_filt, vsc_data, tint, ic, mode=1):
    from matplotlib import gridspec, pyplot as plt
    from ne2fpe import ne2fpe
    from pyrfu.plot import plot_line, plot_spectr
    
    plt.style.use('../msc-project/figstyle.mplstyle')
    from matplotlib.font_manager import fontManager
    fontManager.addfont('../msc-project/fonts/TIMES.TTF')

    # mode = 1 : only scatter plot and fpe overlaid on epsd
    # mode = 2 : also show ne and FPI density 
    log_ne = np.log(ne)

    ne_fit = N0 * np.exp(-vsc_data / beta)

    ne_fpi_ = mms.get_data('ne_fpi_fast_l2', tint, ic)
    ne_fpi = ne_fpi_.drop_duplicates(dim='time')
    fpe_fit = ne2fpe(ne_fit)

    # Plotting
    plt.close('all')
    fig = plt.figure(figsize=(14, 6))
    if mode == 1:
        gs = gridspec.GridSpec(2, 2, width_ratios=[0.3, 0.7])#, hspace=0, left=0.06, top=0.9, bottom=0.1, width_ratios=[1, 3], height_ratios=[0.5, 0.35, 0.15])
        ax_fit = fig.add_subplot(gs[:, 0])
        ax_spectrum = fig.add_subplot(gs[:, 1])
    
    elif mode == 2:
        gs = gridspec.GridSpec(2, 2, width_ratios=[0.3, 0.7])#, hspace=0, left=0.06, top=0.9, bottom=0.1, width_ratios=[1, 3], height_ratios=[0.5, 0.35, 0.15])
        ax_fit = fig.add_subplot(gs[:, 0])
        ax_spectrum = fig.add_subplot(gs[0, 1])
        ax_density = fig.add_subplot(gs[1, 1])

    # Linear fit result and data points
    hist2d = ax_fit.hist2d(log_ne, -vsc_fitted, bins = [np.unique(log_ne.data), 30], cmap='jet', cmin=4)
    ax_fit.scatter(log_ne, -vsc_fitted, color='black', alpha=0.7)
    ax_fit.plot(np.log(ne_fit), -vsc_data, color='magenta', linewidth=1)

    # Spectral data and final fpe overlaid
    ax_s, ax_c = plot_spectr(ax_spectrum, epsd_data_filt, clim='auto', yscale='log', cscale='log', cmap='jet')
    plot_line(ax_s,fpe_fit, color='red', alpha=0.7, linewidth=1)
    if mode == 2:
        # Obtained density and FPI density
        plot_line(ax_density, ne_fit, color='red', alpha=0.7, linewidth=1)
        plot_line(ax_density, ne_fpi, color='gold', alpha=0.9, linewidth=1)

    fig.suptitle(f'{tint}, {ic}')
    return fig

def _write_to_cdf(filepath, tint, ic, ne_ts_data):

    # Store files in filepath/year/month/day with filenames as 'YYYY-MM-DDTHH-MM-SS.cdf'
    year = tint[0][:4]
    month = tint[0][5:7]
    folderpath = f'{filepath}/MMS{ic}/{year}/{month}/'
    
    # Create the directory if it doesn't exist
    os.makedirs(folderpath, exist_ok=True)
    filename = f'MMS{ic}_'+tint[0][:19].replace(':','-')+'.cdf'
    fullpath_str = folderpath+filename
    ne_ts_data.to_netcdf(fullpath_str)


def _write_to_csv(filepath, tint, beta, N0, ic, sw_mode, vsc_mean):
    folderpath = f'{filepath}stats/'
    filename = 'fitting_stats.csv'
    fullpath = folderpath+filename
    # Ensure file exists and write header if not present
    os.makedirs(folderpath, exist_ok=True)
    if not os.path.exists(fullpath):
        output_header = ['start', 'end', 'beta', 'N0', 'ic', 'sw mode', 'vscmed']
        with open(fullpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(output_header)

    # Write results to file
    output = [tint[0], tint[1], beta, N0, ic, sw_mode, vsc_mean]
    with open(fullpath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output)
        


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 5:
        print("Error: Incorrect number of arguments.")
        print("Usage:")
        print("  python sw_density_from_fpe.py <start_time> <end_time> <spacecraft id> <plotting mode (0: OFF, 1: fit and fpe, 2: fit, fpe and FPI density)>")
        print("Example:")
        print("  python sw_density_from_fpe.py 2024-05-03T17:00:00.000000000 2024-05-04T08:14:03.000000000 1 2")
        sys.exit(1)

    tint = [args[1], args[2]]
    ic = int(args[3])
    plot_mode = int(args[4])
    sw_density_from_fpe(tint, ic, plot_mode)