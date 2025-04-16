import numpy as np
import xarray as xr
from pandas import Timedelta
import os
import csv
import random
from pyrfu import mms, pyrf


def get_cdf_filenames(folder: str, mode: str = "all", n: int = 5):
    # Ensure the folder exists
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' not found.")
    
    # Look at all subdirectories and collect .cdf file paths
    filenames = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".cdf"):
                filenames.append(os.path.join(root, f))

    if mode == "random":
        return random.sample(filenames, 1)
    elif mode == "all":
        return filenames
    else:
        raise ValueError("Invalid mode. Use 'all' or 'random'.")


def _tint2windows(time_axis, window_len = None):
    # Split time interval into windows of 'window_len' seconds
    start_time, end_time = time_axis[0], time_axis[-1]
    if window_len == None:
        return [str(start_time.data), str(end_time.data)]
    
    
    intervals = []
    t = start_time
    while t < end_time:
        t_next = t + Timedelta(seconds=window_len)
        if t_next > end_time:  # Extend last interval if needed
            t_next = end_time
        intervals.append([str(t.data), str(t_next.data)])
        t = t_next  # Move to next interval

    return intervals


def _read_data(cdf_filename):
        #units', 'beta', 'N0', 'ic', 'tint'
    ne_fit = xr.open_dataarray(cdf_filename)
    _, beta, N0, ic, tint, sw_mode = ne_fit.attrs.values()
    
    try:
        ne_fpi = mms.get_data('ne_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        return None
        # ne_fpi = xr.DataArray(np.full_like(ne_fit.time, np.nan), coords={'time': ne_fit.time})

    try:
        b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        return None
        # b_xyz = xr.DataArray(np.full((len(ne_fit.time), 3), np.nan), coords={'time': ne_fit.time, 'comp': ['x', 'y', 'z']})

    try:
        t_gse_e = mms.get_data("te_gse_fpi_fast_l2", tint, ic).drop_duplicates(dim='time')
        t_fac_e = mms.rotate_tensor(t_gse_e, "fac", b_xyz, "pp")
        t_e = pyrf.trace(t_fac_e) / 3
    except FileNotFoundError:
        return None
        # t_e = xr.DataArray(np.full_like(ne_fit.time, np.nan), coords={'time': ne_fit.time})

    # try:
    #     n_i = mms.get_data('ni_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
    # except FileNotFoundError:
    #     n_i = xr.DataArray(np.full_like(ne_fit.time, np.nan), coords={'time': ne_fit.time})

    try:
        vsc = mms.get_data('v_edp_fast_l2', tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        return None
        # vsc = xr.DataArray(np.full_like(ne_fit.time, np.nan), coords={'time': ne_fit.time})
    
    vsc = vsc.resample(time="20s").median()
    if len(ne_fpi) == 0 or len(ne_fit) == 0 or len(vsc) == 0:
        return None
    else:
        return ne_fit, ne_fpi, t_e, vsc, N0, beta, ic, tint, sw_mode


def _safe_mean(dataarray):
    return np.nan if dataarray.count() == 0 else dataarray.mean().data


def _write_to_csv(folderpath, output):
    
    filename = 'density_correction_stats.csv'
    
    # Ensure file exists and write header if not present
    os.makedirs(folderpath, exist_ok=True)
    if not os.path.exists(folderpath+filename):
        output_header = [
            'start', 'end', 'c0', 'vsc_mean', 't_e_mean', 'beta', 'N0', 'ic', 'sw_mode'
        ]
        with open(folderpath+filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(output_header)

    # Write results to file
    with open(folderpath+filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output)
        
def _print_output(string):
    print(string)
    
def _cdf_filename2datestr(cdf_filename):
    datestr = cdf_filename[:-4].split('_')[-1]
    datestr_fmt = datestr[::-1].replace('-', ':', 2)[::-1]
    return datestr_fmt


def calc_fpi_corr(cdf_filename, outpath=None, progress_track=None):
    logstatus = ''
    if progress_track is not None:
        index, tot = progress_track
        logstatus = f'[{index+1}/{tot}]\t\t{_cdf_filename2datestr(cdf_filename)}\t'
 
    data_result = _read_data(cdf_filename)
    if data_result is None:
        _print_output(logstatus+'X')
        return 
    else:
        ne_fit, ne_fpi, t_e, vsc, N0, beta, ic, tint, sw_mode = data_result
    # Downsample fpi data to obtained fit timeline (spacecraft spin res.)
    timeline_resamp = ne_fit.time
    ne_fpi_downsamp = pyrf.resample(ne_fpi, timeline_resamp)
    
    # split data into subtints
    window_len_s = 600
    window_tints = _tint2windows(timeline_resamp, window_len_s)
    # Store output of every segment
    output_list = []
    for window_tint in window_tints:
        ne_fit_clip = pyrf.time_clip(ne_fit, window_tint)
        ne_fpi_clip = pyrf.time_clip(ne_fpi_downsamp, window_tint)
        vsc_clip = pyrf.time_clip(vsc, window_tint)
        t_e_clip = pyrf.time_clip(t_e, window_tint)

        # Compute averages
        # ne_avg = ne_fit_clip.mean(dim='time').data
        # ne_fpi_avg = ne_fpi_clip.mean(dim='time').data
        ne_fit_mean = _safe_mean(ne_fit_clip)
        ne_fpi_mean = _safe_mean(ne_fpi_clip)
        vsc_mean = _safe_mean(vsc_clip)
        t_e_mean = _safe_mean(t_e_clip)
        
        # Compute correction factor: c0 = ne_avg / ne_fpi_avg
        if np.isnan(ne_fit_mean) or np.isnan(ne_fpi_mean):
            c0 = np.nan
            _print_output(logstatus+'C')
            return
        else:
            c0 = ne_fit_mean / ne_fpi_mean
        
        # prepare output
        start_str, stop_str = str(window_tint[0]), str(window_tint[-1])
        output = (
            start_str,
            stop_str,
            float(c0),
            float(vsc_mean),
            float(t_e_mean),
            float(beta),
            float(N0),
            int(ic),
            int(sw_mode)
        )
        # write to file if outpath provided, else just return output data (for single function calls)
        if outpath is not None:
            _write_to_csv(outpath, output)
            
        else:
            output_list.append(output)
        
    _print_output(logstatus+'Y')
    # Prevent returning if called from loop over all tints, written to files instead
    if outpath is None:
        # Convert to record array to specify dtypes of every element
        dtypes = [
            ('start_str', 'U30'), ('stop_str', 'U30'),
            ('c0', 'f8'), ('vsc_mean', 'f8'), ('t_e_mean', 'f8'),
            ('beta', 'f8'), ('N0', 'f8'),
            ('ic', 'i4'), ('sw_mode', 'i4')
        ]
        output_arr = np.array(output_list, dtype=dtypes)
        return output_arr
    
if __name__ == "__main__":
    cdf_filename='varg'
    calc_fpi_corr(cdf_filename, outpath='output_data/stats/')
