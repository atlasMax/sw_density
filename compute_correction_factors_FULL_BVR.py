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
    
    # FPI densities
    try:
        ne_fpi = mms.get_data('ne_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        return None
    
    try:
        ni_fpi = mms.get_data('ni_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        return None
    
    # FPI bulk velocities
    try:
        ve_gse_fpi = mms.get_data('ve_gse_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
        vex_fpi = ve_gse_fpi.isel(comp=0)
        vey_fpi = ve_gse_fpi.isel(comp=1)
        vez_fpi = ve_gse_fpi.isel(comp=2)
    except FileNotFoundError:
        return None
    
    try:
        vi_gse_fpi = mms.get_data('vi_gse_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
        vix_fpi = vi_gse_fpi.isel(comp=0)
        viy_fpi = vi_gse_fpi.isel(comp=1)
        viz_fpi = vi_gse_fpi.isel(comp=2)
    except FileNotFoundError:
        return None
    
    
    try:
        b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, ic).drop_duplicates(dim='time')
        bx = b_xyz.sel(represent_vec_tot='x')
        by = b_xyz.sel(represent_vec_tot='y')
        bz = b_xyz.sel(represent_vec_tot='z')
    except FileNotFoundError:
        return None

    try:
        t_gse_e = mms.get_data("te_gse_fpi_fast_l2", tint, ic).drop_duplicates(dim='time')
        t_fac_e = mms.rotate_tensor(t_gse_e, "fac", b_xyz, "pp")
        t_e = pyrf.trace(t_fac_e) / 3
    except FileNotFoundError:
        return None

    try:
        t_gse_i = mms.get_data("ti_gse_fpi_fast_l2", tint, ic).drop_duplicates(dim='time')
        t_fac_i = mms.rotate_tensor(t_gse_i, "fac", b_xyz, "pp")
        t_i = pyrf.trace(t_fac_i) / 3
    except FileNotFoundError:
        return None


    try:
        r_xyz = mms.get_data('r_gse_mec_srvy_l2', tint, ic)
        rx = r_xyz.isel(comp=0)
        ry = r_xyz.isel(comp=1)
        rz = r_xyz.isel(comp=2)
    except FileNotFoundError:
        return None
    try:
        vsc = mms.get_data('v_edp_fast_l2', tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        return None
  
    vsc = vsc.resample(time="20s").median()
    if len(ne_fpi) == 0 or len(ne_fit) == 0 or len(vsc) == 0:
        return None
    else:
        return ne_fit, ne_fpi, ni_fpi, vex_fpi, vey_fpi, vez_fpi, vix_fpi, viy_fpi, viz_fpi, t_e, t_i, bx, by, bz, rx, ry, rz, vsc, N0, beta, ic, tint, sw_mode


def _safe_mean(dataarray):
    return np.nan if dataarray.count() == 0 else dataarray.mean().data


def _write_to_csv(folderpath, output_list):
    
    filename = 'density_correction_stats_full_BVR_0523.csv'
    
    # Ensure file exists and write header if not present
    os.makedirs(folderpath, exist_ok=True)
    if not os.path.exists(folderpath+filename):
        output_header = [
            'start', 'end', 'ce', 'ci', 'ne_fit', 'ne_fpi', 'ni_fpi', 'vex_fpi', 'vey_fpi', 'vez_fpi', 'vix_fpi', 'viy_fpi', 'viz_fpi', 't_e', 't_i', 'bx', 'by', 'bz', 'rx', 'ry', 'rz', 'vsc', 'N0', 'beta', 'ic', 'tint', 'sw_mode'
        ]
        with open(folderpath+filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(output_header)

    # Write results to file
    with open(folderpath+filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(output_list)
        
def _print_output(string):
    print(string)
    
def _cdf_filename2datestr(cdf_filename):
    datestr = cdf_filename[:-4].split('_')[-1]
    datestr_fmt = datestr[::-1].replace('-', ':', 2)[::-1]
    return datestr_fmt


def calc_fpi_corr_full(cdf_filename, outpath=None, progress_track=None):
    logstatus = ''
    if progress_track is not None:
        index, tot = progress_track
        logstatus = f'[{index+1}/{tot}]\t\t{_cdf_filename2datestr(cdf_filename)}\t'
 
    data_result = _read_data(cdf_filename)
    if data_result is None:
        _print_output(logstatus+'X')
        return 
    else:
        ne_fit, ne_fpi, ni_fpi, vex_fpi, vey_fpi, vez_fpi, vix_fpi, viy_fpi, viz_fpi, t_e, t_i, bx, by, bz, rx, ry, rz, vsc, N0, beta, ic, tint, sw_mode = data_result
    # Downsample fpi data to obtained fit timeline (spacecraft spin res.)
    timeline_resamp = ne_fit.time
    ne_fpi_downsamp = pyrf.resample(ne_fpi, timeline_resamp)
    ni_fpi_downsamp = pyrf.resample(ni_fpi, timeline_resamp)
    # split data into subtints
    window_len_s = 600 # 10 minutes
    window_tints = _tint2windows(timeline_resamp, window_len_s)
    # Store output of every segment
    output_list = []
    for window_tint in window_tints:
        # Clip intervals
        ne_fit_clip = pyrf.time_clip(ne_fit, window_tint)
        ne_fpi_clip = pyrf.time_clip(ne_fpi_downsamp, window_tint)
        ni_fpi_clip = pyrf.time_clip(ni_fpi_downsamp, window_tint)
        vex_fpi_clip = pyrf.time_clip(vex_fpi, window_tint)
        vey_fpi_clip = pyrf.time_clip(vey_fpi, window_tint)
        vez_fpi_clip = pyrf.time_clip(vez_fpi, window_tint)
        vix_fpi_clip = pyrf.time_clip(vix_fpi, window_tint)
        viy_fpi_clip = pyrf.time_clip(viy_fpi, window_tint)
        viz_fpi_clip = pyrf.time_clip(viz_fpi, window_tint)
        t_e_clip = pyrf.time_clip(t_e, window_tint)
        t_i_clip = pyrf.time_clip(t_i, window_tint)
        bx_clip = pyrf.time_clip(bx, window_tint)
        by_clip = pyrf.time_clip(by, window_tint)
        bz_clip = pyrf.time_clip(bz, window_tint)
        rx_clip = pyrf.time_clip(rx, window_tint)
        ry_clip = pyrf.time_clip(ry, window_tint)
        rz_clip = pyrf.time_clip(rz, window_tint)
        vsc_clip = pyrf.time_clip(vsc, window_tint)

        # Compute averages
        ne_fit_mean = _safe_mean(ne_fit_clip)
        ne_fpi_mean = _safe_mean(ne_fpi_clip)
        ni_fpi_mean = _safe_mean(ni_fpi_clip)
        vex_fpi_mean = _safe_mean(vex_fpi_clip)
        vey_fpi_mean = _safe_mean(vey_fpi_clip)
        vez_fpi_mean = _safe_mean(vez_fpi_clip)
        vix_fpi_mean = _safe_mean(vix_fpi_clip)
        viy_fpi_mean = _safe_mean(viy_fpi_clip)
        viz_fpi_mean = _safe_mean(viz_fpi_clip)
        t_e_mean = _safe_mean(t_e_clip) 
        t_i_mean = _safe_mean(t_i_clip) 
        bx_mean = _safe_mean(bx_clip)
        by_mean = _safe_mean(by_clip)
        bz_mean = _safe_mean(bz_clip)
        rx_mean = _safe_mean(rx_clip)
        ry_mean = _safe_mean(ry_clip)
        rz_mean = _safe_mean(rz_clip)
        vsc_mean = _safe_mean(vsc_clip)
        
        # Compute correction factors: c0 = ne_avg / n*_fpi_avg
        if np.isnan(ne_fit_mean) or np.isnan(ne_fpi_mean) or np.isnan(ni_fpi_mean):
            _print_output(logstatus+'C')
            return
        else:
            c0_e = ne_fit_mean / ne_fpi_mean
            c0_i = ne_fit_mean / ni_fpi_mean
            
        
        # prepare output
        start_str, stop_str = str(window_tint[0]), str(window_tint[-1])
        output = (
            start_str,
            stop_str,
            float(c0_e),
            float(c0_i),
            float(ne_fit_mean),
            float(ne_fpi_mean),
            float(ni_fpi_mean),
            float(vex_fpi_mean),
            float(vey_fpi_mean),
            float(vez_fpi_mean),
            float(vix_fpi_mean),
            float(viy_fpi_mean),
            float(viz_fpi_mean),
            float(t_e_mean),
            float(t_i_mean),
            float(bx_mean),
            float(by_mean),
            float(bz_mean),
            float(rx_mean),
            float(ry_mean),
            float(rz_mean),
            float(vsc_mean),
            float(beta),
            float(N0),
            int(ic),
            int(sw_mode)
        )
        
        # write to file if outpath provided, else just return output data (for single function calls)
        if outpath is not None:
            #_write_to_csv(outpath, output)
            output_list.append(output)
            
    if outpath is not None and output_list:
        _write_to_csv(outpath, output_list)
            
        
    _print_output(logstatus+'Y')
    # Prevent returning if called from loop over all tints, written to files instead
    if outpath is None:
        # Convert to record array to specify dtypes of every element
        dtypes = [
            ('start_str', 'U30'), ('stop_str', 'U30'),
            ('c0_e', 'f8'), ('c0_i', 'f8'),
            ('ne_fit_mean', 'f8'), ('ne_fpi_mean', 'f8'), ('ni_fpi_mean', 'f8'),
            ('vex_mean', 'f8'), ('vey_mean', 'f8'), ('vez_mean', 'f8'),
            ('vix_mean', 'f8'), ('viy_mean', 'f8'), ('viz_mean', 'f8'),
            ('t_e_mean', 'f8'), ('t_i_mean', 'f8'),
            ('bx_mean', 'f8'), ('by_mean', 'f8'), ('bz_mean', 'f8'),
            ('vsc_mean', 'f8'),
            ('beta', 'f8'), ('N0', 'f8'),
            ('ic', 'i4'), ('sw_mode', 'i4')
        ]
        output_arr = np.array(output_list, dtype=dtypes)
        return output_arr
    
if __name__ == "__main__":
    cdf_filename='varg'
    calc_fpi_corr_full(cdf_filename, outpath='output_data/stats/')
