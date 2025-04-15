import xarray as xr
import numpy as np
from scipy import constants
from pyrfu import mms, pyrf

import os
import csv
import random
from typing import List
import multiprocessing as mp
from pandas import Timedelta
import sys
from tqdm import tqdm
from multiprocessing import Pool

mms.db_init(default="local", local="../../../data/mms")


def calculate_plasma_frequency(inp, inverse=False):
    """Convert number density in cc to plasma frequency or vice versa."""
    # Constants for plasma frequency calculation
    ep0 = constants.epsilon_0
    m_e = constants.electron_mass
    q_e = constants.elementary_charge
    if not inverse:  # ne (cc) -> ne (SI) -> f_pe
        return 1 / (2 * np.pi) * np.sqrt(inp * 1e6 * q_e**2 / (m_e * ep0))
    else:  # f_pe -> ne (SI -> ne (cc))
        w_pe_peaks = 2 * np.pi * inp
        n_e_peaks = w_pe_peaks**2 * ep0 * m_e / q_e**2
        return n_e_peaks * 1e-6  # Convert to cubic centimeters
    


def get_cdf_filenames(folder: str, mode: str = "all", n: int = 5) -> List[str]:
    """
    Retrieve filenames from a specified folder.
    
    Args:
        folder (str): Path to the folder containing the .csv files.
        mode (str): "all" to return all filenames, "random" to return n random filenames.
        n (int): Number of random filenames to return (if mode is "random").
        
    Returns:
        List[str]: List of filenames matching the pattern in the folder.
    """
    
    # Ensure the folder exists
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' not found.")
    
    # Get all CSV filenames matching the pattern
    filenames = [f for f in os.listdir(folder) if f.endswith(".cdf")]
    
    if mode == "random":
        return random.sample(filenames, min(n, len(filenames)))
    elif mode == "all":
        return filenames
    else:
        raise ValueError("Invalid mode. Use 'all' or 'random'.")
    
    
def tint2windows(time_axis, window_len = None):
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
    
    
    
def compare_fpi_swparams(tint, ic, window_len_s,ne, ne_fpi, vsc, t_e, n_i, b_xyz, beta_fit, N0_fit, sw_mode, num_tot, corr):
    # Compare obtained fit and ne_fpi

    # Downsample fpi data to obtained fit timeline (spacecraft spin res.)
    timeline_resamp = ne.time
    ne_fpi_downsamp = pyrf.resample(ne_fpi, timeline_resamp)
    
    
    # SPLIT TINT
    window_tints = tint2windows(timeline_resamp, window_len_s)
    for window_tint in window_tints:

        
        ne_clip = pyrf.time_clip(ne, window_tint)
        ne_fpi_clip = pyrf.time_clip(ne_fpi_downsamp, window_tint)
        vsc_clip = pyrf.time_clip(vsc, window_tint)
        t_e_clip = pyrf.time_clip(t_e, window_tint)
        n_i_clip = pyrf.time_clip(n_i, window_tint)
        b_xyz_clip = pyrf.time_clip(b_xyz, window_tint)

        
        ne_avg = ne_clip.mean(dim='time').data
        ne_fpi_avg = ne_fpi_clip.mean(dim='time').data

        # c0 = ne_avg / ne_fpi_avg
        c0 = np.nan if np.isnan(ne_avg) or np.isnan(ne_fpi_avg) else ne_avg / ne_fpi_avg


            
        # Prepare data to output
        print()
        start_str, stop_str = str(window_tint[0]), str(window_tint[-1])
        def safe_mean(dataarray):
            return np.nan if dataarray.count() == 0 else dataarray.mean().data

        def safe_std(dataarray):
            return np.nan if dataarray.count() == 0 else dataarray.std().data

        # Apply safe versions to avoid issues
        vsc_mean = safe_mean(vsc_clip)
        t_e_mean = safe_mean(t_e_clip)
        t_e_std = safe_std(t_e_clip)
        n_i_mean = safe_mean(n_i_clip)
        n_i_std = safe_std(n_i_clip)
        b_norm = safe_mean(pyrf.norm(b_xyz_clip))
        # vsc_mean = vsc_clip.mean().data
        # t_e_mean = t_e_clip.mean().data
        # t_e_std = t_e_clip.mean(skipna=True).data
        # n_i_mean = n_i_clip.mean().data
        # n_i_std = n_i_clip.std().data
        # b_norm = pyrf.norm(b_xyz_clip).mean().data
        # Ensure file exists and write header if not present
        filepath = f"MMS{ic}_density_calibration_stats_2402b_w={window_len_s}.csv"
        # filepath = f"MMS{ic}_test.csv"
        
        output = [
            start_str,
            stop_str,
            c0,
            vsc_mean,
            t_e_mean,
            t_e_std,
            n_i_mean,
            n_i_std,
            b_norm,
            beta_fit,
            N0_fit,
            sw_mode,
            num_tot,
            corr
            ]
        if not os.path.exists(filepath):
            output_header = [
            'start', 'stop', 'c0', 'vsc_mean', 't_e_mean', 't_e_std', 'n_i_mean', 'n_i_std', 'b_norm', 'beta_fit', 'N0_fit', 'sw_mode', 'num_tot', 'corr'
            ]
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(output_header)

        # Write results to file
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(output)
        print(f'Written to: {filepath}')


def process_file(args):
    """Process a single file and write offset coefficients."""
    filename, ic = args  # Unpack the tuple

    folder = f"out2402/MMS{ic}/60/"
    print(f"Processing {filename} for MMS{ic}...")

    ds_loaded = xr.open_dataset(folder + filename)
    ne = ds_loaded['ne_ts_final']
    beta_fit = ds_loaded.attrs['beta_fit']
    N0_fit = ds_loaded.attrs['N0_fit']
    num_tot = ds_loaded.attrs['num_tot']
    corr = ds_loaded.attrs['pearsonr']

    tint = [str(ne.time.data[0]), str(ne.time.data[-1])]

    # try:
    #     ne_fpi = mms.get_data(f'ne_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
    #     b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, ic).drop_duplicates(dim='time')
    #     t_gse_e = mms.get_data("te_gse_fpi_fast_l2", tint, ic).drop_duplicates(dim='time')
    #     n_i = mms.get_data('ni_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
    #     t_fac_e = mms.rotate_tensor(t_gse_e, "fac", b_xyz, "pp")
    #     t_e = pyrf.trace(t_fac_e) / 3
    #     vsc = mms.get_data('v_edp_fast_l2', tint, ic).drop_duplicates(dim='time')
    # except FileNotFoundError as e:
    #     print(f"Error: {e}")
    #     return  # Skip this file

    try:
        ne_fpi = mms.get_data('ne_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        ne_fpi = xr.DataArray(np.full_like(ne.time, np.nan), coords={'time': ne.time})

    try:
        b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        b_xyz = xr.DataArray(np.full((len(ne.time), 3), np.nan), coords={'time': ne.time, 'comp': ['x', 'y', 'z']})

    try:
        t_gse_e = mms.get_data("te_gse_fpi_fast_l2", tint, ic).drop_duplicates(dim='time')
        t_fac_e = mms.rotate_tensor(t_gse_e, "fac", b_xyz, "pp")
        t_e = pyrf.trace(t_fac_e) / 3
    except FileNotFoundError:
        t_e = xr.DataArray(np.full_like(ne.time, np.nan), coords={'time': ne.time})

    try:
        n_i = mms.get_data('ni_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        n_i = xr.DataArray(np.full_like(ne.time, np.nan), coords={'time': ne.time})

    try:
        vsc = mms.get_data('v_edp_fast_l2', tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        vsc = xr.DataArray(np.full_like(ne.time, np.nan), coords={'time': ne.time})


    # Check if FPI in solar wind mode
    try:
        dis = mms.get_data('defi_fpi_fast_l2', tint, ic)
        dis_energies = dis.energy.data
        minDISEnergy_sw, maxDISEnergy_sw = 210.0, 8700.0
        sw_mode = 1 if (np.min(dis_energies) == minDISEnergy_sw and np.max(dis_energies) == maxDISEnergy_sw) else 0

    except (ValueError, FileNotFoundError) as e:
        print(e)
        sw_mode = 2

    if len(ne_fpi) == 0 or len(ne) == 0 or len(vsc) == 0:
        print(f'MISSING {len(ne_fpi)} {len(ne)} {len(vsc)}')
        return  # Skip if data is missing

    vsc = vsc.resample(time="20s").median()
    window_len_s = 600
    compare_fpi_swparams(tint, ic, window_len_s, ne, ne_fpi, vsc, t_e, n_i, b_xyz, beta_fit, N0_fit, sw_mode, num_tot, corr)


def write_offset_coeffs(num_workers=None):
    """Parallelized function to process multiple .cdf files for all MMS spacecraft."""
    tasks = []
    
    for ic in range(1, 5):  # Loop over MMS1 to MMS4
        folder = f"out2402/MMS{ic}/60/"
        filenames = get_cdf_filenames(folder, mode="all")

        for filename in filenames:
            tasks.append((filename, ic))  # Store tuples of (filename, ic)

    if num_workers is None:
        num_workers = min(int(mp.cpu_count() * 0.75), len(tasks))

    with Pool(num_workers) as pool:
        for _ in tqdm(pool.imap(process_file, tasks), total=len(tasks)):
            pass  # tqdm updates automatically

    print("All files processed.")


   
if __name__ == "__main__":
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else None
    write_offset_coeffs(num_workers)

