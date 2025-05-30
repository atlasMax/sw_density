from pyrfu import pyrf
from pyrfu import mms
import matplotlib.pyplot as plt
import numpy as np
from pyrfu.plot import plot_line, plot_spectr
import os, csv
import multiprocessing as mp
import logging
# Suppress INFO messages
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Suppress INFO messages from pyrfu





def refine_mask(ionfs_ts, min_duration=60*10):
    ionfs_mask = ionfs_ts.data
    diffs = np.diff(ionfs_mask)
    # Locate where mask goes from 0 -> 1 or 1 -> 0. Add 1 to get point after change
    start_idxs = np.where(diffs == -1)[0] + 1
    end_idxs = np.where(diffs == 1)[0] + 1

    # Handle edge cases to ensure start and end indices are same len
    if ionfs_mask[0] == 0:
        start_idxs = np.insert(start_idxs, 0, 0)
    if ionfs_mask[-1] == 0:
        end_idxs = np.append(end_idxs, len(ionfs_mask))

    time_axis = ionfs_ts.time.data
    ionfs_mask_clean = np.copy(ionfs_mask)
    min_duration = 60*10 # seconds
    for start, end in zip(start_idxs, end_idxs):
        dur = (time_axis[end - 1] - time_axis[start]).astype('timedelta64[s]')
        # If ionfs duration too short, set mask to zero (mark as SW)
        if dur < min_duration:
            ionfs_mask_clean[start:end] = 1
            
    ionfs_ts_clean = pyrf.ts_scalar(time_axis, ionfs_mask_clean)
    return ionfs_ts_clean


def _write_to_csv(folderpath, out):

    filename = 'ifs_separated_sw_tints.csv'

    # Ensure file exists and write header if not present
    os.makedirs(folderpath, exist_ok=True)
    if not os.path.exists(folderpath+filename):
        output_header = [
            'start', 'end', 'ic', 'sw_mode', 'flag (0: sw, 1: ifs)'
        ]
        with open(folderpath+filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(output_header)

    # Write results to file
    with open(folderpath+filename, "a", newline="") as f:
        writer = csv.writer(f)
        # writer.writerows(out)  
        writer.writerow(out)  
        

def separate_sw_ionfs(sw_tint):
    tint = list(sw_tint[:2])
    ic = int(sw_tint[2])
    sw_mode = int(sw_tint[3])
    try:
        defi = mms.get_data('defi_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
    except (FileNotFoundError, ValueError):
        return None
    dt = pyrf.calc_dt(defi)
    if dt < 4:
        return None
    DEFi_thresh = 10**(4.7)

    defi_top3 = defi[:, -3:]

    defi_mean = np.mean(defi_top3, axis=1)
    window_size = int(500 / dt)
    # defi_movmean = pyrf.movmean(defi_mean, window_size)
    defi_movmean = defi_mean.rolling(time=window_size, center=True).mean()
    # print(f'Moving mean with window size = {window_size}')
    ionfs_mask = np.where(defi_movmean > DEFi_thresh, 1, 0)
    ionfs_ts = pyrf.ts_scalar(defi_movmean.time.data, ionfs_mask)
        

    
    ionfs_ts_clean = refine_mask(ionfs_ts)
    ionfs_mask_clean = ionfs_ts_clean.data
    # Pick out times
    diffs = np.diff(ionfs_ts_clean)
    # Locate where mask goes from 0 -> 1 or 1 -> 0. Add 1 to get point after change
    start_idxs = np.where(diffs == -1)[0] + 1
    end_idxs = np.where(diffs == 1)[0] + 1

    # Handle edge cases to ensure start and end indices are same len
    if ionfs_mask_clean[0] == 0:
        start_idxs = np.insert(start_idxs, 0, 0)
    if ionfs_mask_clean[-1] == 0:
        end_idxs = np.append(end_idxs, len(ionfs_mask_clean))

    time_axis = ionfs_ts_clean.time.data
    out = []
    for start, end in zip(start_idxs, end_idxs):
        tint_sw = [str(time_axis[start]), str(time_axis[end-1])]
        # out.append([tint_sw[0], tint_sw[1], 0])
        out = [tint_sw[0], tint_sw[1], ic, sw_mode, 0]
        _write_to_csv(folderpath,out)
        
        tint_fs = [str(time_axis[end-1]), str(time_axis[start+1])]
        # out.append([tint_fs[0], tint_fs[1], 1])
        out = [tint_fs[0], tint_fs[1], ic, sw_mode, 1]
        
        _write_to_csv(folderpath, out)
    # return out

sw_tints = np.genfromtxt(f'sw_tints_new/compiled_sw_tints.csv', dtype=str, skip_header=1, delimiter=',')#, names=True)
ntot = len(sw_tints)
folderpath = 'sw_tints_new/'

with mp.Pool(processes=32) as pool:
    results = pool.map(separate_sw_ionfs, sw_tints[:], chunksize=128)
    # for i, res in enumerate(results):
    #     if res is not None:
    #         _write_to_csv(folderpath, res)
