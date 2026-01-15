import glob
import numpy as np
import logging
import os
import csv
import sys
import time

from pyrfu import mms, pyrf

import multiprocessing as mp

# local imports
from paths import MMS_REGIONFILES_DIR, COMPILED_TINTS_DIR


# Suppress INFO messages
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Suppress INFO messages

def _read_sw_tints_from_regionfiles(ic):
    pattern = f"{MMS_REGIONFILES_DIR}/mms{ic}_edp_sdp_regions_*"
    regionfiles = glob.glob(pattern)

    used_regionfiles_path = COMPILED_TINTS_DIR / "used_regionfiles.csv"
    # load used_regionfiles and check if any of regionfiles_names are already in it. only keep new files.
    if used_regionfiles_path.exists():
        used_regionfiles = np.genfromtxt(used_regionfiles_path, dtype=str, delimiter=",", ndmin=1).tolist()
        # Overwrite regionfiles
        regionfiles = [rf for rf in regionfiles if os.path.basename(rf) not in used_regionfiles]
    else:
        used_regionfiles = []
        

    sw_tints = []

    for filepath in regionfiles:
        region_data = np.genfromtxt(filepath, skip_header=True, dtype=str)
        times, flags = np.hsplit(region_data, 2)

        for i in range(len(flags) - 1):
            if flags[i] == '1':
                t0 = times[i][0][:-1]
                t1 = times[i + 1][0][:-1]
                tint = [t0, t1]
                # only add if not already in list
                if tint not in sw_tints:
                    sw_tints.append(tint)
        
        # Mark current regionfile as "used" so it will be skipped in future runs if new regionfiles are added.
        with open(used_regionfiles_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(filepath)])
            

    return sorted(sw_tints)


def _check_vsc(ts_mask, tint, ic):
    try:
        vsc_ = mms.get_data("v_edp_fast_l2", tint, ic)
    except FileNotFoundError:
        # print('NO EDP DATA FOUND FOR TINT', tint, '--- SKIPPING!')
        return None

    if vsc_.time.size > 0:
        vsc = vsc_.drop_duplicates(dim='time')
        vsc = pyrf.resample(vsc, ts_mask.time)#, method='linear')
        # Make exception for single NaN in vsc, can happen when time series goes over midnight?
        nans = np.argwhere(np.isnan(vsc.data))[:,0]
        ts_mask[nans] = 0
        if len(nans) < 10:
            for nan in nans:
                if vsc[nan-1] != None and vsc[nan+1] != None:
                    ts_mask[nan] = 1
        
    else:
        ts_mask[:] = 0
        
    return ts_mask

def _check_aspoc(ts_mask, tint, ic):
    try:
        aspoc = mms.get_data('ionc_aspoc_srvy_l2', tint, ic).drop_duplicates(dim='time')
    except FileNotFoundError:
        # print('NO ASPOC DATA FOR', tint, '--- SKIPPING')
        return None
    
    aspoc = pyrf.resample(aspoc, ts_mask.time, method='linear')

    # Keep only data where ASPOC is OFF (< 2), else NaN
    aspoc_off = aspoc.where(aspoc < 2, other=None)

    # ts_mask = pyrf.resample(ts_mask, aspoc_off.time)
    

    ts_mask[np.isnan(aspoc_off)] = 0
 

    
    return ts_mask

def _split_tint(data, tint):
    # Identify the time index where data jumps to/from NaN
    try:
        diff = data.differentiate(coord='time')
    except ValueError as e:
        return None
    
    diff_norm = diff / (np.max(diff) + 1e-9)
    toggle_idxs = np.where(np.abs(diff_norm) > 0.1)[0]
    
    # Make sure end point of tint is included if mask == 1 there
    if data.data[-1] == 1.0:
        toggle_idxs = np.append(toggle_idxs, -1)

    valid_tints = []
    start = 0
    for idx in toggle_idxs:
        start_time, end_time = data.time.data[start], data.time.data[idx]
        dt = np.timedelta64(end_time - start_time)
        if (np.all(data.data[start:idx] == 1)) & (dt > np.timedelta64(10,'s')):
            valid_tint = [str(start_time), str(end_time)]
            valid_tints.append(valid_tint)

        start = idx+1

    return valid_tints if len(valid_tints) > 0 else [tint]

def _check_swmode(tint, ic):
    ### Detect FPI operational mode
    # 0 - Solar wind mode OFF
    # 1 - Solar wind mode ON
    # 2 - FPI data not available
    try:
        dis = mms.get_data('defi_fpi_fast_l2', tint, ic)
        dis_energies = dis.energy.data
        # Solar wind tables from G-41 in
        # https://spdf.gsfc.nasa.gov/pub/data/mms/documents/MMS-CMAD.pdf
        minDISEnergy_sw, maxDISEnergy_sw = 210.0, 8700.0
        if np.min(dis_energies) == minDISEnergy_sw and np.max(dis_energies) == maxDISEnergy_sw:
            sw_mode = 1
        else:
            sw_mode = 0
            
    except (ValueError, FileNotFoundError, TypeError) as e:
        # print(e)
        sw_mode = 2

    return sw_mode


def _log_failed(failpath, tint, ic, message, index):
    with open(failpath, 'a', newline='') as f:
        writer = csv.writer(f)
        output = [tint[0], tint[1], ic, 2, message, index]
        writer.writerow(output)
    print(f'({index+1})', end=' ', flush=True)


def _preprocess_tint(tint, index, ic):
    """
    Process a single time interval.

    Returns
    -------
    dict with keys:
        status : "ok" | "fail"
        data   : list of rows (if ok)
        error  : str (if fail)
        index  : int
    """

    # Return fail if tint does not contain 2 elements
    if len(tint) != 2:
        return {
            "status": "fail",
            "error": "tint len != 2",
            "index": index,
            "tint": tint,
            "ic": ic,
        }

    t0, t1 = tint

    #  create mask time series at 1s resolution
    try:
        ts_axis = np.arange(
            np.datetime64(t0),
            np.datetime64(t1),
            np.timedelta64(1, "s"),
        ).astype("datetime64[ns]")

        ts_mask = pyrf.ts_scalar(ts_axis, np.ones_like(ts_axis))

    except Exception as e:
        return {
            "status": "fail",
            "error": f"mask creation",
            "index": index,
            "tint": tint,
            "ic": ic,
        }

    # --- Vsc availability ---
    ts_mask = _check_vsc(ts_mask, tint, ic)
    if ts_mask is None:
        return {
            "status": "fail",
            "error": "_check_vsc",
            "index": index,
            "tint": tint,
            "ic": ic,
        }

    # --- ASPOC status ---
    ts_mask = _check_aspoc(ts_mask, tint, ic)
    if ts_mask is None:
        return {
            "status": "fail",
            "error": "_check_aspoc",
            "index": index,
            "tint": tint,
            "ic": ic,
        }

    # --- split into valid sub-intervals ---
    valid_tints = _split_tint(ts_mask, tint)
    if not valid_tints:
        return {
            "status": "fail",
            "error": "_split_tint",
            "index": index,
            "tint": tint,
            "ic": ic,
        }

    # --- check SW mode and collect rows ---
    rows = []
    for vt in valid_tints:
        sw_mode = _check_swmode(vt, ic)
        rows.append([index, vt[0], vt[1], ic, sw_mode])

    return {
        "status": "ok",
        "data": rows,
    }



def _preprocess_tint_old(tint, index, filepath, failpath, ic, print_progress = False):
    if type(tint) != list:
        tint = tint.tolist()
    if len(tint) != 2:
        _log_failed(failpath, tint, ic, 'Tint len != 2', index)
        return  
    
    ### Create mask time series that is == 1 for intervals to keep, and == 0 for unwanted intervals
    ts_axis = np.arange(np.datetime64(tint[0]), np.datetime64(tint[-1]), np.timedelta64(1,'s')).astype('datetime64[ns]')
    ts_data = np.ones_like(ts_axis)
    ts_mask = pyrf.ts_scalar(ts_axis, ts_data)
            
    ### Check if spacecraft potential data from EDP is availale
    ts_mask_vsc = _check_vsc(ts_mask, tint, ic)
    if ts_mask_vsc is None:
        _log_failed(failpath, tint, ic, '_check_vsc failed', index)
        return
    
    ### ASPOC status
    ts_mask_vsc_aspoc = _check_aspoc(ts_mask_vsc, tint, ic)
    if ts_mask_vsc_aspoc is None:
        _log_failed(failpath, tint, ic, '_check_aspoc failed', index)
        return
    
    ### Split original tint into smaller ones where both ASPOC OFF and Vsc available
    valid_tints = _split_tint(ts_mask_vsc_aspoc, tint)
    if valid_tints is None:
        _log_failed(failpath, tint, ic, '_split_tint failed', index)
        return

    # For each valid tint in valid_tints, check SW mode and write to file
    for valid_tint in valid_tints:
        sw_mode = _check_swmode(valid_tint, ic)
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            output = [valid_tint[0], valid_tint[1], ic, sw_mode]
            writer.writerow(output)
    # Print progress
    if print_progress:
        update_percent = numtot // 21
        if np.mod(index, update_percent) == 0:
            # print(f'{(index/numtot)*100:.2f} %', end=' ', flush=True)
            print(f'|', end='', flush=True)
            
    return valid_tints

def _write_completed_tints(dir, results):
    compiled_path = dir / "compiled_sw_tints.csv"
    failed_path   = dir / "failed_sw_tints.csv"
    with open(compiled_path, "a", newline="") as f_ok, \
         open(failed_path, "a", newline="") as f_fail:

        writer_compiled = csv.writer(f_ok)
        writer_failed   = csv.writer(f_fail)
        
        # Only write header if the file is empty
        if f_ok.tell() == 0:
            writer_compiled.writerow(["index", "start", "end", "ic", "swmode"])
        if f_fail.tell() == 0:
            writer_failed.writerow(["index", "start", "end", "ic", "error"])

        for res in results:
            if res is None:
                continue

            if res["status"] == "ok":
                writer_compiled.writerows(res["data"])
            else:
                writer_failed.writerow([
                    res["tint"][0],
                    res["tint"][1],
                    res["ic"],
                    res["error"],
                    res["index"],
                ])


if __name__ == "__main__":
    COMPILED_TINTS_DIR.mkdir(parents=True, exist_ok=True)
    total_time_mins = 0.0
    results_all = []
    for ic in [1, 2, 3, 4]:
        sc_time_start = time.time() # start time for current spacecraft
        
        sw_tints_sorted = _read_sw_tints_from_regionfiles(ic)
        

        # Get number of workers from command-line argument
        num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4  # Default to 4
        
        # Run in parallel using multiprocessing 
        with mp.Pool(num_workers) as pool:
            results_ic = pool.starmap(_preprocess_tint, [(tint, i, ic) for i, tint in enumerate(sw_tints_sorted[:64])], chunksize=32)
        
        # Add results of current spacecraft to total result
        results_all.extend(results_ic)
        
        # Print duration for current spacecraft
        sc_time_end = time.time()   # end timer for current spacecraft
        sc_dur_mins = (sc_time_end - sc_time_start) / 60
        if sc_dur_mins < 1:
            print(f'\n> MMS{ic} finished in {sc_dur_mins*60:.2f} seconds')
            
        else:
            print(f'\n> MMS{ic} finished in {sc_dur_mins:.2f} minutes')
        total_time_mins += sc_dur_mins 
    
    # Write
    print(f'Writing to {COMPILED_TINTS_DIR}... ',end='')
    _write_completed_tints(COMPILED_TINTS_DIR, results_all)
    
    # Print total duration for all spacecraft
    if total_time_mins < 1:
        print(f'Finished in {total_time_mins*60:.2f} seconds')
        
    else:
        print(f'Finished in {total_time_mins:.2f} minutes')       
