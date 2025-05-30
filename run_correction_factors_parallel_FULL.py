import numpy as np
import time
import multiprocessing as mp
import os
import sys
import requests
from compute_correction_factors_FULL import calc_fpi_corr_full, get_cdf_filenames

import logging
# Suppress INFO messages
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Suppress INFO messages from pyrfu



if __name__ == "__main__":
    ne_cal_path = sys.argv[1]
    num_workers = int(sys.argv[2])
    filenames = get_cdf_filenames(ne_cal_path, 'all')
    nfiles = len(filenames)
    time_start = time.time() 

    out_path = ne_cal_path+'stats/'
    os.makedirs(out_path, exist_ok=True)

    print(f'Calculating correction factors determination')
    print(f'> {nfiles} files with {num_workers} workers in parallel.')
    print(f'> Correction factors written to {out_path}')


    # print(f'Progress\tsc index\ttint\tstatus (Y: success, X: not converged, T: too short, N: too few peaks, D: data not found)')
    # Run in parallel using multiprocessing with progress tracking
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(calc_fpi_corr_full, [(filename, out_path, [i, nfiles]) for i, filename in enumerate(filenames[:])], chunksize=48)


    time_end = time.time()
    dur_mins = (time_end - time_start) / 60
    print(f'Finished in {dur_mins:.2f} minutes')


def notify_discord(message: str):
    url = "https://discord.com/api/webhooks/1360279999646531624/C_a4EY7G0PsYzFvqX5CdEETAFz3dREwVeRvoUWMGDTMB-_iXR_olnowzOM4EkeaYTBM4"
    data = {
        "content": message
    }
    try:
        requests.post(url, json=data)
    except Exception as e:
        print(f"Failed to send Discord message: {e}")
notify_discord(f'@everyone correction factor calculation finished in {dur_mins:.2f} minutes.')