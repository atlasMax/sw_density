import numpy as np
import time
import multiprocessing as mp
import os
import sys
import requests
from sw_density_from_fpe import sw_density_from_fpe

import logging
# Suppress INFO messages
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Suppress INFO messages from pyrfu


log_filename = 'sw_density_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)


# Read compiled solar wind intervals where:
#   ASPOC OFF
#   SC pot data available
start_times, end_times, sc_ids, sw_mode = np.genfromtxt('sw_tints/compiled_sw_tints4.csv', dtype=str, delimiter=',', skip_header=1).T


sw_tints = np.array([start_times, end_times]).T.tolist()
ntints = len(sw_tints)
time_start = time.time() 

out_path = 'output_data/'
os.makedirs(out_path, exist_ok=True)


num_workers = 32


print(f'Running solar wind electron density determination')
print(f'> {ntints} time intervals with {num_workers} workers in parallel.')
print(f'> Resulting electron density written to {out_path} as .cdf files')


print(f'Progress\tsc index\ttint\tstatus (Y: success, X: not converged, T: too short, N: too few peaks, D: data not found)')
# Run in parallel using multiprocessing with progress tracking
with mp.Pool(num_workers) as pool:
    results = pool.starmap(sw_density_from_fpe, [(tint, sc_ids[i], sw_mode[i],0, [i, ntints], out_path) for i, tint in enumerate(sw_tints[:])], chunksize=32)


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
notify_discord(f'@everyone sw density finished in {dur_mins:.2f} minutes.')