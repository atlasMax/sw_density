import csv
import os
import numpy as np
# Load all intervals with dtype=str to parse times manually
ionfs_data = np.genfromtxt('sw_tints_new/ifs_separated_sw_tints.csv', delimiter=',', dtype=str, skip_header=1)
ionfs_starts = np.array([np.datetime64(s) for s in ionfs_data[:, 0]])
ionfs_ends = np.array([np.datetime64(e) for e in ionfs_data[:, 1]])
ionfs_flags = ionfs_data[:, 4].astype(int) 

# Select only ambient SW intervals (flag == 0)
sw_starts = ionfs_starts[ionfs_flags == 0]
sw_ends = ionfs_ends[ionfs_flags == 0]

def _write_to_csv(folderpath, out):

    filename = 'only_sw_tints_BVR_0527.csv'

    # Ensure file exists and write header if not present
    os.makedirs(folderpath, exist_ok=True)
    if not os.path.exists(folderpath+filename):
        # output_header = [
        #     'start', 'end', 'c0_e', 'c0_i', 'ne_fit_mean', 'ne_fpi_mean', 'ni_fpi_mean', 'v_e_mean', 'v_i_mean', 't_e_mean', 't_i_mean', 'vsc_mean', 'beta', 'N0', 'ic', 'sw_mode'
        # ]
        # output_header = [
        #     'start', 'end', 'c0_e', 'c0_i', 'ne_fit_mean', 'ne_fpi_mean', 'ni_fpi_mean', 'vex_fpi', 'vey_fpi', 'vez_fpi', 'vix_fpi', 'viy_fpi', 'viz_fpi', 't_e', 't_i', 'bx', 'by', 'bz', 'vsc', 'N0', 'beta', 'ic', 'sw_mode'
        # ]
        output_header = [
            'start','end','ce','ci','ne_fit','ne_fpi','ni_fpi','vex_fpi','vey_fpi','vez_fpi','vix_fpi','viy_fpi','viz_fpi','t_e','t_i','bx','by','bz','rx','ry','rz','vsc','N0','beta','ic','sw_mode'
        ]
        with open(folderpath+filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(output_header)

    # Write results to file
    with open(folderpath+filename, "a", newline="") as f:
        writer = csv.writer(f)
        # writer.writerows(out)  
        writer.writerow(out)  


density_data = np.genfromtxt(f'output_data_new_2/stats/density_correction_stats_full_BVR_0523.csv', delimiter=',', names=True, skip_header=0, dtype=None)
folderpath = 'output_data_new_2/stats/'
for data_tint in density_data[:]:
    t_start = np.datetime64(data_tint[0])
    t_end = np.datetime64(data_tint[1])

    # t_end= np.datetime64("2020-02-08T13:09:17.411627000")
    # t_start   = np.datetime64("2020-02-08T12:48:03.900697000")
    # Check if it lies entirely within any ambient SW interval
    in_sw = np.any((t_start >= sw_starts) & (t_end <= sw_ends))
    if in_sw:
        _write_to_csv(folderpath, data_tint)
        
    
    
# For one analysis interval:
# t_start = np.datetime64("2018-01-06T12:44:58.763082500")
# t_end   = np.datetime64("2018-01-06T09:47:20.673910000")


