import numpy as np
import os
import csv

def fit_beta_vsc(path):
    data = np.genfromtxt(path+'stats/fitting_stats.csv', delimiter=',', skip_header=1)
    beta = data[:, 2]
    ics = data[:,4]
    vsc_mean = data[:,6]
    
    
    filename = 'beta_vsc_cal.csv'
    fullpath = path+filename
    print('PRINTING FITTING PARAMS TO', fullpath)
    os.makedirs(path, exist_ok=True)
    if not os.path.exists(fullpath):
        output_header = ['slope', 'intecept']
        with open(fullpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(output_header)

    
    for ic in [1, 2, 3, 4]:
        ff = (beta >= 0) & (beta < 10) & (abs(vsc_mean) < 20) & (ics == ic)
        res = np.polyfit(vsc_mean[ff], beta[ff], deg=1)  
        with open(fullpath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(res)


fit_beta_vsc('output_data_new_2/cal/')