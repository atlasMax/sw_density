import glob
import numpy as np

def _read_regionfiles_to_swtints():
    # Loop through all four MMS S/C
    for ic in range(1, 5):
        print(f'Region files for MMS{ic} at...',end='')
        
        # Get region file for specified S/C names by glob
        dir_regs = "mms-regionfiles/"
        file_pattern = dir_regs + f"mms{ic}_edp_sdp_regions_*"
        filenames = glob.glob(file_pattern)
        print(f'{file_pattern}')
        
        # Initialize an empty list to store the tint pairs
        sw_tints = []

        for filepath in filenames:
            print('\t'+filepath)
            # Load data with times and region flags
            region_data = np.genfromtxt(filepath, skip_header=True, dtype=str)
            times, region_flags = np.hsplit(region_data, 2)

            # Iterate through rows and check region flag for solar wind region (flag == '1')
            nrows = len(times)
            for i in range(1, nrows - 1):
                if region_flags[i] == '1':
                    # Create a tint spanning from times[i-1] to times[i+1]
                    sw_tint = [times[i][0][:-1], times[i+1][0][:-1]]
                    
                    # Prevent duplicates: Check if tint is unique. If not, do not add
                    if sw_tint not in sw_tints:
                        sw_tints.append(sw_tint)

        # Convert the list of tints to a NumPy array
        sw_tints = np.array(sw_tints)
        filename = f'sw_tints/mms{ic}_sw_tints.txt'
        print('\t'+20*'-')
        print('\t>'+f' Saving to {filename}...')
        print('\t'+20*'-'+'\n')
        np.savetxt(filename, sw_tints, fmt='%s', delimiter=' ')
        
        
def _preprocess_tints(flag_swmode = True, flag_foreshock = False):
    
