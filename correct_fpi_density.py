import numpy as np
from pyrfu import pyrf, mms
from pyrfu.plot import plot_line, plot_spectr
from ne2fpe import ne2fpe
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext



import xarray as xr
import os
import random
import numpy as np
from compute_correction_factors import calc_fpi_corr

def _get_cdf_filenames(folder: str, mode: str = "all", n: int = 5):
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


def _unpack_structured(arr):
    return tuple(arr[name] for name in arr.dtype.names)

def _get_cdf_filenames_from_tint(tint, ic):
    filenames = _get_cdf_filenames('output_data', 'all')
    # tint_start = tint[0]
    # year = tint_start[:4]
    # month = tint_start[5:7]
    # day = tint_start[8:10]
    # hh = tint_start[11:13]
    # mm = tint_start[14:16]
    # ss = tint_start[17:19]

    start, end = np.datetime64(tint[0]), np.datetime64(tint[1])
    valid_filenames = []
    for f in filenames:
        datestr = f[:-4].split('_')[-1]
        datestr_fmt = datestr[::-1].replace('-', ':', 2)[::-1]
        tf = np.datetime64(datestr_fmt)
        ic_idx = f.find('MMS')+3
        ic_file = int(f[ic_idx])
        if start <= tf <= end and ic_file == ic:
            valid_filenames.append(f)
            
    if len(f) == 0:
        return None
    else:
        print(f'Found {len(valid_filenames)} .cdf files matching tint!')
        return valid_filenames

def _correct_segments(valid_filenames):
    # valid_filenames = get_cdf_filenames_from_tint(tint, ic)
    
    segments = []
    
    for valid_filename in valid_filenames:
        output = calc_fpi_corr(valid_filename)
        start_strs, stop_strs, c0s, vsc_means, t_e_means, betas, N0s, ics, sw_modes = _unpack_structured(output)
        tint_segments = np.array([[s, e] for s,e in zip(start_strs, stop_strs)])
        segments.append([tint_segments, c0s])

    return segments


def correct_fpi_density(ne_fpi, tint, ic, plot_results = True):
    """Returns corrected FPI electron density.

    Args:
        ne_fpi (_type_): _description_
        tint (_type_): _description_
        ic (_type_): _description_
        plot_results (bool, optional): _description_. Defaults to True.

    Raises:
        IndexError: _description_

    Returns:
        _type_: _description_
        
        
            
    Example use:
    
    >>> from correct_fpi_density import correct_fpi_density
    >>> tint = ['2024-05-03T16:00:00.000000000', '2024-05-04T10:00:00.000000000']
    >>> ic = 1
    >>> ne_fpi = mms.get_data('ne_fpi_fast_l2', tint, ic).drop_duplicates(dim='time')
    >>> ne_fpi_corr, ne_fit, fig = correct_fpi_density(ne_fpi, tint, ic, plot_results=True)
    >>> fig.show()
    """    
    
    
    # segments_perfile = _correct_segments(tint, ic)
    valid_filenames = _get_cdf_filenames_from_tint(tint, ic)
    segments_perfile = _correct_segments(valid_filenames)
    
    ne_fit = xr.concat([xr.open_dataarray(fn) for fn in valid_filenames],
                dim='time',).sortby('time')
    
    corrected_segments_tot = []
    c0_segments_tot = []
    for row in segments_perfile:
        tint_segments = row[0]
        c_values = row[1]
        if len(tint_segments) != len(c_values):
            raise IndexError('Lists not equal size')
        
        corrected_segments = []
        c0_segments = []
        for tint_segment, c0 in zip(tint_segments, c_values):
            ne_fpi_clip = pyrf.time_clip(ne_fpi, tint_segment)
            ne_fpi_clip_corr = c0 * ne_fpi_clip
            corrected_segments.append(ne_fpi_clip_corr)
            c0_segments.append(c0)
            
        # Combine corrected density per file
        ne_fpi_corr_perfile = xr.concat(corrected_segments, dim='time').sortby('time')
        corrected_segments_tot.append(ne_fpi_corr_perfile)
        c0_segments_tot.append(c0_segments) 
        
    # Combine all 'ne_fpi_clip_corr' to one time series called 'ne_fpi_corr'
      
    avgc0 = np.mean([np.mean(i) for i in c0_segments_tot])
    ne_fpi_corr = xr.concat(corrected_segments_tot, dim='time').sortby('time')
    
    ne_fpi_corr.attrs['avg c0'] = avgc0
    ne_fpi_corr.attrs['c0s'] = c0_segments_tot
    
    if plot_results:
        epsd = mms.db_get_ts(f'mms{ic}_dsp_fast_l2_epsd',f'mms{ic}_dsp_epsd_omni', tint)

        # Convert spectral power from (V/m)^{2}/Hz -> (mV/m)^{2}/Hz
        epsd.data *= 1e6
        epsd.attrs['UNITS'] = '(mV/m)^{2}/Hz'


        plt.close('all')
        lw = 1.5
        fig, axs = plt.subplots(2, figsize=(13, 5), sharex=True)
        ax1, ax2 = axs
        plot_line(ax1, ne_fpi_corr, color='black', label=r'$n_\mathrm{e}^\mathrm{FPI-corr}$', linewidth=lw)
        plot_line(ax1, ne_fit, color='crimson', label=r'$n_\mathrm{e}^\mathrm{fit}$', linewidth=lw)
        plot_line(ax1, ne_fpi, color='gold', label=r'$n_\mathrm{e}^\mathrm{FPI}$', linewidth=lw)
        ax1.text(0.2, 0.8, r'$\langle C_\mathrm{FPI} \rangle$'+f'={ne_fpi_corr.attrs['avg c0']:.2f}', transform=ax1.transAxes)
        ax_s, ax_c = plot_spectr(ax2, epsd, clim='auto', yscale='log', cscale='log', cmap='jet')
        ax_c.grid(0)
        plot_line(ax2, ne2fpe(ne_fpi_corr), color='black', label=r'$f_\mathrm{pe}^\mathrm{FPI,corr}$', linewidth=lw)
        plot_line(ax2, ne2fpe(ne_fit), color='crimson', label=r'$f_\mathrm{pe}^\mathrm{fit}$', linewidth=lw)
        plot_line(ax2, ne2fpe(ne_fpi), color='gold', label=r'$f_\mathrm{pe}^\mathrm{FPI}$', linewidth=lw)



        ax1.legend(loc='upper right', fontsize=10, ncols=3)
        ax1.set_xlim(ne_fit.time.data[0], ne_fit.time.data[-1])



        ax2.legend(loc='upper right', fontsize=10, ncols=3)
        ax2.set_ylim(6000, 105000)
        ax2.yaxis.set_major_locator(LogLocator(base=10.0))
        ax2.yaxis.set_major_formatter(LogFormatterMathtext())
        return ne_fpi_corr, ne_fit, fig
    else:
        return ne_fpi_corr, ne_fit

    