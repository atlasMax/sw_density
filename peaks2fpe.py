import os
import csv

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy import constants
from scipy import stats
from epsd_peakfinder_exp2 import epsd_peakfinder
from ne_from_scpot import fit_ne

from pyrfu import mms, pyrf
from pyrfu.plot import plot_line, plot_spectr, use_pyrfu_style, plot_surf
mms.db_init(default="local", local="../../../data/mms")


def get_swtints(filepath, tint_minlength_minutes):
    # Load SW tints (each row is two strings: start and end times)
    sw_tints = np.genfromtxt(filepath, dtype=str)

    # Minimum tint length as timedelta
    tint_minlength_td = np.timedelta64(tint_minlength_minutes, 'm')

    valid_swtints = []
    for tint in sw_tints:
        start, end = pyrf.iso86012datetime64(tint)
        duration = (end - start).astype('timedelta64[m]')
        tint_aslist = [str(start), str(end)]
        if duration >= tint_minlength_td and tint_aslist not in valid_swtints:
            valid_swtints.append(tint_aslist)  # Store as list of strings

    # Output results
    if valid_swtints:
        print(f"Found {len(valid_swtints)} valid solar wind tints of length > {tint_minlength_minutes // 60} hours.")
    else:
        print("No valid solar wind tints found.")
        
    return valid_swtints


def split_tint(time_series, tint_minlength_minutes):
    # Full time interval of input time series
    tint_orig = [str(time_series.time.data[0]), str(time_series.time.data[-1])]
    # print((time_series.time.data[-1] - time_series.time.data[0]) / np.timedelta64(1, 'h'))
    
    # Identify gaps in timeseries data and cut up original tint into smaller tints containing only data
    diffs=np.diff(time_series.time)
    timegap_thresh = np.timedelta64(30,'s')
    jump_idxs = np.where(diffs >= timegap_thresh)[0]

    if len(jump_idxs) > 0:
        jump_idxs = np.insert(jump_idxs, 0, 0)
        jump_idxs = np.append(jump_idxs, -1)

        clip_intervals = []
        # Do not include last 
        for i in range(len(jump_idxs) - 1):
            start_idx, stop_idx = jump_idxs[i], jump_idxs[i+1]
            if i > 0 : start_idx += 1
            start, stop = time_series.time.data[start_idx], time_series.time.data[stop_idx]
            tint_len_ns = stop - start
            tint_len_minutes = tint_len_ns / np.timedelta64(1, 'm')
            if tint_len_minutes >= tint_minlength_minutes:               
                clip_tint = [str(start), str(stop)]
                clip_intervals.append(clip_tint)
                print('Created new sub-tint of, ',tint_len_minutes, 'mins.')
            else:
                print('Sub-tint not long enough', (tint_len_minutes), 'mins.')
    else:
        print('Intervals of missing data not found')
        clip_intervals = [tint_orig]
    
    return clip_intervals


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


def define_frequency_floor(vsc_rs, f_peaks, fref, frange):
    """Determine the best frequency floor using correlation with SC potential."""
    nt = f_peaks.size
    corrs, scores = np.zeros(len(frange)), np.zeros(len(frange))
    f_idxs = np.zeros(len(frange), dtype='object')

    # print('Idx \t Min. freq \t Corr. coeff \t fraction \t score')
    for i, floor in enumerate(frange):
        # print(i, end='\t')
        idxs = []
        flim = fref + floor
        # print(f'{np.min(flim.data)}', end='\t')

        idxs = np.where(f_peaks.data >= flim.data)[0]
        frac = len(idxs)/nt
        # print(frac)
        if frac > 0.1:
            f_idxs[i] = idxs
            
            
            peaks_ne_lim = calculate_plasma_frequency(f_peaks, inverse=True)
            corr = stats.pearsonr(-vsc_rs[idxs], np.log(peaks_ne_lim[idxs]))[0]
            # print(f'{corr:.3f}',end='\t')
            # print(f'{len(idxs)}/{nt}', end='\t')
 
            score = 0.8*corr + 0.2*frac
            # print(f'{score:.4f}')
            corrs[i] = corr
            scores[i] = score
        else:
            print()
            break

    best_corr_idx = np.argmax(scores, axis=0)
    floor_idxs = f_idxs[best_corr_idx]
    print(f"Best frequency floor index: {best_corr_idx}, correlation: {corrs[best_corr_idx]:.3f}")
    return best_corr_idx, f_idxs[best_corr_idx], corrs[best_corr_idx]


def preprocess_peaks(f_peaks, floor_idxs, time_window='20s'):
    """Apply a frequency floor and remove NaNs using a running median."""
    f_peaks_abovefloor = f_peaks[floor_idxs]
    try:
        f_peaks_median = f_peaks_abovefloor.resample(time=time_window).median()
        f_peaks_median_nonan = f_peaks_median[~np.isnan(f_peaks_median)]

    except RuntimeError as e:
            print('MEDIAN FAILED, SET EQUAL TO PEAKS ABOVE FLOOR',e)
            f_peaks_median_nonan = f_peaks_abovefloor


    # Full data and mean
    return f_peaks_abovefloor, f_peaks_median_nonan





def improve_plasma_peaks(vsc, f_peaks, fref, frange):
    # Step 2: Apply frequency floor
    best_corr_idx, floor_idxs, _ = define_frequency_floor(vsc, f_peaks, fref, frange)
    f_peaks_abovefloor, f_peaks_median_nonan = preprocess_peaks(f_peaks, floor_idxs)
    return f_peaks_abovefloor, f_peaks_median_nonan, best_corr_idx


def ne_from_scpot(vsc_input, f_peaks_tofit, std_thresh):
    fit_result = fit_ne(vsc_input, f_peaks_tofit,  std_thresh=std_thresh)
    if fit_result is None:
        print('FITTING FAILED')
        return fit_result
    else:
        vsc_fit, ne_fit, outliers_idxs, N0_fit, beta_fit, fit_covar = fit_result
        print('Fitting successful with 1 sigma errors:', np.sqrt(np.diag(fit_covar)))
        # Use fitted parameters to calculate full time series of ne at spacecraft potential resolution.
        # ARE NE_FULL AND NE_FIT THE SAME?
        ne_full = N0_fit * np.exp(-vsc_input / beta_fit)
        
    
        return ne_full, vsc_fit, ne_fit, outliers_idxs, N0_fit, beta_fit, fit_covar


def write_to_file(filepath, output):


    # Ensure file exists and write header if not present
    if not os.path.exists(filepath):
        output_header = [
        "start", "end", "pearsonr", "num_tot", 
        "num_outliars", "std_thresh", "N_fit", "beta_fit", 
        "N_err", "beta_err", "vsc_mean", "vsc_median", "vsc_var", "ic"
        ]
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(output_header)

    # Write results to file
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output)
        # writer.writerow([
        #     tint_clip[0], tint_clip[1], corr, len(peaks_ne),
        #     len(outliers_idxs), std_thresh, N0_fit, beta_fit,
        #     np.sqrt(fit_covar[0][0]), np.sqrt(fit_covar[1][1])
        # ])


def main(ic, tint_minlength_minutes , plot_floor = False):
    # Get  of specified minimum length in minutes
    swtints = get_swtints(f'sw_tints/mms{ic}_sw_tints.txt', tint_minlength_minutes)
    num_swtints = len(swtints)
    continue_from = 0
    #tint = swtints[np.random.randint(0, len(swtints))]
    #tint = ['2019-02-17T07:01:02', '2019-02-19T02:00:55']
    for i, tint in enumerate(swtints[continue_from:]):
        print(f'OUT: {i+1+continue_from}/{len(swtints)}, {tint}')
        try:
            vsc = mms.get_data("v_edp_fast_l2", tint, ic)
        except FileNotFoundError:
            print('NO EDP DATA FOUND FOR TINT', tint, '--- SKIPPING!')
            continue
        vsc = vsc.drop_duplicates(dim='time')
        #vsc = vsc.resample(time="20s").median()
        if vsc.time.size > 0:
            clip_intervals = split_tint(vsc, tint_minlength_minutes)
        else:
            continue
        # Main loop per tint - do analysis separately on each tint (if multiple)
        for tint_clip in clip_intervals:
            # Length of tint in hours
            start_dt, end_dt = np.datetime64(tint_clip[0]), np.datetime64(tint_clip[-1])
            tint_clip_len = (end_dt - start_dt)/ np.timedelta64(1, 'h')
            if tint_clip_len <= tint_minlength_minutes/60:
                continue
            # Get new data for each clipped tint, skipping points where scpot does not have data
            epsd_ = mms.db_get_ts(f'mms{ic}_dsp_fast_l2_epsd',f'mms{ic}_dsp_epsd_omni', tint_clip)
            try:
                n_e_fpi_ = mms.get_data("ne_fpi_fast_l2", tint_clip, ic)
            except FileNotFoundError:
                 print('NO FPI DATA FOUND FOR TINT', tint_clip, '--- SKIPPING!')
                 continue
            # Drop duplicates
            epsd = epsd_.drop_duplicates(dim='time')
            n_e_fpi = n_e_fpi_.drop_duplicates(dim='time')

            # Clip full time series of scpot 
            vsc_clipped = pyrf.time_clip(vsc, tint_clip)
            # Re-sample clipped time series of scpot to spacecraft spin resolution (20 sec)
            vsc_clipped_rssr = vsc_clipped.resample(time="20s").median()


            # Step 1 - Find location of plasma frequency peaks in EPSD spectrum
            f_peaks = epsd_peakfinder(epsd, fmax=60000)
            # Define frequency floor range
            if not len(f_peaks) > 500:
                print('TOO FEW PEAKS FOUND, SKIPPING', tint_clip)
                continue 
            # Resample scpot to time series of frequency peaks for 1-to-1 data points 
            try:
                    vsc_rs = pyrf.resample(vsc_clipped_rssr, f_peaks)
            except RuntimeError as e:
                    print('USING NEREAST METHOD',e)
                    vsc_rs = vsc_clipped_rssr.sel(time=f_peaks.time, method='nearest')


            # Step 2 - Introduce a frequency floor with a shape following the expected plasma frequency variations


            f0 = np.min(f_peaks) / 100
            
            fmax, df =  30e3, 300
            frange = np.arange(f0, fmax, df)
            beta_guess = 0.5
            ne_shape = np.exp(-vsc_rs / beta_guess)
            f_shape = calculate_plasma_frequency(ne_shape)
            fref = f_shape / np.max(f_shape) * np.median(f_peaks)
            f_peaks_abovefloor, f_peaks_median_nonan, best_corr_idx = improve_plasma_peaks(vsc_rs, f_peaks, fref, frange) 

            
            # Step 3 - Fit the selected peaks to scpot
            f_peaks_tofit = f_peaks_abovefloor
            # f_peaks_tofit = f_peaks_median_nonan

            try:
                    vsc_tofit = pyrf.resample(vsc_clipped_rssr, f_peaks_tofit)
            except RuntimeError as e:
                    print("USING NEAREST METHOD", e)
                    vsc_tofit = vsc_clipped_rssr.sel(time=f_peaks_tofit.time, method="nearest")

            # Calculate Pearson correlation between SC potential and plasma density.
            peaks_ne = calculate_plasma_frequency(f_peaks_tofit, inverse=True)
            corr = stats.pearsonr(-vsc_tofit, np.log(peaks_ne))[0]
            # print(f"Pearson correlation coefficient: {corr:.3f}")

            # ne_full, vsc_fit, ne_fit, outliers_idxs, N0_fit, beta_fit, fit_covar
            std_thresh = 2
            fitting_results = ne_from_scpot(vsc_tofit, f_peaks_tofit, std_thresh=std_thresh)
            if fitting_results is None:
                continue
            ne_full, vsc_fit, ne_fit, outliers_idxs, N0_fit, beta_fit, fit_covar = fitting_results
            
            vsc_mean = np.mean(vsc_clipped_rssr.data)
            vsc_median = np.median(vsc_clipped_rssr.data)
            vsc_variance = np.var(vsc_clipped_rssr.data)
            
            # (#)/(!"#)!"#()!()"#()#")
            ne_full = N0_fit * np.exp(-vsc_clipped_rssr / beta_fit)


            # Convert from ne [cc] to fpe [Hz]
            fpe_fit = calculate_plasma_frequency(ne_full)

            # Step  - Plotting
            plt.close('all')
            fig = plt.figure(figsize=(15, 8))
            gs = gridspec.GridSpec(4, 2, hspace=0, 
                            width_ratios=[0.3, 0.7], height_ratios=[0.3, 0.3, 0.2, 0.1],
                            left=0.06, top=0.96)


            # Column 1: 
            ax_fit = fig.add_subplot(gs[:3, 0])
            ax_fit_density = fig.add_subplot(gs[3,0], sharex=ax_fit)


            # Column 2: 
            ax_spctr = fig.add_subplot(gs[:2, 1])
            ax_comp = fig.add_subplot(gs[2:, 1], sharex=ax_spctr)
            # ax_sw = fig.add_subplot(gs[2:, 1], sharex=ax_spctr)

            # Colors
            color_peaks = 'red'
            color_scfit = 'black'

            # SCATTER PLOT OF FIT

            # Split into outliers (out) and non-outliers (nout)
            ne_out = peaks_ne[outliers_idxs]
            ne_nout = peaks_ne[~outliers_idxs]
            vsc_out = vsc_fit[outliers_idxs]
            vsc_nout = vsc_fit[~outliers_idxs]

            # Plotting - Vsc vs log(ne)
            log_ne_out = np.log(ne_out)
            log_ne_nout = np.log(ne_nout)

            # Valid data points (non-outliers)
            ax_fit.scatter(log_ne_nout, vsc_nout, s=10, color='black', alpha=0.15, label='Data')

            # Mark median
            # v_psp_median = psp_nout.groupby(log_n_e_clean).median()
            # ax_fit.scatter(np.unique(log_ne_clean), v_psp_median, color='magenta', marker='o', s=7)

            # Outlier points
            ax_fit.scatter(log_ne_out, vsc_out, s=10, color='red', marker='x', label='Outliers '+rf'$({{{std_thresh}}}\sigma)$')
            N_err, beta_err = np.sqrt(np.diag(fit_covar))
            # Plot best fit line log(ne) vs. Vsc, and display values of fitted parameters
            ax_fit.plot(np.log(ne_fit), vsc_fit, color='magenta', linewidth=0.7, label='Fit')
            ax_fit.text(0.05, 0.85, rf'$N_\mathrm{{e}} = ${N0_fit:.2f}$\times \exp(V_\mathrm{{SC}}/${beta_fit:.2f})'+'\n'+rf'$r_p$ = {corr:.2f}'+'\n'+r'$\sigma^2$'+f' = {vsc_variance:.3f}', transform = ax_fit.transAxes, fontsize=11)

            # Labels and ticks
            ax_fit.tick_params(axis='x', which='both', top=False, right=True, bottom=False, labelbottom=False)
            # ax_fit.set_xticklabels([])
            ax_fit.set_ylabel('Spacecraft pot. [V]', fontsize=15)
            # ax_fit1.set_xlabel(r'$\log N_e$ (cm$^{-3})$', fontsize=15)
            ax_fit.legend(fontsize=10, loc='lower right', frameon=True)
            ax_fit.set_title(f"{len(ne_nout)} ({len(peaks_ne)}) data points")
            # ax_fit.set_xticks(np.unique(np.log(peaks_ne)))

            # SCATTER PLOT DENSITY 
            # Histogram data
            hist_data_all = np.log(peaks_ne.data)  # Full data (log scale)
            # hist_data_nout = np.log(ne_nout)  # Non-outlier data (log scale)
            hist_data_out = np.log(ne_out.data)  # Outlier data (log scale)


            # Define bins using the full dataset for consistency
            bins = np.arange(np.min(hist_data_all), np.max(hist_data_all), 0.13)
            bins = np.linspace(np.min(hist_data_all), np.max(hist_data_all), len(np.unique(peaks_ne.data)))
            bins = np.unique(np.log(peaks_ne.data))

            # Plot histograms
            ax_fit_density.hist(hist_data_all, bins=bins, color='black', alpha=0.5, label='All data', align='mid')
            # ax_fit_density.hist(hist_data_nout, bins=bins, color='grey', alpha=0.8, label='Non-outliers')
            ax_fit_density.hist(hist_data_out, bins=bins, color='red', alpha=0.8, label=f'Outliers ({(len(hist_data_out)/len(hist_data_all)*100):.2f}%)')



            # Add labels and legend
            ax_fit_density.set_xlabel(r'$\log N_e$ [cm$^{-3}]$', fontsize=15)
            ax_fit_density.set_ylabel('count', fontsize=15)
            ax_fit_density.tick_params(axis='both', top=True, right=True, bottom=True)
            ax_fit_density.legend(fontsize=8, loc='best', frameon=True)



            ### EPSD and peaks
            cmap = plt.cm.jet
            cmap.set_bad('white')
            spectr_options = dict(yscale="log", cscale="log", cmap=cmap, grid=True)
            legend_options = dict(ncol=4, handlelength=1.5, frameon=True)


            # Median subtracted spectrum
            epsd_sub = epsd - np.median(epsd, axis=0)
            epsd_sub_cut1 = epsd_sub[:, 6:]
            ax_epsd, cax_epsd = plot_spectr(ax_spctr, epsd_sub_cut1, clim='auto', zorder=15, **spectr_options)
            # ax_epsd1.set_ylim(ymin=2000)

            # # All under as well
            # plot_line(ax_epsd, f_peaks, marker='.', linestyle='', markerfacecolor='none',
            # markeredgecolor='lime', markeredgewidth=0.6, label='Peeks above floor', zorder=4, markersize=8)


            # Mark full peaks in spectrum
            plot_line(ax_epsd, f_peaks_abovefloor, marker='.', linestyle='', markerfacecolor='none',
            markeredgecolor='black', markeredgewidth=0.6, label='Peeks above floor', zorder=4, markersize=8)

            # Mark median peaks in spectrum
            plot_line(ax_epsd, f_peaks_median_nonan, marker='.', linestyle='', markerfacecolor='magenta',
            markeredgecolor='none', markeredgewidth=0.4, label='SC spinres median', zorder=8, markersize=5, alpha=0.85)

            if plot_floor:
                    # Floor
                    floor_ts = fref + frange[best_corr_idx]
                    plot_line(ax_epsd, floor_ts, linestyle='solid', color='navy', zorder=3, markersize=5)

                    # # Interpolate `peaks` to the closest frequency bin
                    # discrete_freqs = epsd.E_Freq.values  # Discrete frequency bins
                    # peaks_interpolated = discrete_freqs[np.abs(discrete_freqs[:, None] - floor_ts.values).argmin(axis=0)]

                    # # Add interpolated values back to a new DataArray
                    # peaks_to_bins = xr.DataArray(peaks_interpolated, coords=[floor_ts.time], dims=["time"]).resample(time='1800s').median()
                    # plot_line(ax_epsd, peaks_to_bins, linestyle='solid', color='navy', zorder=3, markersize=5)


            # plot_line(ax_epsd, f_peaks_filtered, marker='x', linestyle='', markerfacecolor='none',
            # markeredgecolor='black', markeredgewidth=0.4, label='Peeks above floor', zorder=3, markersize=6)


            # Plot ne-PSP-fit over spectrum
            # plot_line(ax_epsd, f_fit, color=color_scfit, linewidth=2, label="Initial SCpot Fit", zorder=5)
            ax_epsd.legend(loc='lower right', fontsize=11, **legend_options)
            cax_epsd.set_ylabel(r"$E^2$" + "\n" + r"$[\mathrm{mV}^2~\mathrm{m}^{-2}~\mathrm{Hz}^{-1}$]")
            ax_epsd.set_ylabel(r"$f~[\mathrm{Hz}]$")
            ax_epsd.set_xlim(xmin=epsd.time.data[0])

            ax_epsd.set_title(f'{tint_clip[0][:19]} - {tint_clip[1][:19]} ({tint_clip_len:.1f} h)')
            ax_epsd.tick_params(labelbottom=False, bottom=False)


            ### Plasma frequency comparison

            # ne-Vsc-fit of plasma freq. fpe
            plot_line(ax_comp, fpe_fit, color=color_scfit, linewidth=1, label="Initial SCpot Fit", linestyle="solid", zorder=4)


            # FPI plasma freq.
            f_pe_fpi = calculate_plasma_frequency(n_e_fpi)
            plot_line(ax_comp, f_pe_fpi, color='gold', linewidth=1, zorder=3, label=f'FPI data)')

            ax_comp.set_ylabel(r"$f~[\mathrm{Hz}]$")
            ax_comp.legend(loc='lower left',fontsize=8,**legend_options)



            floor_ts = fref + frange[best_corr_idx]
            ymin1 = np.min(floor_ts.data)
            # if ymin1 is not None:
            #         # ax_epsd.set_ylim(ymin1/5, 5*ymin1)
            #         ax_epsd.set_ylim(np.min(0.9*f_peaks_abovefloor), np.max(1.1*f_peaks_abovefloor))
            # else:
            #         ax_epsd.set_ylim(fref, 3*fref)

            ax_epsd.set_ylim(2e3, 1e5)
            # ax_epsd.yaxis.set_major_locator(
            #         mpl.ticker.LogLocator(base=10.0, numticks=4),
            # )
            # # Minor ticks (every multiple of the base, e.g., 2x10^x, 3x10^x, etc.)
            # ax_epsd.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
            # ax_epsd.yaxis.set_minor_formatter(mpl.ticker.LogFormatter())
            # ax_epsd.tick_params(axis='y', which='minor', left=True,labelleft=True)


            # Major ticks (10^x positions)
            ax_epsd.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=10))

            # Minor ticks (every multiple of the base, e.g., 2x10^x, 3x10^x, etc.)
            ax_epsd.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))

            # Use the same formatter for both major and minor ticks
            formatter = mpl.ticker.LogFormatterMathtext()
            ax_epsd.yaxis.set_major_formatter(formatter)
            ax_epsd.yaxis.set_minor_formatter(formatter)


            # Force y-ticks and labels for ax_epsd
            # y_ticks = ax_epsd.get_yticks()  # Get existing y-tick locations
            # ax_epsd.set_yticks(y_ticks)     # Set these ticks explicitly
            # ax_epsd.set_yticklabels([f"{tick:.0f}" for tick in y_ticks])  # Ensure labels are displayed for all ticks

            # ax_epsd.set_yticks([ymin1, 10000, 100000])

            ymin2 = 5000
            # ax_epsd2.set_yticks([10000, 100000])

            # ax_epsd2.set_yticks([10000, 100000])
            # Output figures and write to output file
            output_dir = f'out1701/{tint_minlength_minutes}/plots/'
            os.makedirs(output_dir, exist_ok=True)
            # plt.savefig(f'out/{tint_minlength_minutes}/plots/{tint[0][:13]}-MMS{ic}.png', dpi=300)
            plt.close()
            output_to_write = [tint_clip[0], tint_clip[1], corr, len(peaks_ne),
                    len(outliers_idxs), std_thresh, N0_fit, beta_fit,
                    np.sqrt(fit_covar[0][0]), np.sqrt(fit_covar[1][1]), vsc_mean, vsc_median, vsc_variance, ic]
            write_to_file(f'out1701/{tint_minlength_minutes}/stats-minlen{tint_minlength_minutes}min.csv',output_to_write)
            #plt.show()

# for ic in range(1, 5):
#     main(ic, 30, plot_floor=True)
ic = 1
main(ic, 30, plot_floor=True)