import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import numpy as np
import matplotlib.gridspec as gridspec
from pyrfu.plot import plot_line, plot_spectr
from pyrfu import pyrf

from matplotlib.font_manager import FontProperties, fontManager

custom_font_path='./fonts/TIMES.TTF'
# font_prop = FontProperties(fname=)

fontManager.addfont(custom_font_path)
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['xtick.labelsize'] = 16
# plt.rcParams['ytick.labelsize'] = 16
# plt.rcParams['xtick.major.size'] = 5
# plt.rcParams['xtick.major.width'] = 2
# plt.rcParams['xtick.minor.size'] = 3
# plt.rcParams['xtick.minor.width'] = 1.5
# plt.rcParams['ytick.major.size'] = 5
# plt.rcParams['ytick.major.width'] = 2
# plt.rcParams['ytick.minor.size'] = 3
# plt.rcParams['ytick.minor.width'] = 1.5
# plt.rcParams['axes.titlesize'] = 18
# plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.linewidth'] = 1.6


plt.style.use('figstyle.mplstyle')

def resample_to_peaks(data, target):
    """Resample data to match the time series of target peaks."""
    try:
        return pyrf.resample(data, target)
    except RuntimeError as e:
        print("\tUSING NEAREST POINTS:", e)
        return data.sel(time=target.time, method="nearest")

def plot_ne_vsc_spectra(ne_peaks_final, vsc_fit2, outliers_idxs2, epsd_cut, power_min, f_peaks, f_peaks_final, f_peaks_final_movstd, ne_fit2,
                                    N0_fit2, beta_fit2, corr, vsc_tofit_var, std_thresh2, fpe_full_final,
                                    sigma_fpe, upper_best, lower_best, show_final_errorbars, scatter_errorbars, debug_window, tint, tint_clip_len):
                """Plots ne vs. Vsc scatter and electric field spectral density."""
                
                plt.close('all')
                fig = plt.figure(figsize=(16, 7))
                gs = gridspec.GridSpec(3, 2, hspace=0, left=0.06, top=0.9, bottom=0.1, width_ratios=[1, 3], height_ratios=[0.5, 0.35, 0.15])
                
                ax_fit = fig.add_subplot(gs[:2, 0])
                ax_fit_density = fig.add_subplot(gs[2, 0], sharex=ax_fit)
                ax = fig.add_subplot(gs[0, 1])
                ax2 = fig.add_subplot(gs[1:, 1], sharex=ax, sharey=ax)
                
                cmap = plt.cm.jet
                cmap.set_bad('white')
                spectr_options = dict(yscale="log", cscale="log", cmap=cmap, grid=True)
                
                # Split into outliers and non-outliers
                ne_out, ne_nout = ne_peaks_final[outliers_idxs2], ne_peaks_final[~outliers_idxs2]
                vsc_out, vsc_nout = vsc_fit2[outliers_idxs2], vsc_fit2[~outliers_idxs2]

                # Plot Vsc vs log(ne)
                log_ne_out, log_ne_nout = np.log(ne_out), np.log(ne_nout)
                
                # TESTING 2D HIST
                # ax_fit.hist2d(log_ne_nout, vsc_nout, bins = [np.unique(np.log(ne_peaks_final.data)), 100], cmap=cmap, cmin=15)
                
                ax_fit.scatter(log_ne_nout, vsc_nout, s=10, color='black', alpha=0.15, label='Data')
                # ax_fit.scatter(log_ne_out, vsc_out, s=10, color='red', marker='x', label=f'Outliers ({std_thresh2}'+r'$\sigma$)')
                

                ax_fit.plot(np.log(ne_fit2), vsc_fit2, color='magenta', linewidth=1.2, label='Fit')
                ax_fit.text(0.07, 0.94, 
                            rf'$N_\mathrm{{e}} = ${N0_fit2:.2f}$\times \exp(V_\mathrm{{SC}}/${beta_fit2:.2f})'+\
                            # '\n'+rf'$r_p$ = {corr:.2f}'+\
                            # '\n'+r'$\sigma^2$'+f' = {vsc_tofit_var:.3f}',
                            '',
                            transform = ax_fit.transAxes, fontsize=15, color='magenta')

                ax_fit.set_ylabel('-Spacecraft potential [V]', fontsize=15)
                ax_fit.legend(fontsize=12, loc='lower right', frameon=True)
                ax_fit.set_title(f"{len(ne_nout)} ({len(ne_peaks_final)}) data points")
                
                if scatter_errorbars:
                    # Plot median and std for each bar
                    N_vals =  np.unique(log_ne_nout.data)
                    grouped_vals = {val: vsc_nout[log_ne_nout.data == val] for val in N_vals}
                    
                    # Define threshold for minimum number of points required
                    min_count = 100
                    grouped_vals_median = [np.median(i) if len(i) > min_count else np.nan for i in grouped_vals.values()]
                    grouped_vals_std = [np.std(i) if len(i) > min_count else np.nan for i in grouped_vals.values()]
                    
                    ax_fit.errorbar(N_vals, grouped_vals_median, yerr=grouped_vals_std, fmt='o', color='lime')
                    
                

                
                # Histogram
                hist_data_all, hist_data_out = np.log(ne_peaks_final.data), log_ne_out
                bins = np.unique(np.log(ne_peaks_final.data))
                ax_fit_density.hist(hist_data_all, bins=bins, color='black', alpha=0.5, label='All data', align='mid')
                ax_fit_density.hist(hist_data_out, bins=bins, color='red', alpha=0.8, label=f'Outliers ({(len(hist_data_out)/len(hist_data_all)*100):.2f}%)')
                ax_fit_density.set_xlabel(r'$\log N_e$ [cm$^{-3}$]', fontsize=14)
                ax_fit_density.set_ylabel('Count', fontsize=15)
                # ax_fit_density.legend(fontsize=8, loc='best', frameon=True)              
         
                # Mask the spectral data to include only the visible frequency range
                ymin, ymax = min(8000, 0.5*np.min(fpe_full_final)), max(1.2e5, 2*np.max(fpe_full_final))
                visible_mask = (epsd_cut['E_Freq'] >= ymin) & (epsd_cut['E_Freq'] <= ymax)
                visible_data = epsd_cut.where(visible_mask, drop=True).data
                # Update color limits based on visible data
                if visible_data.size > 0:  # Ensure there's visible data
                    vmin, vmax = power_min, np.nanmax(visible_data)
  
                
                
                # Spectral plot
                # ax, cax_epsd = plot_spectr(ax, epsd_cut, clim=[vmin,vmax], **spectr_options)
                ax = plot_spectr(ax, epsd_cut, clim=[vmin,vmax], colorbar='none', **spectr_options)
                
                # cax_epsd.set_ylabel("Electric power spectral density" + "\n" + r"$[\mathrm{mV}^2~\mathrm{m}^{-2}~\mathrm{Hz}^{-1}$]")
                plot_line(ax, f_peaks_final, marker='.', markersize=2, linestyle='', color='black')#, label=rf'Final: $[N, \beta] = $ {np.round([N0_fit2, beta_fit2], 2)}', zorder=6)
                if debug_window:
                    ax.fill_between(x=upper_best.time, y1=upper_best, y2=lower_best, color='green', alpha=0.4, zorder=28)
                    plot_line(ax, f_peaks, marker='.', markersize=4, linestyle='', color='magenta', label='Initial')
                    
                # ax.legend(loc='upper left')
                ax.set_ylabel(r"$f~[\mathrm{Hz}]$")
                # ax.set_title(f'Median-subtracted EPSD\n{tint[0][:19]} - {tint[1][:19]} ({tint_clip_len:.1f} h)')
                
                # Second spectral plot
                # ax2, cax_epsd2 = plot_spectr(ax2, epsd_cut, clim=[vmin,vmax], **spectr_options)
                ax2 = plot_spectr(ax2, epsd_cut, clim=[vmin,vmax], colorbar='none', **spectr_options)
                
                # cax_epsd2.set_ylabel("Electric power spectral density" + "\n" + r"$[\mathrm{mV}^2~\mathrm{m}^{-2}~\mathrm{Hz}^{-1}$]")
                plot_line(ax2, fpe_full_final, color='crimson', alpha=0.9, linewidth=1.6, zorder=6, label=r'Fit plasma frequency $f_\mathrm{pe}^\mathrm{fit}$')
                
                if show_final_errorbars:
                    # f_peaks_final_movstd = f_peaks_final_movstd.drop_duplicates(dim='time')
                    # fpe_full_final_ds = resample_to_peaks(fpe_full_final, f_peaks_final_movstd)
                    # fpe_full_final_upper = fpe_full_final_ds + f_peaks_final_movstd
                    # fpe_full_final_lower = fpe_full_final_ds - f_peaks_final_movstd
                    # fpe_full_final_upper = f_peaks_final + f_peaks_final_movstd
                    # fpe_full_final_lower = f_peaks_final - f_peaks_final_movstd
                    # ax2.fill_between(x=fpe_full_final_upper.time, y1=fpe_full_final_upper, y2=fpe_full_final_lower,
                                    # color='navy', alpha=0.15, zorder=5, label=r'Error range $(3 \sigma)$')
                    fpe_full_final_upper = fpe_full_final + sigma_fpe
                    fpe_full_final_lower = fpe_full_final - sigma_fpe
                    ax2.fill_between(x=fpe_full_final.time, y1=fpe_full_final_upper, y2=fpe_full_final_lower,
                                    color='navy', alpha=0.15, zorder=5, label=r'Error range $(3 \sigma)$')                    
                ax2.set_ylabel(r"$f~[\mathrm{Hz}]$")
                ax.set_ylim(ymin, ymax)
                ax2.legend(loc='upper left', fontsize=14)

                # Create a colorbar axis spanning both spectral plots
                cax = fig.add_axes([0.91, 0.1, 0.01, 0.8])  # Adjust position/size if needed
                cax.grid(0)
                # Use the colorbar from ax (both plots share the same colormap and limits)
                try: #IDK?
                    cbar = fig.colorbar(ax.collections[0], cax=cax)
                    cbar.set_label("Electric power spectral density" + "\n" + r"$[\mathrm{mV}^2~\mathrm{m}^{-2}~\mathrm{Hz}^{-1}$]")

                except ValueError as e:
                    print(e)


                ax.yaxis.set_major_locator(LogLocator(base=10.0))

                ax.yaxis.set_major_formatter(LogFormatterMathtext())


                ax.set_xlim(xmin=epsd_cut.time.data[0])
                return fig