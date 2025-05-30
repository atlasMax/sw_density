addpath('irfu-matlab')
mf = mfilename
tint=[irf_time([2006 01 01 1 1 0]) irf_time([2006 12 31 23 59 0])];
omni_data = irf_get_data_omni(tint, 'n', 'omni_min');
omni_t = omni_data(:,1);
omni_n = omni_data(:,2);
tss = TSeries(omni_t, omni_n);

% fig=figure;
% h = irf_plot({n_omni});
% drawnow; % <-- Force Matlab to actually update the figure window
% saveas(fig, 'matlab_omni.png')