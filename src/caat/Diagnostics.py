import numpy as np
import matplotlib.pyplot as plt


class Diagnostic:
    """
    Class to implement a number of diagnostics
    to check quality of the GP fitting, including
    but not limited to plots and metrics
    """

    def identify_outlier_points(
            self,
            filt, 
            gp_times, 
            gp_prediction, 
            gp_std_deviation,
            phases,
            mags,
            errs, 
            nsigma=3
    ):
        """
        Check the goodness of fit by finding
        any points more than N sigma away from
        the GP fit for each filter
        """

        outliers = []
        for i, phase in enumerate(phases):
            gp_ind = np.argmin(abs(gp_times - phase))
            if abs(mags[i] - gp_prediction[gp_ind]) > nsigma * (errs[i] + gp_std_deviation[gp_ind]):
                outliers.append({'phase': phase, 'mag': mags[i], 'err': errs[i]})
        
        if len(outliers) > 0:
            print('WARNING: Outlier points identified for filter {}: {}'.format(filt, outliers))

    def check_late_time_slope(
            self,
            filt,
            gp_times,
            gp_prediction,
            phases,
            use_fluxes=False
    ):
        """
        Verify that the late-time slope of the
        GP fit for a given filter is negative,
        as expected for late-time SN light curves
        """

        last_phase = np.sort(phases)[-1]
        second_to_last_phase = np.sort(phases)[-2]
        last_phase_ind = np.argmin(abs(phases - last_phase))
        second_to_last_phase_ind = np.argmin(abs(phases - second_to_last_phase))
        ### Check that the GP fit between the last two data points is decreasing in brightness
        gp_slope = (gp_prediction[last_phase_ind] - gp_prediction[second_to_last_phase_ind]) / (last_phase - second_to_last_phase)
        if (use_fluxes and gp_slope > 0.0) or (not use_fluxes and gp_slope < 0.0):
            print('WARNING: Late-time slope of the GP is increasing for filter ', filt)

        ### Check that the GP fit extrapolation after the last data point is decreasing in brightness
        gp_extrapolation = (gp_prediction[-1] - gp_prediction[last_phase_ind]) / (gp_times[-1] - last_phase)
        if (use_fluxes and gp_extrapolation > 0.0) or (not use_fluxes and gp_extrapolation < 0.0):
            print('WARNING: GP extrapolation at late times is increasing for filter ', filt)

    def check_gradient_between_filters(
            self,
            filt_wls,
            phase_grid,
            wl_grid,
            gp_grid,
            phases_to_check
    ):
        """
        Check that the gradient between adjacent filters
        at representative time slices is smooth, i.e. free
        from second-order (or higher) bumps and wiggles
        """

        for phase in phases_to_check:
            phase_ind = np.argmin(abs(phase_grid - phase))

            sed = gp_grid[:, phase_ind]
            plt.plot(wl_grid, sed)
            plt.title(phase_grid[phase_ind])
            plt.show()

            for i, wl in enumerate(filt_wls):
                if i == len(filt_wls)-1:
                    # Reached the last filter
                    break
                blue_wl_ind = np.argmin(abs(wl_grid - wl))
                red_wl_ind = np.argmin(abs(wl_grid - filt_wls[i+1]))

                filter_gradient = gp_grid[blue_wl_ind:red_wl_ind, phase_ind]

                ### Check smoothness of filter gradient

