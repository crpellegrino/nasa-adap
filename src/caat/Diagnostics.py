import numpy as np
import matplotlib.pyplot as plt
from caat.utils import query_svo_service, bin_spec
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
            nsigma=3,
            log_transform=None
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
                if log_transform is not None:
                    outliers.append(
                        {
                            'phase': round(phase, 2),
                            'mag': round(mags[i], 2),
                            'err': round(errs[i], 2)
                        }
                    )

                else:
                    outliers.append(
                        {
                            'phase': round(phase, 2),
                            'mag': round(mags[i], 2),
                            'err': round(errs[i], 2)
                        }
                    )
        
        if len(outliers) > 0:
            logger.warning('WARNING: Outlier points identified for filter {}: {}'.format(filt, outliers))

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
        if len(phases) < 2:
            return 
        last_phase = np.sort(phases)[-1]
        second_to_last_phase = np.sort(phases)[-2]
        last_phase_ind = np.argmin(abs(gp_times - last_phase))
        second_to_last_phase_ind = np.argmin(abs(gp_times - second_to_last_phase))
        if last_phase > 0 and second_to_last_phase > 0:
            ### Check that the GP fit between the last two data points is decreasing in brightness
            gp_slope = (gp_prediction[last_phase_ind] - gp_prediction[second_to_last_phase_ind]) / (last_phase - second_to_last_phase)
            if (use_fluxes and gp_slope > 0.0) or (not use_fluxes and gp_slope < 0.0):
                logger.warning(f'WARNING: Late-time slope of the GP is increasing for filter {filt}')

            ### Check that the GP fit extrapolation after the last data point is decreasing in brightness
            gp_extrapolation = (gp_prediction[-1] - gp_prediction[last_phase_ind]) / (gp_times[-1] - last_phase)
            if (use_fluxes and gp_extrapolation > 0.0) or (not use_fluxes and gp_extrapolation < 0.0):
                logger.warning(f'WARNING: GP extrapolation at late times is increasing for filter {filt}')

    def check_gradient_between_filters(
            self,
            filt_wls,
            phase_grid,
            wl_grid,
            gp_grid,
            std_grid,
            phases_to_check
    ):
        """
        Check that the gradient between adjacent filters
        at representative time slices is smooth, i.e. free
        from second-order (or higher) bumps and wiggles
        """

        if len(filt_wls) < 2:
            return 

        for phase in phases_to_check:
            phase_ind = np.argmin(abs(phase_grid - phase))

            sed = gp_grid[:, phase_ind]
            std = std_grid[:, phase_ind]
            plt.plot(wl_grid, sed, color='k')
            plt.fill_between(wl_grid, sed - std, sed + std)
            plt.title('SED at {} days'.format(round(phase_grid[phase_ind], 0)))
            plt.xlabel('Wavelength')
            plt.ylabel('Flux Relative to Peak')
            plt.show()

            for i, wl in enumerate(filt_wls):
                if i == len(filt_wls)-1:
                    # Reached the last filter
                    break
                blue_wl_ind = np.argmin(abs(wl_grid - wl))
                red_wl_ind = np.argmin(abs(wl_grid - filt_wls[i+1]))

                filter_gradient = gp_grid[blue_wl_ind:red_wl_ind, phase_ind]
                filter_gradient_std = std_grid[blue_wl_ind:red_wl_ind, phase_ind]

                ### Check smoothness of filter gradient

                # Fit the gradient as a 1d function
                fit = np.poly1d(np.polyfit(wl_grid[blue_wl_ind:red_wl_ind], filter_gradient, 1))
                bad_fit_inds = np.where((abs(filter_gradient - fit(wl_grid[blue_wl_ind:red_wl_ind])) > 3 * abs(filter_gradient_std)))[0]
                if len(bad_fit_inds) > 0:
                    logger.warning('WARNING: gradient between filters not smooth at wavelengths {}'.format(wl_grid[blue_wl_ind:red_wl_ind][bad_fit_inds]))

                    plt.plot(
                        wl_grid[blue_wl_ind:red_wl_ind], 
                        fit(wl_grid[blue_wl_ind:red_wl_ind]),
                        color='gray',
                        linestyle='--'
                    )
                    plt.plot(
                        wl_grid[blue_wl_ind:red_wl_ind],
                        filter_gradient,
                        color='k'
                    )
                    plt.fill_between(
                        wl_grid[blue_wl_ind:red_wl_ind],
                        filter_gradient - filter_gradient_std,
                        filter_gradient + filter_gradient_std
                    )
                    plt.xlabel('Wavelength')
                    plt.ylabel('Flux Relative to Peak')
                    plt.show()

    def check_uvm2_flux(
            self,
            phase_grid,
            wl_grid,
            gp_grid,
            std_grid,
            phases_to_check
    ):
        """
        """

        for phase in phases_to_check:
            phase_ind = np.argmin(abs(phase_grid - phase))

            sed = gp_grid[:, phase_ind]
            std = std_grid[:, phase_ind]

            trans_wl, trans_eff = query_svo_service('Swift', 'UVM2')
            trans_eff /= max(trans_eff)

            ### Bin the transmission curve to common resolution as SED
            binned_trans_wl, binned_trans_eff = bin_spec(trans_wl, trans_eff, wl_grid)
            
            ### Find indices of the transmission curve where 
            ### the transmission drops below 20%
            tail_inds = np.where((binned_trans_eff < 0.2))[0]

            ### Get indices of SED that fall within the binned transmission curve
            sed_inds = np.where((wl_grid >= binned_trans_wl[0]) & (wl_grid <= binned_trans_wl[-1]))[0]

            ### Calculate flux in the transmission curve
            ### Here we're shifting the flux upward by the minimum value
            ### in the entire filter to avoid issues with summing positive
            ### and negative normalized fluxes
            shifted_full_sed_flux = sed[sed_inds] - np.nanmin(sed[sed_inds])
            flux = np.nansum(shifted_full_sed_flux * 
                             binned_trans_eff[sed_inds]
            )

            ### Calculate flux in the tail
            shifted_tail_sed_flux = sed[tail_inds] - np.nanmin(sed[sed_inds])
            tail_flux = np.nansum(shifted_tail_sed_flux * 
                                  binned_trans_eff[tail_inds]
            )

            plt.plot(binned_trans_wl[sed_inds], shifted_full_sed_flux, color='black', label='GP SED')
            plt.scatter(binned_trans_wl[tail_inds], shifted_tail_sed_flux, marker='o', color='blue')
            plt.plot(binned_trans_wl, binned_trans_eff, color='red', label='UVM2 Transmission Efficiency')
            plt.xlabel('Wavelength')
            plt.ylabel('Relative Flux / Transmission Efficiency')
            plt.legend()
            plt.show()

            ### Compare the two values--is tail flux > 20% of the total?
            ratio = round(tail_flux / flux, 2)
            if ratio > 0.2:
                logger.warning(f'WARNING: {ratio} percent of UVM2 flux falls in red tail')
            else:
                logger.info(f'All good: Only {ratio} percent of UVM2 flux falls in red tail')
