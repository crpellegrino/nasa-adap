import os
import json
import warnings
from typing import Union, Optional
from statistics import mean, stdev
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from extinction import fm07 as fm
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve
import astropy.units as u
from dustmaps.sfd import SFDQuery
from scipy.stats import iqr

# from .CAAT import CAAT
from .GP import GP#, Fitter
# from .Kernels import RBFKernel, WhiteKernel, MaternKernel
from .Plot import Plot
# from .SN import SN
# from .SNCollection import SNCollection, SNType
from caat.utils import colors

warnings.filterwarnings("ignore")


class GP3D(GP):
    """
    GP fit to all bands and epochs
    """

    @staticmethod
    def interpolate_grid(grid, interp_array, filter_window=171):
        """
        Function to remove NaNs by interpolating between actual measurements in a phase/wl grid
        Takes as input a grid to interpolate over, an array containing values of the grid
        along the interpolation dimension, as well as a filter window for smoothing
        using a Savitsky-Golay filter
        """
        for i, row in enumerate(grid):
            notnan_inds = np.where((~np.isnan(row)))[0]
            if len(notnan_inds) > 4:
                interp = interp1d(interp_array[notnan_inds], row[notnan_inds], "linear")

                interp_row = interp(interp_array[min(notnan_inds) : max(notnan_inds)])
                savgol_l = savgol_filter(interp_row, filter_window, 3, mode="mirror")
                row[min(notnan_inds) : max(notnan_inds)] = savgol_l
                grid[i] = row
        return grid

    def build_samples_3d(
        self,
        filt,
        phasemin,
        phasemax,
        log_transform=False,
        sn_set=None,
        use_fluxes=False,
    ):
        """
        Builds the data set from the SN collection for a given filter
        and returns, along with the phases, wls, and mags,
        the uncertainty in the measurements as the standard deviation
        of the photometry at each phase step
        """

        phases, mags, errs, wls = self.process_dataset_for_gp(
            filt,
            phasemin,
            phasemax,
            log_transform=log_transform,
            sn_set=sn_set,
            use_fluxes=use_fluxes,
        )

        min_phase, max_phase = sorted(phases)[0], sorted(phases)[-1]
        phase_grid = np.linspace(min_phase, max_phase, len(phases))
        phase_grid_space = (max_phase - min_phase) / len(phases)

        err_grid = np.ones(len(phase_grid))
        for mjd in phase_grid:
            ind = np.where((phases < mjd + phase_grid_space / 2) & (phases > mjd - phase_grid_space / 2))[0]
            mags_at_this_phase = mags[ind]
            if len(mags_at_this_phase) > 1:
                if use_fluxes:
                    std_mag = max(np.std(mags_at_this_phase), 0.01)
                else:
                    std_mag = max(np.std(mags_at_this_phase), 0.01)
            elif len(mags_at_this_phase) == 1:
                std_mag = errs[ind]
            else:
                if use_fluxes:
                    std_mag = 0.01
                else:
                    std_mag = 0.1

            err_grid[ind] = std_mag

        return phases, wls, mags, err_grid

    def process_dataset_for_gp_3d(
        self,
        filtlist,
        phasemin,
        phasemax,
        log_transform=False,
        fit_residuals=False,
        set_to_normalize=None,
        use_fluxes=False,
    ):
        """
        Processes the data set for the GP3D object's SN collection and
        (optionally) a SN set filter-by-filter and returns
        arrays of the SN collection's photometric details
        as well as the photometric details of the SN set to normalize to
        """

        all_phases, all_wls, all_mags, all_errs = [], [], [], []

        for filt in filtlist:
            phases, wl_grid, mags, err_grid = self.build_samples_3d(
                filt,
                phasemin,
                phasemax,
                log_transform=log_transform,
                use_fluxes=use_fluxes,
            )

            all_phases = np.concatenate((all_phases, phases.flatten()))
            all_wls = np.concatenate((all_wls, wl_grid.flatten()))
            all_mags = np.concatenate((all_mags, mags.flatten()))
            all_errs = np.concatenate((all_errs, err_grid.flatten()))

        if not fit_residuals:
            return all_phases, all_wls, all_mags, all_errs

        ### Create the template grid from the observations
        if set_to_normalize is not None:
            (
                all_template_phases,
                all_template_wls,
                all_template_mags,
                all_template_errs,
            ) = ([], [], [], [])
            for filt in filtlist:
                phases, wl_grid, mags, err_grid = self.build_samples_3d(
                    filt,
                    phasemin,
                    phasemax,
                    log_transform=log_transform,
                    sn_set=set_to_normalize,
                    use_fluxes=use_fluxes,
                )

                all_template_phases = np.concatenate((all_template_phases, phases.flatten()))
                all_template_wls = np.concatenate((all_template_wls, wl_grid.flatten()))
                all_template_mags = np.concatenate((all_template_mags, mags.flatten()))
                all_template_errs = np.concatenate((all_template_errs, err_grid.flatten()))
        else:
            all_template_phases = all_phases
            all_template_wls = all_wls
            all_template_mags = all_mags
            all_template_errs = all_errs

        return (
            all_phases,
            all_wls,
            all_mags,
            all_errs,
            all_template_phases,
            all_template_wls,
            all_template_mags,
            all_template_errs,
        )

    def construct_median_grid(
        self,
        phasemin,
        phasemax,
        filtlist,
        all_template_phases,
        all_template_wls,
        all_template_mags,
        all_template_errs,
        log_transform=False,
        plot=False,
        use_fluxes=False,
    ):
        """
        Takes as input the photometry from the sn set to normalize
        and constructs a 2D template grid consisting of the median photometry
        at each phase and wl step
        """

        if log_transform is not False:
            phase_grid_linear = np.arange(phasemin, phasemax, 1 / 24.0)  # Grid of phases to iterate over, by hour
            phase_grid = np.log(phase_grid_linear + log_transform)  # Grid of phases in log space

            wl_grid_linear = np.arange(
                min(self.wle[f] for f in filtlist) - 500,
                max(self.wle[f] for f in filtlist) + 500,
                99.5,
            )  # Grid of wavelengths to iterate over, by 100 A
            wl_grid = np.log10(wl_grid_linear)

        else:
            phase_grid = np.arange(phasemin, phasemax, 1 / 24.0)  # Grid of phases to iterate over, by hour
            wl_grid = np.arange(
                min(self.wle[f] for f in filtlist) - 500,
                max(self.wle[f] for f in filtlist) + 500,
                99.5,
            )  # Grid of wavelengths to iterate over, by 100 A

        mag_grid = np.empty((len(phase_grid), len(wl_grid)))
        mag_grid[:] = np.nan
        err_grid = np.copy(mag_grid)

        for i in range(len(phase_grid)):
            for j in range(len(wl_grid)):

                ### Get all data that falls within this phase + 5 days, and this wl +- 100 A
                if log_transform is not False:
                    inds = np.where(
                        (np.exp(all_template_phases) - np.exp(phase_grid[i]) <= 5.0)
                        & (np.exp(all_template_phases) - np.exp(phase_grid[i]) > 0.0)
                        & (abs(10**all_template_wls - 10 ** wl_grid[j]) <= 500)
                    )[0]
                else:
                    inds = np.where(
                        (all_template_phases - phase_grid[i] <= 5.0)
                        & (all_template_phases - phase_grid[i] > 0.0)
                        & (abs(all_template_wls - wl_grid[j]) <= 500)
                    )[0]

                if len(inds) > 0:

                    median_mag = np.median(all_template_mags[inds])
                    iqr = np.subtract(*np.percentile(all_template_mags[inds], [75, 25]))

                    mag_grid[i, j] = median_mag
                    err_grid[i, j] = iqr

        mag_grid = self.interpolate_grid(mag_grid.T, phase_grid)
        mag_grid = mag_grid.T
        mag_grid = self.interpolate_grid(mag_grid, wl_grid, filter_window=31)
        err_grid = self.interpolate_grid(err_grid.T, phase_grid)
        err_grid = err_grid.T
        err_grid = self.interpolate_grid(err_grid, wl_grid, filter_window=31)

        if log_transform is not False:
            X, Y = np.meshgrid(np.exp(phase_grid) - log_transform, 10**wl_grid)
        else:
            X, Y = np.meshgrid(phase_grid, wl_grid)

        Z = mag_grid.T

        if plot:
            Plot().plot_construct_grid(
                gp_class=self,
                X=X,
                Y=Y,
                Z=Z,
                phase_grid=phase_grid,
                mag_grid=mag_grid,
                wl_grid=wl_grid,
                err_grid=err_grid,
                log_transform=log_transform,
                filtlist=filtlist,
                grid_type="median",
                use_fluxes=use_fluxes,
            )

        return phase_grid, wl_grid, mag_grid, err_grid

    def construct_polynomial_grid(
        self,
        phasemin,
        phasemax,
        filtlist,
        all_template_phases,
        all_template_wls,
        all_template_mags,
        all_template_errs,
        log_transform=False,
        plot=False,
        use_fluxes=False,
    ):
        """
        Takes as input the photometry from the sn set to normalize
        and constructs a 2D template grid consisting of the polynomial fit
        to the SN set to normalize photometry at each phase and wl step
        """

        if log_transform is not False:
            phase_grid_linear = np.arange(phasemin, phasemax, 1 / 24.0)  # Grid of phases to iterate over, by hour
            phase_grid = np.log(phase_grid_linear + log_transform)  # Grid of phases in log space

            wl_grid_linear = np.arange(
                min(self.wle[f] for f in filtlist) - 500,
                max(self.wle[f] for f in filtlist) + 500,
                99.5,
            )  # Grid of wavelengths to iterate over, by 100 A
            wl_grid = np.log10(wl_grid_linear)

        else:
            phase_grid = np.arange(phasemin, phasemax, 1 / 24.0)  # Grid of phases to iterate over, by hour
            wl_grid = np.arange(
                min(self.wle[f] for f in filtlist) - 500,
                max(self.wle[f] for f in filtlist) + 500,
                99.5,
            )  # Grid of wavelengths to iterate over, by 100 A

        mag_grid = np.empty((len(phase_grid), len(wl_grid)))
        mag_grid[:] = np.nan
        err_grid = np.copy(mag_grid)

        for j in range(len(wl_grid)):

            ### Get all data that falls within this wl +- 100 A
            if log_transform is not False:
                inds = np.where((abs(10**all_template_wls - 10 ** wl_grid[j]) <= 100))[0]
                ### Add an array of fake measurements to anchor the ends of the fit
                anchor_phases = np.asarray(
                    [
                        np.log(phasemin + log_transform),
                        np.log(phasemin + 2.5 + log_transform),
                        np.log(phasemax + log_transform),
                    ]
                )
            else:
                inds = np.where((abs(all_template_wls - wl_grid[j]) <= 100))[0]
                anchor_phases = np.asarray([phasemin, phasemin + 2.5, phasemax])

            if len(inds) > 0:

                fit_coeffs = np.polyfit(
                    np.concatenate((all_template_phases[inds], anchor_phases)),
                    np.concatenate((all_template_mags[inds], np.asarray([-5.0, -4.0, -5.0]))),  # np.ones(len(anchor_phases)) * -5.0)),
                    3,
                    w=1
                    / (
                        np.sqrt(
                            (
                                np.concatenate(
                                    (
                                        all_template_errs[inds],
                                        np.ones(len(anchor_phases)) * 0.05,
                                    )
                                )
                            )
                            ** 2
                            + (np.ones(len(all_template_errs[inds]) + len(anchor_phases)) * 0.1) ** 2
                        )
                    ),
                )
                fit = np.poly1d(fit_coeffs)
                grid_mags = fit(phase_grid)

                mag_grid[:, j] = grid_mags
                err_grid[:, j] = np.ones(len(phase_grid)) * np.median(abs(all_template_mags[inds] - fit(all_template_phases[inds])))

        ### Interpolate over the wavelengths to get a complete 2D grid
        mag_grid = self.interpolate_grid(mag_grid, wl_grid, filter_window=31)
        err_grid = self.interpolate_grid(err_grid, wl_grid, filter_window=31)

        if log_transform is not False:
            X, Y = np.meshgrid(np.exp(phase_grid) - log_transform, 10**wl_grid)
        else:
            X, Y = np.meshgrid(phase_grid, wl_grid)

        Z = mag_grid.T

        if plot:
            Plot().plot_construct_grid(gp_class=self, X=X, Y=Y, Z=Z, grid_type="polynomial", use_fluxes=use_fluxes)

        return phase_grid, wl_grid, mag_grid, err_grid

    def subtract_data_from_grid(
        self,
        sn,
        phasemin,
        phasemax,
        filtlist,
        phase_grid,
        wl_grid,
        mag_grid,
        err_grid,
        log_transform=False,
        plot=False,
        use_fluxes=False,
    ):
        """
        Takes the (shifted) photometry from a given SN and subtracts from it
        the template grid constructed from either the median of all the normalization
        SN photometry, or the polynomial fit to all the normalization SN photometry
        Returns the phase and wavelength of each data point from the given SN, as well as
        the residuals in its magnitude (or flux) and its uncertainty
        """

        ### Subtract off templates for each SN LC
        residuals = {}
        for filt in filtlist:

            if len(sn.shifted_data) == 0:
                if sn.info.get("searched", False):
                    shifted_mjd, shifted_mag, err, nondets = [], [], [], []
                else:
                    sn.correct_for_galactic_extinction()
                    shifted_mjd, shifted_mag, err, nondets = sn.shift_to_max(filt, shift_fluxes=use_fluxes, try_other_filts=False)
            else:
                if filt in sn.shifted_data.keys():
                    if use_fluxes:
                        sn.correct_for_galactic_extinction()
                        sn.convert_to_fluxes(phasemin=phasemin, phasemax=phasemax)
                        shifted_mag = np.asarray([phot["flux"] for phot in sn.shifted_data[filt]])
                        err = np.asarray([phot["fluxerr"] for phot in sn.shifted_data[filt]])
                        nondets = np.asarray([phot.get("nondetection", False) for phot in sn.shifted_data[filt]])
                    else:
                        sn.correct_for_galactic_extinction()
                        shifted_mag = np.asarray([phot["mag"] for phot in sn.shifted_data[filt]])
                        err = np.asarray([phot["err"] for phot in sn.shifted_data[filt]])
                        nondets = np.asarray([phot.get("nondetection", False) for phot in sn.shifted_data[filt]])

                    shifted_mjd = np.asarray([phot["mjd"] for phot in sn.shifted_data[filt]])
                else:
                    shifted_mjd = []

            # There's a bug that I have to track down where sometimes filters without data
            # have a NaN as their redshift until data is read in, so the second clause
            # of this if statement will skip over those filters
            # (I don't know why this isn't caught above)
            if len(shifted_mjd) > 0 and not np.isnan(sn.info.get("z", 0)):

                if log_transform is not False:
                    shifted_mjd = sn.log_transform_time(shifted_mjd, phase_start=log_transform)
                    inds_to_fit = np.where((shifted_mjd > np.log(phasemin + log_transform)) & (shifted_mjd < np.log(phasemax + log_transform)))[0]
                    ### The wl corresponding to wl_ind is no more than the wl grid spacing away from the true wl being measured
                    wl_ind = np.argmin(abs(10**wl_grid - self.wle[filt] * (1 + sn.info.get("z", 0))))
                else:
                    inds_to_fit = np.where((shifted_mjd > phasemin) & (shifted_mjd < phasemax))[0]
                    wl_ind = np.argmin(abs(10**wl_grid - self.wle[filt] * (1 + sn.info.get("z", 0))))

                phases = shifted_mjd[inds_to_fit]
                mags = shifted_mag[inds_to_fit]
                errs = err[inds_to_fit]
                current_nondets = nondets[inds_to_fit]

                if plot and len(phases) > 0:
                    Plot().plot_subtract_data_from_grid(
                        gp_class=self,
                        sn_class=sn,
                        phase_grid=phase_grid,
                        mag_grid=mag_grid,
                        wl_ind=wl_ind,
                        filt=filt,
                        log_transform=log_transform,
                        use_fluxes=use_fluxes,
                    )

                for i, phase in enumerate(phases):
                    ### Get index of current phase in phase grid
                    if log_transform is not None:
                        ### The phase corresponding to phase_ind is no more than the phase grid spacing away from the true phase being measured
                        phase_ind = np.argmin(abs(np.exp(phase_grid) - np.exp(phase)))
                    else:
                        phase_ind = np.argmin(abs(phase_grid - phase))

                    if np.isnan(mag_grid[phase_ind, wl_ind]):
                        print(f"NaN Found: phase {np.exp(phase)}, wl {10**wl_grid[wl_ind]}")
                        continue

                    if log_transform is not None:
                        wl_residual = np.log10(self.wle[filt] * (1 + sn.info.get("z", 0)))
                    else:
                        wl_residual = self.wle[filt] * (1 + sn.info.get("z", 0))

                    residuals.setdefault(filt, []).extend(
                        [
                            {
                                "phase_residual": phase,
                                "wl_residual": wl_residual,
                                "mag_residual": mags[i] - mag_grid[phase_ind, wl_ind],
                                "err_residual": errs[i],
                                "mag": mags[i],
                                "nondetection": current_nondets[i],
                            }
                        ]
                    )

                    if plot:
                        plt.errorbar(
                            phase,
                            mags[i] - mag_grid[phase_ind, wl_ind],
                            yerr=np.sqrt(errs[i] ** 2 + err_grid[phase_ind, wl_ind] ** 2),
                            marker="o",
                            color="k",
                        )
                        plt.errorbar(
                            phase,
                            mags[i],
                            yerr=errs[i],
                            fmt="o",
                            color=colors.get(filt, "k"),
                        )

        return residuals

    def run_gp(
        self,
        filtlist,
        phasemin,
        phasemax,
        test_size=0.9,
        plot=False,
        log_transform=False,
        fit_residuals=False,
        set_to_normalize=None,
        subtract_median=False,
        subtract_polynomial=False,
        interactive=False,
        use_fluxes=False,
    ):
        """
        Function to run the Gaussian Process Regression
        ===============================================
        Takes as input:
        filtlist: A list of filters to fit the data of
        phasemin and phasemax: Endpoints of the phase range to fit
        test_size: The train/test split fraction (only if fit_residuals is False)
        plot: Optional flag to plot fits and intermediate figures
        log_transform: Flag to log-transform the phases (natural log) and wavelengths (log base 10)
        fit_residuals: Flag to fit the residuals of the measured SN photometry and the "template" grid
        set_to_normalize: Flag to provide a SNCollection object to construct a "template" grid
        median_combine_gps: Flag to median combine the GP fits to each individual SN to create a final median light curve template
        interactive: Flag to interactively choose to include each GP fit to the final median-combined template
        use_fluxes: Flag to fit in fluxes (or flux residuals), rather than magnitudes
        """
        if interactive:
            plot = True

        if fit_residuals:
            (
                all_phases,
                all_wls,
                all_mags,
                all_errs,
                all_template_phases,
                all_template_wls,
                all_template_mags,
                all_template_errs,
            ) = self.process_dataset_for_gp_3d(
                filtlist,
                phasemin,
                phasemax,
                log_transform=log_transform,
                fit_residuals=True,
                set_to_normalize=set_to_normalize,
                use_fluxes=use_fluxes,
            )
            kernel_params = []
            gaussian_processes = []

            if subtract_polynomial:
                phase_grid, wl_grid, mag_grid, err_grid = self.construct_polynomial_grid(
                    phasemin,
                    phasemax,
                    filtlist,
                    all_template_phases,
                    all_template_wls,
                    all_template_mags,
                    all_template_errs,
                    log_transform=log_transform,
                    plot=plot,
                    use_fluxes=use_fluxes,
                )
            elif subtract_median:
                phase_grid, wl_grid, mag_grid, err_grid = self.construct_median_grid(
                    phasemin,
                    phasemax,
                    filtlist,
                    all_template_phases,
                    all_template_wls,
                    all_template_mags,
                    all_template_errs,
                    log_transform=log_transform,
                    plot=plot,
                    use_fluxes=use_fluxes,
                )
            else:
                raise Exception("Must toggle either subtract_median or subtract_polynomial as True to run GP3D")

            for sn in self.collection.sne:
              
                residuals = self.subtract_data_from_grid(
                    sn,
                    phasemin,
                    phasemax,
                    filtlist,
                    phase_grid,
                    wl_grid,
                    mag_grid,
                    err_grid,
                    log_transform=log_transform,
                    plot=False,  # TODO: are we sure we want to hard-code False for grid subtraction?
                    use_fluxes=use_fluxes,
                )

                phase_residuals = np.asarray([p["phase_residual"] for phot_list in residuals.values() for p in phot_list])
                wl_residuals = np.asarray([p["wl_residual"] for phot_list in residuals.values() for p in phot_list])
                mag_residuals = np.asarray([p["mag_residual"] for phot_list in residuals.values() for p in phot_list])
                err_residuals = np.asarray([p["err_residual"] for phot_list in residuals.values() for p in phot_list])

                x = np.vstack((phase_residuals, wl_residuals)).T
                y = mag_residuals
                if len(y) > 1:
                    # We have enough points to fit
                    err = err_residuals

                    gaussian_process = GaussianProcessRegressor(kernel=self.kernel, alpha=err, n_restarts_optimizer=10)
                    gaussian_process.fit(x, y)

                    if plot:
                        fig, ax = Plot().create_empty_subplot()

                    filts_fitted = []

                    for filt in filtlist:

                        if log_transform is not False:
                            if len(wl_residuals[abs(10**wl_residuals - self.wle[filt] * (1 + sn.info.get("z", 0))) < 1]) == 0:
                                continue
                            test_times_linear = np.arange(phasemin, phasemax, 1.0 / 24)
                            test_times = np.log(test_times_linear + log_transform)
                            test_waves = np.ones(len(test_times)) * np.log10(self.wle[filt] * (1 + sn.info.get("z", 0)))
                            filts_fitted.append(filt)
                        else:
                            if len(wl_residuals[wl_residuals == self.wle[filt]] * (1 + sn.info.get("z", 0))) == 0:
                                continue
                            test_times = np.arange(phasemin, phasemax, 1.0 / 24)
                            test_waves = np.ones(len(test_times)) * self.wle[filt] * (1 + sn.info.get("z", 0))
                            filts_fitted.append(filt)

                        ### Trying to convert back to normalized magnitudes here
                        if log_transform is not None:
                            wl_ind = np.argmin(abs(10**wl_grid - self.wle[filt] * (1 + sn.info.get("z", 0))))
                        else:
                            wl_ind = np.argmin(abs(wl_grid - self.wle[filt] * (1 + sn.info.get("z", 0))))
                        template_mags = []
                        for i in range(len(phase_grid)):
                            template_mags.append(mag_grid[i, wl_ind])

                        template_mags = np.asarray(template_mags)

                        test_prediction, std_prediction = gaussian_process.predict(np.vstack((test_times, test_waves)).T, return_std=True)
                        if log_transform is not False:
                            test_times = np.exp(test_times) - log_transform

                        # Plot the SN photometry
                        shifted_mjd = np.asarray([phot["mjd"] for phot in sn.shifted_data[filt]])
                        if log_transform is not False:
                            shifted_mjd = sn.log_transform_time(shifted_mjd, phase_start=log_transform)

                        if log_transform is not False:
                            inds_to_fit = np.where(
                                (shifted_mjd > np.log(phasemin + log_transform)) & (shifted_mjd < np.log(phasemax + log_transform))
                            )[0]
                        else:
                            inds_to_fit = np.where((shifted_mjd > phasemin) & (shifted_mjd < phasemax))[0]

                        
                        if plot:
                            key_to_plot = "flux" if use_fluxes else "mag"
                            Plot().plot_run_gp_overlay(
                                fig=fig,
                                ax=ax,
                                gp_class=self,
                                sn_class=sn,
                                test_times=test_times,
                                test_prediction=test_prediction,
                                std_prediction=std_prediction,
                                template_mags=template_mags,
                                residuals=residuals,
                                phase_residuals=phase_residuals,
                                wl_residuals=wl_residuals,
                                err_residuals=err_residuals,
                                inds_to_fit=inds_to_fit,
                                log_transform=log_transform,
                                filt=filt,
                                use_fluxes=use_fluxes,
                                key_to_plot=key_to_plot,
                                phasemin=phasemin,
                                phasemax=phasemax,
                            )

                        if (subtract_median or subtract_polynomial) and interactive:
                            use_for_template = input("Use this fit to construct a template? y/n")

                    if subtract_median or subtract_polynomial:

                        if log_transform is not None:
                            waves_to_predict = np.unique(wl_residuals)
                            diffs = abs(
                                np.subtract.outer(10**wl_grid, 10**waves_to_predict)
                            )  # The difference between our measurement wavelengths and the wl grid

                        else:
                            waves_to_predict = np.unique(wl_residuals)
                            diffs = abs(np.subtract.outer(wl_grid, waves_to_predict))

                        phases_to_predict = np.unique(phase_residuals)

                        ### Compare the wavelengths of our measured filters to those in the wl grid
                        ### and fit for those grid wls that are within 500 A of one of our measurements
                        wl_inds_fitted = np.unique(np.where((diffs < 500.0))[0])
                        phase_inds_fitted = np.unique(np.where((phase_grid >= min(phases_to_predict)) & (phase_grid <= max(phases_to_predict)))[0])

                        x, y = np.meshgrid(phase_grid[phase_inds_fitted], wl_grid[wl_inds_fitted])
                        try:
                            test_prediction, std_prediction = gaussian_process.predict(np.vstack((x.ravel(), y.ravel())).T, return_std=True)
                        except:
                            print('WARNING:   BROKEN FIT FOR ', sn.name)
                            continue
                        test_prediction = np.asarray(test_prediction)

                        template_mags = []

                        for wl_ind in wl_inds_fitted:

                            for phase_ind in phase_inds_fitted:
                                template_mags.append(mag_grid[phase_ind, wl_ind])
                                ###NOTE: Some of these template mags are NaNs

                        template_mags = np.asarray(template_mags).reshape((len(x), -1))

                        ### Put the fitted wavelengths back in the right spot on the grid
                        ### and append to the gaussian processes array
                        test_prediction_reshaped = test_prediction.reshape((len(x), -1)) + template_mags

                        test_prediction_smoothed = np.empty(test_prediction_reshaped.shape)
                        for i, col in enumerate(test_prediction_reshaped.T):
                            window_size = max(int(round(len(col) / (2*len(filts_fitted)), 0)), 5) # Window size of approximately half a filter length scale
                            # Use astropy convolve function to handle NaNs
                            test_prediction_smoothed[:,i] = convolve(col, np.ones(window_size)/window_size, boundary='extend') # Boxcar smoothing

                        std_prediction_reshaped = std_prediction.reshape((len(x), -1)) + template_mags
                        std_prediction_smoothed = np.empty(std_prediction_reshaped.shape)
                        for i, col in enumerate(std_prediction_reshaped.T):
                            window_size = max(int(round(len(col) / (2*len(filts_fitted)), 0)), 5)
                            std_prediction_smoothed[:,i] = convolve(col, np.ones(window_size)/window_size, boundary='extend')

                        gp_grid = np.empty((len(wl_grid), len(phase_grid)))
                        gp_grid[:] = np.nan
                        for i, col in enumerate(test_prediction_reshaped[:,]):
                            current_wl_grid_ind = wl_inds_fitted[i]
                            for j in range(len(col)):
                                current_phase_grid_ind = phase_inds_fitted[j]
                                gp_grid[current_wl_grid_ind, current_phase_grid_ind] = col[j]

                        if plot:
                            Plot().plot_run_gp_surface(
                                gp_class=self,
                                x=np.exp(x)-log_transform,
                                y=10**(y),
                                test_prediction_reshaped=test_prediction_smoothed,#test_prediction_reshaped,
                                use_fluxes=use_fluxes,
                            )
                            from caat.Diagnostics import Diagnostic
                            d = Diagnostic()
                            d.check_gradient_between_filters(
                                [self.wle[f] for f in filts_fitted],
                                np.exp(phase_grid[phase_inds_fitted]) - log_transform,
                                10**(wl_grid[wl_inds_fitted]),
                                test_prediction_smoothed,
                                std_prediction_smoothed,
                                [-15.0, 0.0, 50.0]
                            )
                            if 'UVM2' in filts_fitted:
                                d.check_uvm2_flux(
                                    np.exp(phase_grid[phase_inds_fitted]) - log_transform,
                                    10**(wl_grid[wl_inds_fitted]),
                                    test_prediction_smoothed,
                                    std_prediction_smoothed,
                                    [-15.0, 0.0, 50.0]
                                )
                        if not plot:
                            use_for_template = "y"
                        elif not interactive:
                            use_for_template = "y"

                        if use_for_template == "y":
                            gaussian_processes.append(gp_grid)
                    kernel_params.append(gaussian_process.kernel_.theta)

            #if not subtract_median and not subtract_polynomial: 
            #    return gaussian_processes, phase_grid, wl_grid
            #return None, phase_residuals, kernel_params
            return gaussian_processes, phase_grid, kernel_params, wl_grid

        else:  # called if we are not fitting to residuals
            all_phases, all_wls, all_mags, all_errs = self.process_dataset_for_gp_3d(
                filtlist,
                phasemin,
                phasemax,
                log_transform=log_transform,
                use_fluxes=use_fluxes,
            )
            x = np.vstack((all_phases, all_wls)).T
            y = all_mags
            err = all_errs

            X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(x, y, err, test_size=test_size)

            ### Run the GP
            gaussian_process = GaussianProcessRegressor(kernel=self.kernel, alpha=Z_train, n_restarts_optimizer=10)
            gaussian_process.fit(X_train, Y_train)

            self.gaussian_process = gaussian_process

            return gaussian_process, X_test, None

    def predict_gp(
        self,
        filtlist,
        phasemin,
        phasemax,
        test_size=0.9,
        plot=False,
        log_transform=False,
        fit_residuals=False,
        set_to_normalize=False,
        subtract_median=False,
        subtract_polynomial=False,
        use_fluxes=False,
    ):
        """
        Function to predict light curve behavior using Gaussian Process Regression
        ===============================================
        Takes as input:
        filtlist: A list of filters to fit the data of
        phasemin and phasemax: Endpoints of the phase range to fit
        test_size: The train/test split fraction (only if fit_residuals is False)
        plot: Optional flag to plot fits and intermediate figures
        log_transform: Flag to log-transform the phases (natural log) and wavelengths (log base 10)
        fit_residuals: Flag to fit the residuals of the measured SN photometry and the "template" grid
        set_to_normalize: Flag to provide a SNCollection object to construct a "template" grid
        median_combine_gps: Flag to median combine the GP fits to each individual SN to create a final median light curve template
        use_fluxes: Flag to fit in fluxes (or flux residuals), rather than magnitudes
        """
        if not subtract_median and not subtract_polynomial:  # test_size is not None:
            ### Fitting sample of SNe altogether

            gaussian_process, X_test, kernel_params, _ = self.run_gp(
                filtlist,
                phasemin,
                phasemax,
                test_size=test_size,
                plot=plot,
                log_transform=log_transform,
                fit_residuals=fit_residuals,
                subtract_polynomial=subtract_polynomial,
                subtract_median=subtract_median,
                use_fluxes=use_fluxes,
            )

            if plot:
                fig, ax = Plot().create_empty_subplot()

            if test_size is not None:
                for filt in filtlist:

                    test_times = np.linspace(min(X_test[:, 0]), max(X_test[:, 0]), 60)
                    if log_transform is not None:
                        test_waves = np.ones(len(test_times)) * np.log10(self.wle[filt])

                    else:
                        test_waves = np.ones(len(test_times)) * self.wle[filt]

                    test_prediction, std_prediction = gaussian_process.predict(np.vstack((test_times, test_waves)).T, return_std=True)

                    if plot:
                        Plot().plot_predict_gp(
                            fig=fig,
                            ax=ax,
                            gp_class=self,
                            test_prediction=test_prediction,
                            std_prediction=std_prediction,
                            log_transform=log_transform,
                            filt=filt,
                            use_fluxes=use_fluxes,
                        )

        else:
            ### We're fitting each SN individually and then median combining the full 2D GP
            gaussian_processes, phase_grid, _, wl_grid = self.run_gp(
                filtlist,
                phasemin,
                phasemax,
                plot=plot,
                log_transform=log_transform,
                fit_residuals=fit_residuals,
                set_to_normalize=set_to_normalize,
                subtract_median=subtract_median,
                subtract_polynomial=subtract_polynomial,
                use_fluxes=use_fluxes,
            )

            median_gp = np.nanmedian(np.dstack(gaussian_processes), -1)

            if log_transform is not False:
                X, Y = np.meshgrid(np.exp(phase_grid) - log_transform, 10**wl_grid)
            else:
                X, Y = np.meshgrid(phase_grid, wl_grid)

            # for i, col in enumerate(median_gp.T):
            #     median_gp[:,i] = savgol_filter(col, 51, 3)
            median_gp = self.interpolate_grid(median_gp.T, wl_grid, filter_window=31)
            median_gp = median_gp.T


            iqr_grid = iqr(np.dstack(gaussian_processes), axis=-1, nan_policy='omit')
            # for i, col in enumerate(iqr_grid.T):
            #     iqr_grid[:,i] = savgol_filter(col, 51, 3)
            iqr_grid = self.interpolate_grid(iqr_grid.T, wl_grid, filter_window=31)
            iqr_grid = iqr_grid.T
            
            Z = median_gp

            Plot().plot_construct_grid(gp_class=self, X=X, Y=Y, Z=Z, Z_lower=Z-iqr_grid, Z_upper=Z+iqr_grid, grid_type="final", use_fluxes=use_fluxes)
        # elif subtract_polynomial:      
        #     gaussian_processes, phase_grid, wl_grid = self.run_gp(filtlist, phasemin, 
        #                                                           phasemax, plot=plot, 
        #                                                           log_transform=log_transform, 
        #                                                           fit_residuals=fit_residuals, 
        #                                                           set_to_normalize=set_to_normalize,
        #                                                           subtract_median=False, 
        #                                                           subtract_polynomial=True)
        #     print(np.shape(gaussian_processes), np.shape(phase_grid),np.shape(wl_grid))
            
        #     # fig = plt.figure()
        #     # ax = fig.add_subplot(111, projection='3d')
            
        #     # if log_transform is not False:
        #     #     X, Y = np.meshgrid(np.exp(phase_grid) - log_transform, 10**wl_grid)
        #     # else:
        #     #     X, Y = np.meshgrid(phase_grid, wl_grid)

        #     # # Z = median_gp

        #     # ax.plot_surface(X, Y, Z)
        #     # #ax.axes.set_zlim3d(bottom=-5, top=5)
        #     # ax.invert_zaxis()
        #     # ax.set_xlabel('Phase Grid')
        #     # ax.set_ylabel('Wavelengths')
        #     # ax.set_zlabel('Magnitude')
        #     # plt.title('Final Polynomial GP Fit')
        #     # plt.show()

        # else:
        #     raise Exception("Must toggle either subtract_median or subtract_polynomial as True to run GP3D")

