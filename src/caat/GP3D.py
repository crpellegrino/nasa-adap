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

from .GP import GP
from .Plot import Plot
from .Diagnostics import Diagnostic
from .DataCube import DataCube
from .SN import SN
from .SNCollection import SNCollection, SNType
from .SNModel import SNModel, SurfaceArray
from caat.utils import colors
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")

class GP3D(GP):
    """
    Class to perform GP fitting simultaneously across wavelength and phase
    for a given collection of SNe.
    Reads in a list of SNe to fit, as well as a collection of SNe to normalize
    that sample against, along with a set of optional flags for different
    fitting routines and parameters. 
    Each SN in the science and comparison samples should already have its photometric
    data pre-processed using the routines available in the `SN` class, run within the
    `DataCube` class, and read directly from the SN datacube files created by the latter. 
    """

    def __init__(
            self,
            collection: Union[SNCollection, SNType], 
            kernel: Union[RBF, Matern, WhiteKernel],
            filtlist: list,
            phasemin: int,
            phasemax: int,
            set_to_normalize: Union[SNCollection, SNType, None] = None,
            use_fluxes: bool = False, 
            log_transform: bool = False, 
            mangle_sed: bool = False
        ):

        super().__init__(collection, kernel, filtlist, phasemin, phasemax, use_fluxes, log_transform)
        self.set_to_normalize = set_to_normalize
        self.mangle_sed = mangle_sed

        self.prepare_data()

    def prepare_data(self):
        """
        Use the flags set in __init__ to filter the pandas dataframes for each SN
        in the science and control samples
        """
        for collection in [self.collection, self.set_to_normalize]:
            for sn in collection.sne:
                # Read the correct cube based on self.mangle_sed
                if self.mangle_sed:
                    data_cube_filename = os.path.join(
                        sn.base_path,
                        sn.classification,
                        sn.subtype,
                        sn.name,
                        sn.name + "_datacube_mangled.csv"
                    )
                else:
                    data_cube_filename = os.path.join(
                        sn.base_path,
                        sn.classification,
                        sn.subtype,
                        sn.name,
                        sn.name + "_datacube.csv"
                    )                    
                if os.path.exists(data_cube_filename):
                    cube = pd.read_csv(data_cube_filename)
                else:
                    # For now, we'll just construct it
                    datacube = DataCube(sn=sn)
                    datacube.construct_cube()
                    cube = datacube.cube

                # Drop rows that are out of the phase range
                inds_to_drop_phase = cube.loc[(cube['Phase'] < self.phasemin) | (cube['Phase'] > self.phasemax)].index
                cube = cube.drop(inds_to_drop_phase).reset_index(drop=True)

                # Drop nondetections farther than 2 days away from first/last detection
                try:
                    min_phase = min(cube.loc[cube['Nondetection'] == False]['Phase'].values)
                    inds_to_drop_nondets_before_first_det = cube.loc[(cube['Nondetection'] == True) & (cube['Phase'] < (min_phase - 2.0))].index
                    cube = cube.drop(inds_to_drop_nondets_before_first_det).reset_index(drop=True)
                except ValueError: # no values
                    pass

                try:
                    max_phase = max(cube.loc[cube['Nondetection'] == False]['Phase'].values)
                    inds_to_drop_nondets_after_last_det = cube.loc[(cube['Nondetection'] == True) & (cube['Phase'] > (max_phase + 2.0))].index
                    cube = cube.drop(inds_to_drop_nondets_after_last_det).reset_index(drop=True)
                except ValueError: # no values
                    pass

                # Drop rows corresponding to filters not in the user-provided filter list
                inds_to_drop_filts = cube.loc[~cube['Filter'].isin(self.filtlist)].index
                cube = cube.drop(inds_to_drop_filts).reset_index(drop=True)

                # Log transform the data (as a separate column), if desired
                if self.log_transform is not False:
                    cube['LogPhase'] = np.log(cube['Phase'].values.astype(float) + self.log_transform)
                    cube['LogWavelength'] = np.log10(cube['Wavelength'].values.astype(float))
                    cube['LogShiftedWavelength'] = np.log10(cube['ShiftedWavelength'].values.astype(float))
                
                # Drop nondetections that are within the phase range and less constraining
                # than the first or last detection in each filter
                try:
                    min_flux_before_peak = min(cube.loc[(cube['Nondetection'] == False) & (cube['Phase'] < 0)]['Flux'].values)
                    inds_to_drop_nondets_before_peak = cube.loc[(cube['Nondetection'] == True) & (cube['Phase'] < 0) & (cube['Flux'] > min_flux_before_peak)].index
                    cube = cube.drop(inds_to_drop_nondets_before_peak).reset_index(drop=True)
                except ValueError: # No values pre-peak, so pass
                    pass

                try:
                    min_flux_after_peak = min(cube.loc[(cube['Nondetection'] == False) & (cube['Phase'] > 0)]['Flux'].values)
                    inds_to_drop_nondets_after_peak = cube.loc[(cube['Nondetection'] == True) & (cube['Phase'] > 0) & (cube['Flux'] > min_flux_after_peak)].index
                    cube = cube.drop(inds_to_drop_nondets_after_peak).reset_index(drop=True)
                except ValueError: # No values post-peak, so pass
                    pass

                # Drop nondetections between the first and last detection
                cube_only_dets = cube.loc[cube['Nondetection'] == False]
                if len(cube_only_dets['Phase'].values) > 0:
                    first_detection = min(cube_only_dets['Phase'].values)
                    last_detection = max(cube_only_dets['Phase'].values)
                    inds_to_drop_nondets_between_dets = cube.loc[
                        (
                            cube['Phase'] > first_detection
                        ) & (
                            cube['Phase'] < last_detection
                        ) & (
                            cube['Nondetection'] == True
                        )
                    ].index
                    cube = cube.drop(inds_to_drop_nondets_between_dets).reset_index(drop=True)
                
                sn.cube = cube


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

    def build_samples(
        self,
        filt,
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

        phases, mags, errs, wls = super().process_dataset(
            filt,
            log_transform=log_transform,
            sn_set=sn_set,
            use_fluxes=use_fluxes,
        )

        if len(phases) == 0:
            return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])

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

        err_grid = np.nan_to_num(err_grid, nan=0.01 if use_fluxes else 0.1)

        return phases.astype(float), wls.astype(float), mags.astype(float), err_grid.astype(float)

    def process_dataset(self, set_to_normalize=None):
        """
        Processes the data set for the GP3D object's SN collection or
        (optionally) a SN set filter-by-filter and returns
        dataframes of the SN collection's photometric details
        or the photometric details of the SN set to normalize to
        """

        ### Create the template grid from the observations
        if set_to_normalize is not None:
            (
                all_template_phases,
                all_template_wls,
                all_template_mags,
                all_template_errs,
            ) = ([], [], [], [])

            for filt in self.filtlist:
                phases, wl_grid, mags, err_grid = self.build_samples(
                filt,
                log_transform=self.log_transform,
                sn_set=set_to_normalize,
                use_fluxes=self.use_fluxes,
            )                   

                all_template_phases = np.concatenate((all_template_phases, phases.flatten()))
                all_template_wls = np.concatenate((all_template_wls, wl_grid.flatten()))
                all_template_mags = np.concatenate((all_template_mags, mags.flatten()))
                all_template_errs = np.concatenate((all_template_errs, err_grid.flatten()))
        
        else:
            # Create grid from the SN collection instead
            all_phases, all_wls, all_mags, all_errs = [], [], [], []

            for filt in self.filtlist:
                phases, wl_grid, mags, err_grid = self.build_samples(
                    filt,
                    log_transform=self.log_transform,
                    use_fluxes=self.use_fluxes,
                )

                all_phases = np.concatenate((all_phases, phases.flatten()))
                all_wls = np.concatenate((all_wls, wl_grid.flatten()))
                all_mags = np.concatenate((all_mags, mags.flatten()))
                all_errs = np.concatenate((all_errs, err_grid.flatten()))

            all_template_phases = all_phases
            all_template_wls = all_wls
            all_template_mags = all_mags
            all_template_errs = all_errs

        template_df = pd.DataFrame(
            {
                'Phase': all_template_phases, 
                'Wavelength': all_template_wls, 
                'Mag': all_template_mags, 
                'MagErr': all_template_errs
            }
        )

        return template_df

    def construct_median_grid(
        self,
        phasemin,
        phasemax,
        filtlist,
        template_df,
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
                    inds = template_df[
                        (np.exp(template_df['Phase']) - np.exp(phase_grid[i]) <= np.log(5.0))
                        & (np.exp(template_df['Phase']) - np.exp(phase_grid[i] > 0.0))
                        & (abs(10**template_df['Wavelength'] - 10**wl_grid[j]) <= 500)
                    ].index
                else:
                    inds = template_df[
                        (template_df['Phase'] - phase_grid[i] <= np.log(5.0))
                        & (template_df['Phase'] - phase_grid[i] > 0.0)
                        & (abs(template_df['Wavelength'] - wl_grid[j]) <= 500)
                    ].index

                if len(inds) > 0:

                    median_mag = np.median(template_df["Mag"][inds].values)
                    iqr = np.subtract(*np.percentile(template_df["Mag"][inds], [75, 25]))

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
        template_df,
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

            wl_grid = np.arange(
                np.log10(
                    min(self.wle[f] for f in filtlist) - 500
                ),
                np.log10(
                    max(self.wle[f] for f in filtlist) + 500
                ),
                0.01
            )

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
                
        ### Add an array of fake measurements to anchor the ends of the fit
        if log_transform is not False:
            anchor_inds_begin = np.where((abs(template_df["Phase"] - np.log(phasemin + log_transform)) < 1.0))[0]
            if len(anchor_inds_begin) > 0:
                anchor_mag_begin = min(template_df["Mag"][anchor_inds_begin].values)
            else:
                anchor_mag_begin = -4.0

            anchor_inds_end = np.where((abs(template_df["Phase"] - np.log(phasemax + log_transform)) < 0.3))[0]
            if len(anchor_inds_end) > 0:
                anchor_mag_end = min(template_df["Mag"][anchor_inds_end].values)
            else:
                anchor_mag_end = -4.0

            anchor_phases = np.asarray(
                [
                    np.log(phasemin + log_transform),
                    np.log(phasemin + 2.5 + log_transform),
                    np.log(phasemax + log_transform),
                ]
            )
            anchor_mags = np.asarray([anchor_mag_begin - 1.0, anchor_mag_begin, anchor_mag_end - 1.0])

        else:
            anchor_phases = np.asarray([phasemin, phasemin + 2.5, phasemax])
            anchor_mags = np.asarray([-5.0, -4.0, -5.0])

        for j in range(len(wl_grid)):

            ### Get all data that falls within this wl +- 500 A
            if log_transform is not False:
                inds = template_df[abs(10**template_df["Wavelength"] - 10**wl_grid[j]) <= 499].index

            else:
                inds = template_df[abs(template_df["Mag"] - wl_grid[j]) <= 499].index

            if len(inds) > 0:

                fit_coeffs = np.polyfit(
                    np.concatenate((template_df["Phase"][inds], anchor_phases)),
                    np.concatenate((template_df["Mag"][inds], anchor_mags)),
                    3,
                    w=1
                    / (
                        np.sqrt(
                            (
                                np.concatenate(
                                    (
                                        template_df["MagErr"][inds],
                                        np.ones(len(anchor_phases)) * 0.05,
                                    )
                                )
                            )
                            ** 2
                            + (np.ones(len(template_df["MagErr"][inds]) + len(anchor_phases)) * 0.1) ** 2
                        )
                    ),
                )
                fit = np.poly1d(fit_coeffs)
                grid_mags = fit(phase_grid)

                mag_grid[:, j] = grid_mags
                err_grid[:, j] = np.ones(len(phase_grid)) * np.median(abs(template_df["Mag"][inds] - fit(template_df["Phase"][inds])))

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
        filtlist,
        phase_grid,
        wl_grid,
        mag_grid,
        err_grid,
        log_transform=False,
        plot=False,
        use_fluxes=False
    ):
        """
        Takes the (shifted) photometry from a given SN and subtracts from it
        the template grid constructed from either the median of all the normalization
        SN photometry, or the polynomial fit to all the normalization SN photometry
        Returns the phase and wavelength of each data point from the given SN, as well as
        the residuals in its magnitude (or flux) and its uncertainty
        """

        ### Subtract off templates for each SN LC
        residuals = []
        for filt in filtlist:
            if filt in sn.cube['ShiftedFilter'].values:
                if use_fluxes:
                    mags = sn.cube.loc[sn.cube['ShiftedFilter']==filt]['ShiftedFlux'].values
                    errs = sn.cube.loc[sn.cube['ShiftedFilter']==filt]['ShiftedFluxerr'].values
                    current_nondets = sn.cube.loc[sn.cube['ShiftedFilter']==filt]['Nondetection'].values
                    current_wls = sn.cube.loc[sn.cube['ShiftedFilter']==filt]['ShiftedWavelength' if self.log_transform is False else 'LogShiftedWavelength'].values

                else:
                    mags = sn.cube.loc[sn.cube['ShiftedFilter']==filt]['ShiftedMag'].values
                    errs = sn.cube.loc[sn.cube['ShiftedFilter']==filt]['Magerr'].values
                    current_nondets = sn.cube.loc[sn.cube['ShiftedFilter']==filt]['Nondetection'].values
                    current_wls = sn.cube.loc[sn.cube['ShiftedFilter']==filt]['ShiftedWavelength' if self.log_transform is False else 'LogShiftedWavelength'].values

                phases = sn.cube.loc[sn.cube['ShiftedFilter']==filt]['Phase' if self.log_transform is False else 'LogPhase'].values
            else:
                phases = []

            # There's a bug that I have to track down where sometimes filters without data
            # have a NaN as their redshift until data is read in, so the second clause
            # of this if statement will skip over those filters
            # (I don't know why this isn't caught above)
            if len(phases) > 0 and not np.isnan(sn.info.get("z", 0)):
                for i, phase in enumerate(phases):
                    ### Get index of current phase in phase grid
                    if log_transform is not False:
                        ### The phase corresponding to phase_ind is no more than the phase grid spacing away from the true phase being measured
                        phase_ind = np.argmin(abs(np.exp(phase_grid) - np.exp(phase)))
                    else:
                        phase_ind = np.argmin(abs(phase_grid - phase))

                    wl_ind = np.argmin(abs(wl_grid - current_wls[i]))

                    if np.isnan(mag_grid[phase_ind, wl_ind]):
                        logger.warning(f"NaN Found: phase {np.exp(phase)}, wl {10**wl_grid[wl_ind]}")
                        continue

                    if np.isinf(mags[i] - mag_grid[phase_ind, wl_ind]):
                        logger.warning(f"Infinity found: phase {np.exp(phase)}, wl {10**wl_grid[wl_ind]}")
                        continue

                    residuals.append(
                        {
                            "Filter": filt,
                            "Phase": phase,
                            "Wavelength": current_wls[i],
                            "MagResidual": mags[i] - mag_grid[phase_ind, wl_ind],
                            "MagErr": errs[i],
                            "Mag": mags[i],
                            "Nondetection": current_nondets[i],
                        }
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

        return pd.DataFrame(residuals)
    
    def build_test_wavelength_phase_grid_from_photometry(self, measured_wavelengths, measured_phases, wl_grid, phase_grid):
        """
        Function to build a uniform grid of wavelengths and phases given photometry
        in the form of measured wavelengths and phases as well as a wavelength and phase grid
        corresponding to the template SED grid
        """
        if self.log_transform is not False:
            waves_to_predict = np.unique(measured_wavelengths)
            diffs = abs(
                np.subtract.outer(10**wl_grid, 10**waves_to_predict)
            )  # The difference between our measurement wavelengths and the wl grid

        else:
            waves_to_predict = np.unique(measured_wavelengths)
            diffs = abs(np.subtract.outer(wl_grid, waves_to_predict))

        phases_to_predict = np.unique(measured_phases)

        ### Compare the wavelengths of our measured filters to those in the wl grid
        ### and fit for those grid wls that are within 500 A of one of our measurements
        wl_inds_fitted = np.unique(np.where((diffs < 500.0))[0])
        phase_inds_fitted = np.unique(np.where((phase_grid >= min(phases_to_predict)) & (phase_grid <= max(phases_to_predict)))[0])
        if len(phase_inds_fitted) == 0:
            return [], [], [], [], None
        linear_phases = np.exp(phase_grid[phase_inds_fitted]) - self.log_transform
        phases = np.log(linear_phases - min(linear_phases) + 0.1)
        
        x, y = np.meshgrid(phases, wl_grid[wl_inds_fitted])
        
        return x, y, wl_inds_fitted, phase_inds_fitted, min(linear_phases)
    
    def run_gp_on_full_sample(
        self,
        plot=False,
        subtract_median=False,
        subtract_polynomial=False,
    ):
        """
        Function to run the Gaussian Process Regression on full sample together
        ===============================================
        Takes as input:
        plot: Optional flag to plot fits and intermediate figures
        subtract_median: Flag to calculate residuals by subtracting the SN magnitude/flux from a median template grid
        subtract_polynomial: Flag to calculate residuals by subtracting the SN magnitude/flux from a polynomial template grid
        """
        template_df = self.process_dataset(set_to_normalize=self.set_to_normalize)

        if subtract_polynomial:
            phase_grid, wl_grid, mag_grid, err_grid = self.construct_polynomial_grid(
                self.phasemin,
                self.phasemax,
                self.filtlist,
                template_df,
                log_transform=self.log_transform,
                plot=plot,
                use_fluxes=self.use_fluxes,
            )
        elif subtract_median:
            phase_grid, wl_grid, mag_grid, err_grid = self.construct_median_grid(
                self.phasemin,
                self.phasemax,
                self.filtlist,
                template_df,
                log_transform=self.log_transform,
                plot=plot,
                use_fluxes=self.use_fluxes,
            )
        else:
            raise Exception("Must toggle either subtract_median or subtract_polynomial as True to run GP3D")
        
        x_input, y_input, err, raw_mags = [], [], [], []

        for sn in self.collection.sne:
            
            residuals = self.subtract_data_from_grid(
                sn,
                self.filtlist,
                phase_grid,
                wl_grid,
                mag_grid,
                err_grid,
                log_transform=self.log_transform,
                plot=False,  # TODO: are we sure we want to hard-code False for grid subtraction?
                use_fluxes=self.use_fluxes
            )

            if len(residuals) == 0:
                continue

            if self.log_transform is not False:
                phase_residuals_linear = np.exp(residuals["Phase"].values) - self.log_transform
                phases_to_fit = np.log(phase_residuals_linear - min(phase_residuals_linear) + 0.1)

            else:
                phases_to_fit = residuals["Phase"].values

            if not len(x_input):
                x_input = np.vstack((phases_to_fit, residuals["Wavelength"].values)).T
            else:
                x_input = np.concatenate([x_input, np.vstack((phases_to_fit, residuals["Wavelength"].values)).T])
            y_input = np.concatenate([y_input, residuals["MagResidual"].values])
            raw_mags = np.concatenate([raw_mags, residuals["Mag"].values])
            err = np.concatenate([err, residuals["MagErr"].values])

        gaussian_process = GaussianProcessRegressor(kernel=self.kernel, alpha=err, n_restarts_optimizer=10)
        gaussian_process.fit(x_input, y_input)

        x, y, wl_inds_fitted, phase_inds_fitted, phase_offset = self.build_test_wavelength_phase_grid_from_photometry(
            x_input[:,1], x_input[:,0], wl_grid, phase_grid
        )

        test_prediction, std_prediction = gaussian_process.predict(np.vstack((x.ravel(), y.ravel())).T, return_std=True)

        template_mags = []
        for wl_ind in wl_inds_fitted:
            for phase_ind in phase_inds_fitted:
                template_mags.append(mag_grid[phase_ind, wl_ind])

        template_mags = np.asarray(template_mags).reshape((len(x), -1))

        final_prediction = test_prediction.reshape((len(x), -1)) + template_mags
        final_std_prediction = std_prediction.reshape((len(x), -1))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        Plot().plot_construct_grid(
            gp_class=self, 
            X = np.exp(x) + phase_offset - 0.1,
            Y=10**(y), 
            Z=final_prediction,
            ax=ax, 
            Z_lower= final_prediction - 1.96 * final_std_prediction, 
            Z_upper=final_prediction + 1.96 * final_std_prediction,
            grid_type="final", 
            use_fluxes=self.use_fluxes
        )
        ax.scatter(np.exp(x_input[:,0]) - self.log_transform, 10**x_input[:,1], raw_mags, marker='o', color='k')
        plt.show()
        return gaussian_process, mag_grid, phase_grid, wl_grid

    def optimize_hyperparams(
        self,
        subtract_median=False,
        subtract_polynomial=False
    ):
        """
        Function to optimize kernel hyperparameters by fitting
        the SNe in the collection individually
        ===============================================
        Takes as input:
        subtract_median: Flag to calculate residuals by subtracting the SN magnitude/flux from a median template grid
        subtract_polynomial: Flag to calculate residuals by subtracting the SN magnitude/flux from a polynomial template grid
        Returns:
        kernel_params: The optimized kernel hyperparameters
        """

        template_df = self.process_dataset(set_to_normalize=self.set_to_normalize)
        kernel_params = []

        if subtract_polynomial:
            phase_grid, wl_grid, mag_grid, err_grid = self.construct_polynomial_grid(
                self.phasemin,
                self.phasemax,
                self.filtlist,
                template_df,
                log_transform=self.log_transform,
                plot=False,
                use_fluxes=self.use_fluxes,
            )
        elif subtract_median:
            phase_grid, wl_grid, mag_grid, err_grid = self.construct_median_grid(
                self.phasemin,
                self.phasemax,
                self.filtlist,
                template_df,
                log_transform=self.log_transform,
                plot=False,
                use_fluxes=self.use_fluxes,
            )
        else:
            raise Exception("Must toggle either subtract_median or subtract_polynomial as True to run GP3D")

        for sn in self.collection.sne:
            
            residuals = self.subtract_data_from_grid(
                sn,
                self.filtlist,
                phase_grid,
                wl_grid,
                mag_grid,
                err_grid,
                log_transform=self.log_transform,
                plot=False,  # TODO: are we sure we want to hard-code False for grid subtraction?
                use_fluxes=self.use_fluxes
            )

            if len(residuals) == 0:
                continue

            if self.log_transform is not False:
                phase_residuals_linear = np.exp(residuals["Phase"].values) - self.log_transform
                phases_to_fit = np.log(phase_residuals_linear - min(phase_residuals_linear) + 0.1)

            else:
                phases_to_fit = residuals["Phase"].values

            x = np.vstack((phases_to_fit, residuals["Wavelength"].values)).T
            y = residuals["MagResidual"].values
            if len(y) > 1:
                # We have enough points to fit
                err = residuals["MagErr"].values

                gaussian_process = GaussianProcessRegressor(kernel=self.kernel, alpha=err, n_restarts_optimizer=10)
                gaussian_process.fit(x, y)

                kernel_params.append(gaussian_process.kernel_.theta)

        optimized_kernel_hyperparams = np.asarray(
            [
                np.median(
                    [k[i] for k in kernel_params]
                ) for i in range(len(kernel_params[0]))
            ]
        )
        self.kernel.set_params(
            **{
                "length_scale": optimized_kernel_hyperparams, 
                "length_scale_bounds": "fixed"
            }
        )
        return kernel_params
    
    def iteratively_warp_sed(
        self,
        residuals: pd.DataFrame,
        test_prediction_reshaped: np.ndarray,
        wls_fit: np.ndarray,
        phases_fit: np.ndarray,
        sn: SN,
        convergence_threshold: float = 1.1,
        niter: float = 100
    ):
        """
        Compare photometry to prediction, iteratively warping the 
        predicted flux until it matches the photometry
        """
        for phase in residuals["Phase"].values:
            current_lc_inds = np.where((abs(residuals['Phase'] - phase) <= 0.1))[0]

            if len(current_lc_inds) > 0 and len({filt for filt in residuals['Filter'][current_lc_inds]}) > 1: 
                # We have data in at least two filters at this epoch
                current_lc_cube = residuals[abs(residuals['Phase'] - phase) <= 0.1]
                current_lc = current_lc_cube['Mag'].values
                current_lc = np.concatenate(([0.0], current_lc, [current_lc[-1]/2]))

                errors = np.ones(len(current_lc_inds)) * 100.0
                errors = np.concatenate(([1.0], errors, [1.0]))
                if self.log_transform is not False:
                    wls = [np.log10(self.wle[filt] * (1 + sn.info.get("z", 0))) for filt in current_lc_cube["Filter"].values]
                    wls = np.concatenate(([np.log10(10**min(wls) - 500)], wls, [np.log10(10**max(wls) + 500)]))
                else:
                    wls = [self.wle[filt] * (1 + sn.info.get("z", 0)) for filt in current_lc_cube["Filter"].values]
                    wls = np.concatenate(([min(wls) - 500], wls, [max(wls) + 500]))
                
                sed_inds_at_this_phase = np.where(
                    (
                        wls_fit > min(wls)
                    ) & (
                        wls_fit < max(wls)
                    )
                )[0]
                phase_ind = np.argmin(abs(phases_fit - phase))

                prediction_slice = np.copy(test_prediction_reshaped[sed_inds_at_this_phase, phase_ind])
                n = 0

                for i in range(niter):
                    if all(1 + abs(1 - errors) <= convergence_threshold) or n == niter:
                        break

                    for j, filt in enumerate(current_lc_cube['Filter']):                                       
                        real_flux_inds = np.where((current_lc_cube['Filter'] == filt))[0]+1

                        if len(real_flux_inds) > 1:
                            real_mag = np.average(current_lc[real_flux_inds])
                        else:
                            real_mag = current_lc[real_flux_inds][0]

                        if self.log_transform is not False:
                            wl_for_filt = np.log10(self.wle[filt] * (1 + sn.info.get("z", 0)))
                        else:
                            wl_for_filt = self.wle[filt] * (1 + sn.info.get("z", 0))
                        wl_ind = np.argmin(abs(wls_fit[sed_inds_at_this_phase] - wl_for_filt))
                        
                        predicted_mag = prediction_slice[wl_ind]

                        try:
                            diff = real_mag - predicted_mag
                            if predicted_mag > 0:
                                error = 1 + diff
                            else:
                                error = 1 - diff
                        except ZeroDivisionError:
                            error = 100
                        
                        errors[j+1] = error

                    if any(1 + abs(1 - errors) > convergence_threshold):
                        ### Warp the SED by an interpolation of the error across wavelength
                        ### and rerun the loop
                        if n < niter:
                            n += 1
                            if any(errors > 1e3) or any(np.isinf(errors)):
                                n = niter

                        interp_error_fn = interp1d(wls, errors)
                        interp_to_warp_by = interp_error_fn(wls_fit[sed_inds_at_this_phase])
                        prediction_slice *= interp_to_warp_by

                        if n == niter:
                            logger.debug(f'Couldnt iterate to match flux: {sn.name}, {np.exp(phase) - self.log_transform}')
                
                if n < niter:
                    # Put the warped SED back in the predicted datacube
                    test_prediction_reshaped[sed_inds_at_this_phase, phase_ind] = prediction_slice
        
        return test_prediction_reshaped

    def run_gp_individually(
        self,
        plot=False,
        subtract_median=False,
        subtract_polynomial=False,
        interactive=False,
        run_diagnostics=False,
    ):
        """
        Function to run the Gaussian Process Regression on each SN individually
        ===============================================
        Takes as input:
        plot: Optional flag to plot fits and intermediate figures
        subtract_median: Flag to calculate residuals by subtracting the SN magnitude/flux from a median template grid
        subtract_polynomial: Flag to calculate residuals by subtracting the SN magnitude/flux from a polynomial template grid
        interactive: Flag to interactively choose to include each GP fit to the final median-combined template
        run_diagnostics: Flag to run diagnostic tests to ensure reasonable fits
        """
        if interactive:
            plot = True

        template_df = self.process_dataset(set_to_normalize=self.set_to_normalize)
        gaussian_processes = []

        if subtract_polynomial:
            phase_grid, wl_grid, mag_grid, err_grid = self.construct_polynomial_grid(
                self.phasemin,
                self.phasemax,
                self.filtlist,
                template_df,
                log_transform=self.log_transform,
                plot=plot,
                use_fluxes=self.use_fluxes,
            )
        elif subtract_median:
            phase_grid, wl_grid, mag_grid, err_grid = self.construct_median_grid(
                self.phasemin,
                self.phasemax,
                self.filtlist,
                template_df,
                log_transform=self.log_transform,
                plot=plot,
                use_fluxes=self.use_fluxes,
            )
        else:
            raise Exception("Must toggle either subtract_median or subtract_polynomial as True to run GP3D")

        for sn in self.collection.sne:
            
            residuals = self.subtract_data_from_grid(
                sn,
                self.filtlist,
                phase_grid,
                wl_grid,
                mag_grid,
                err_grid,
                log_transform=self.log_transform,
                plot=False,  # TODO: are we sure we want to hard-code False for grid subtraction?
                use_fluxes=self.use_fluxes
            )

            if len(residuals) == 0:
                continue

            if self.log_transform is not False:
                phase_residuals_linear = np.exp(residuals["Phase"].values) - self.log_transform
                phases_to_fit = np.log(phase_residuals_linear - min(phase_residuals_linear) + 0.1)

            else:
                phases_to_fit = residuals["Phase"].values

            x = np.vstack((phases_to_fit, residuals["Wavelength"].values)).T
            y = residuals["MagResidual"].values
            if len(y) > 1:
                # We have enough points to fit
                err = residuals["MagErr"].values

                gaussian_process = GaussianProcessRegressor(kernel=self.kernel, alpha=err, n_restarts_optimizer=10)
                gaussian_process.fit(x, y)

                if plot:
                    fig, ax = Plot().create_empty_subplot()

                filts_fitted = []

                for filt in self.filtlist:

                    if self.log_transform is not False:
                        
                        test_times_linear = np.arange(
                            min(phase_residuals_linear),
                            max(phase_residuals_linear),
                            1.0 / 24
                        ) 
                        test_times = np.log(test_times_linear - min(phase_residuals_linear) + 0.1)

                        test_waves = np.ones(len(test_times)) * np.log10(self.wle[filt] * (1 + sn.info.get("z", 0)))
                    else:
                        if (
                            not self.mangle_sed 
                            and len(
                                residuals[
                                    residuals["Wavelength"] == self.wle[filt]
                                ]["Wavelength"].values * (1 + sn.info.get("z", 0))
                            ) == 0
                        ):
                            continue
                        test_times = np.arange(self.phasemin, self.phasemax, 1.0 / 24)
                        test_waves = np.ones(len(test_times)) * self.wle[filt] * (1 + sn.info.get("z", 0))

                    ### Trying to convert back to normalized magnitudes here
                    if self.log_transform is not False:
                        wl_ind = np.argmin(abs(10**wl_grid - self.wle[filt] * (1 + sn.info.get("z", 0))))
                    else:
                        wl_ind = np.argmin(abs(wl_grid - self.wle[filt] * (1 + sn.info.get("z", 0))))

                    template_mags = []
                    if self.log_transform is not False:
                        for i in range(len(test_times_linear)):
                            j = np.argmin(abs(np.exp(phase_grid) - self.log_transform - test_times_linear[i]))
                            template_mags.append(mag_grid[j, wl_ind])
                    else:
                        for i in range(len(test_times)):
                            j = np.argmin(abs(phase_grid - test_times[i]))
                            template_mags.append(mag_grid[j, wl_ind])

                    template_mags = np.asarray(template_mags)

                    test_prediction, std_prediction = gaussian_process.predict(np.vstack((test_times, test_waves)).T, return_std=True)
                    if self.log_transform is not False:
                        test_times = np.exp(test_times) + min(phase_residuals_linear) - 0.1
                    
                    # Plot the SN photometry
                    shifted_mjd = sn.cube[sn.cube['Filter']==filt]['Phase'].values.astype(float)
                    if self.log_transform is not False:
                        shifted_mjd = sn.log_transform_time(shifted_mjd, phase_start=self.log_transform)

                    if self.log_transform is not False:
                        inds_to_fit = np.where(
                            (shifted_mjd > np.log(self.phasemin + self.log_transform)) & (shifted_mjd < np.log(self.phasemax + self.log_transform))
                        )[0]
                    else:
                        inds_to_fit = np.where((shifted_mjd > self.phasemin) & (shifted_mjd < self.phasemax))[0]

                    residuals_for_filt = residuals[
                        (residuals["Filter"] == filt)
                        & (residuals["Phase"] > np.log(self.phasemin + self.log_transform))
                        & (residuals["Phase"] < np.log(self.phasemax + self.log_transform))
                        & (residuals["Nondetection"] == False)
                    ]
                    if len(inds_to_fit) > 0:
                        filts_fitted.append(filt)

                        if plot:
                            key_to_plot = "flux" if self.use_fluxes else "mag"
                            try:
                                Plot().plot_run_gp_overlay(
                                    ax=ax,
                                    sn_class=sn,
                                    test_times=test_times,
                                    test_prediction=test_prediction,
                                    std_prediction=std_prediction,
                                    template_mags=template_mags,
                                    residuals=residuals_for_filt,
                                    phase_residuals=residuals["Phase"].values,
                                    wl_residuals=residuals["Wavelength"].values,
                                    err_residuals=residuals["MagErr"].values,
                                    inds_to_fit=inds_to_fit,
                                    log_transform=self.log_transform,
                                    filt=filt,
                                    use_fluxes=self.use_fluxes,
                                    key_to_plot=key_to_plot,
                                )
                            except:
                                continue

                    if run_diagnostics:
                        d = Diagnostic()
                        d.identify_outlier_points(
                            filt,
                            test_times,
                            test_prediction + template_mags,
                            std_prediction,
                            np.exp(
                                residuals_for_filt["Phase"].values
                            )-self.log_transform,
                            residuals_for_filt["Mag"].values,
                            residuals_for_filt["MagErr"].values,
                            log_transform=self.log_transform
                        )

                        d.check_late_time_slope(
                            filt,
                            test_times,
                            test_prediction + template_mags,
                            np.exp(residuals_for_filt["Phase"].values) - self.log_transform,
                            use_fluxes=self.use_fluxes                                
                        )

                    if (subtract_median or subtract_polynomial) and interactive:
                        use_for_template = input("Use this fit to construct a template? y/n")
                
                x, y, wl_inds_fitted, phase_inds_fitted, phase_offset = self.build_test_wavelength_phase_grid_from_photometry(
                    residuals["Wavelength"].values, residuals["Phase"].values, wl_grid, phase_grid
                )
                if len(x) == 0:
                    continue
                try:
                    test_prediction, std_prediction = gaussian_process.predict(np.vstack((x.ravel(), y.ravel())).T, return_std=True)
                except Exception as e:
                    logger.warning(f'WARNING:   BROKEN FIT FOR {sn.name}', exc_info=e)
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
                
                test_prediction_reshaped = self.iteratively_warp_sed(
                    residuals, 
                    test_prediction_reshaped,
                    wl_grid[wl_inds_fitted],
                    phase_grid[phase_inds_fitted],
                    sn
                )
                
                test_prediction_smoothed = np.empty(test_prediction_reshaped.shape)
                for i, col in enumerate(test_prediction_reshaped.T):
                    window_size = max(int(round(len(col) / (2*len(filts_fitted)), 0)), 5) # Window size of approximately half a filter length scale
                    if window_size % 2 == 0:
                        window_size += 1 # Must be odd
                    # Use astropy convolve function to handle NaNs
                    test_prediction_smoothed[:,i] = convolve(col, np.ones(window_size)/window_size, boundary='extend') # Boxcar smoothing

                std_prediction_reshaped = std_prediction.reshape((len(x), -1)) + template_mags
                std_prediction_smoothed = np.empty(std_prediction_reshaped.shape)
                for i, col in enumerate(std_prediction_reshaped.T):
                    window_size = max(int(round(len(col) / (2*len(filts_fitted)), 0)), 5)
                    if window_size % 2 == 0:
                        window_size += 1 # Must be odd
                    std_prediction_smoothed[:,i] = convolve(col, np.ones(window_size)/window_size, boundary='extend')

                gp_grid = np.empty((len(wl_grid), len(phase_grid)))
                gp_grid[:] = np.nan
                for i, col in enumerate(test_prediction_smoothed[:,]):
                    current_wl_grid_ind = wl_inds_fitted[i]
                    for j in range(len(col)):
                        current_phase_grid_ind = phase_inds_fitted[j]
                        gp_grid[current_wl_grid_ind, current_phase_grid_ind] = col[j]

                if plot:
                    Plot().plot_run_gp_surface(
                        gp_class=self,
                        x=np.exp(x) + phase_offset - 0.1,
                        y=10**(y),
                        test_prediction_reshaped=test_prediction_smoothed,
                        use_fluxes=self.use_fluxes,
                    )
                    if run_diagnostics:
                        d = Diagnostic()
                        d.check_gradient_between_filters(
                            [self.wle[f] for f in filts_fitted],
                            np.exp(phase_grid[phase_inds_fitted]) - self.log_transform,
                            10**(wl_grid[wl_inds_fitted]),
                            test_prediction_smoothed,
                            std_prediction_smoothed,
                            [-15.0, 0.0, 50.0]
                        )
                        if 'UVM2' in filts_fitted:
                            d.check_uvm2_flux(
                                np.exp(phase_grid[phase_inds_fitted]) - self.log_transform,
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

        return gaussian_processes, phase_grid, wl_grid

    def predict(
        self,
        plot=False,
        subtract_median=False,
        subtract_polynomial=False,
        run_diagnostics=False,
        fit_separately=True,
    ):
        """
        Function to predict light curve behavior using Gaussian Process Regression
        ===============================================
        Takes as input:
        plot: Optional flag to plot fits and intermediate figures
        subtract_median: Flag to calculate residuals by subtracting the SN magnitude/flux from a median template grid
        subtract_polynomial: Flag to calculate residuals by subtracting the SN magnitude/flux from a polynomial template grid
        run_diagnostics: Flag to run diagnostic tests to ensure reasonable fits
        fit_separately: Run GPR on each SN separately and return the resulting SED surfaces from each
        """
        self.prepare_data()

        if not fit_separately:
            gaussian_process, template_mags, phase_grid, wl_grid = self.run_gp_on_full_sample(
                plot=plot,
                subtract_polynomial=subtract_polynomial,
                subtract_median=subtract_median,
            )
            if self.log_transform is not False:
                snmodel = SNModel(
                    phase_grid=np.exp(phase_grid) - self.log_transform,
                    wl_grid=10**wl_grid,
                    filters_fit=self.filtlist,
                    surface=gaussian_process,
                    template_mags=template_mags,
                    sncollection=self.collection,
                    norm_set=self.set_to_normalize,
                    log_transform=self.log_transform
                )
            else:
                snmodel = SNModel(
                    phase_grid=phase_grid,
                    wl_grid=wl_grid,
                    filters_fit=self.filtlist,
                    surface=gaussian_process,
                    template_mags=template_mags,
                    sncollection=self.collection,
                    norm_set=self.set_to_normalize,
                )
            return snmodel

        else:
            ### We're fitting each SN individually and then median combining the full 2D GP
            gaussian_processes, phase_grid, wl_grid = self.run_gp_individually(
                plot=plot,
                subtract_median=subtract_median,
                subtract_polynomial=subtract_polynomial,
                run_diagnostics=run_diagnostics,
            )

            median_gp = np.nanmedian(np.dstack(gaussian_processes), -1)

            if self.log_transform is not False:
                X, Y = np.meshgrid(np.exp(phase_grid) - self.log_transform, 10**wl_grid)
            else:
                X, Y = np.meshgrid(phase_grid, wl_grid)

            median_gp = self.interpolate_grid(median_gp.T, wl_grid, filter_window=31)
            for i, col in enumerate(median_gp.T):
                median_gp[:,i] = savgol_filter(col, 51, 3)
            median_gp = median_gp.T

            iqr_grid = iqr(np.dstack(gaussian_processes), axis=-1, nan_policy='omit')
            iqr_grid = self.interpolate_grid(iqr_grid.T, wl_grid, filter_window=31)
            for i, col in enumerate(iqr_grid.T):
                iqr_grid[:,i] = savgol_filter(col, 51, 3)
            iqr_grid = iqr_grid.T
            
            Z = median_gp

            Plot().plot_construct_grid(
                gp_class=self, 
                X=X, 
                Y=Y, 
                Z=Z, 
                Z_lower=Z-iqr_grid, 
                Z_upper=Z+iqr_grid, 
                grid_type="final", 
                use_fluxes=self.use_fluxes
            )
            plt.show()

            surface = SurfaceArray(
                surface = np.asarray([median_gp, iqr_grid]),
                phase_grid=phase_grid,
                wl_grid = wl_grid
            )
            if self.log_transform is not False:
                snmodel = SNModel(
                    surface=surface,
                    phase_grid=np.exp(phase_grid)-self.log_transform,
                    wl_grid=10**(wl_grid),
                    filters_fit=self.filtlist,
                    sncollection=self.collection,
                    norm_set=self.set_to_normalize,
                    log_transform=self.log_transform
                )
            else:
                snmodel = SNModel(
                    surface=surface,
                    phase_grid=phase_grid,
                    wl_grid=wl_grid,
                    filters_fit=self.filtlist,
                    sncollection=self.collection,
                    norm_set=self.set_to_normalize,
                )
                
            return snmodel
