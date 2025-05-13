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
import astropy.units as u
from dustmaps.sfd import SFDQuery

# from .CAAT import CAAT
# from .GP3D import GP3D
from .Kernels import RBFKernel, WhiteKernel, MaternKernel
from .Plot import Plot
# from .SN import SN
from .SNCollection import SNCollection, SNType
from .DataCube import DataCube
from caat.utils import WLE
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")


class Fitter:  # pylint: disable=too-few-public-methods
    """
    A Fitter object, fitting the light curves of a class (Type) of supernovae
    """

    def __init__(self, collection):
        # collection long term could be a individual SN or a SNCollection,
        # make sure this works at some point
        self.collection = collection


class GP(Fitter):
    """
    GP fit to a single band
    """

    wle = WLE

    def __init__(
            self, 
            sne_collection: Union[SNCollection, SNType], 
            kernel: Union[RBFKernel, WhiteKernel, MaternKernel],
            filtlist: list,
            phasemin: int, 
            phasemax: int, 
            use_fluxes: bool = False, 
            log_transform: bool = False
        ):

        super().__init__(sne_collection)
        self.kernel = kernel
        self.filtlist = filtlist
        self.use_fluxes = use_fluxes
        self.log_transform = log_transform
        self.phasemin = phasemin
        self.phasemax = phasemax

    def prepare_data(self):
        """
        Use the flags set in __init__ to filter the pandas dataframes for each SN
        in the science and control samples
        """
        for sn in self.collection.sne:

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
            
            # Construct anchor points
            # TODO: Implement

            sn.cube = cube

    def process_dataset_for_gp(
        self,
        filt,
        log_transform=False,
        sn_set=None,
        use_fluxes=False,
    ):
        """
        Loads all the data, shifts the data to peak,
        and concatenates the data for the object's SN collection
        or a provided SN set
        """

        phases, mags, errs, wls = (
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
        )

        if sn_set is None:
            sn_set = self.collection

        for sn in sn_set.sne:

            current_phases = sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['Phase'].values if log_transform is False else sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['LogPhase'].values
            current_mags = sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['ShiftedMag'].values if use_fluxes is False else sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['ShiftedFlux']
            current_errs = sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['Magerr'].values if use_fluxes is False else sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['Fluxerr'] if log_transform is False else sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['ShiftedFluxerr']
            current_wls = sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['ShiftedWavelength'].values if log_transform is False else sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['LogShiftedWavelength'].values

            phases = np.concatenate((phases, current_phases))
            mags = np.concatenate((mags, current_mags))
            errs = np.concatenate((errs, current_errs))
            wls = np.concatenate((wls, current_wls))

        return (
            phases.reshape(-1, 1),
            mags.reshape(-1, 1),
            errs.reshape(-1, 1),
            wls.reshape(-1, 1),
        )
    

    def process_sed_dataset_for_gp(
        self,
        filt,
        log_transform=False,
        sn_set=None,
        use_fluxes=False,
    ):
        """
        TODO: Documentation
        """

        phases, mags, errs, wls = (
            [],
            [],
            [],
            []
        )

        if sn_set is None:
            sn_set = self.collection

        try:

            for sn in sn_set.sne:

                if not sn.info.get("peak_filt"):
                    continue

                peak_flux = max(sn.seds[(sn.seds["Wavelength"] > 3500) & (sn.seds["Wavelength"] < 6500)]["Flux"])

                current_phases = sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['Phase'].values if log_transform is False else sn.cube.loc[sn.cube['ShiftedFilter'] == filt]['LogPhase'].values
                for phase in current_phases:
                    if log_transform is not False:
                        linear_phase = np.exp(phase) - log_transform
                        current_sed = sn.seds[abs(sn.seds["Phase"] - linear_phase) < 0.5]
                    else:
                        current_sed = sn.seds[abs(sn.seds["Phase"] - phase) < 0.5]
                    for _, row in current_sed.iterrows():
                        phases.append(phase)
                        if log_transform is not False:

                            shifted_flux = np.log10(row["Flux"] / peak_flux)
                            mags.append(shifted_flux)
                            errs.append(row["FluxErr"])
                            wls.append(np.log10(row["ShiftedWavelength"]))
                        else:
                            wls.append(row["ShiftedWavelength"])
                            mags.append(row["Flux"])
                            errs.append(row["FluxErr"])


            return np.asarray(phases), np.asarray(mags), np.asarray(errs), np.asarray(wls)
        
        except Exception as e: # TODO: remove this block once all data is processed for science and normalization sets
            logger.warning('Ran into an exception', exc_info=e)
            phases, mags, errs, wls = self.process_dataset_for_gp(filt, log_transform=log_transform, sn_set=sn_set, use_fluxes=use_fluxes)
            return phases, mags, errs, wls

    def run_gp(self, filt, test_size):

        self.prepare_data()

        phases, mags, errs, _ = self.process_dataset_for_gp(filt, log_transform=self.log_transform, sn_set=self.collection, use_fluxes=self.use_fluxes)
        X_train, _, Y_train, _, Z_train, _ = train_test_split(phases, mags, errs, test_size=test_size)

        ### Get array of errors at each timestep
        min_phase, max_phase = sorted(X_train)[0], sorted(X_train)[-1]
        phase_grid = np.linspace(min_phase, max_phase, len(X_train))
        phase_grid_space = (max_phase - min_phase) / len(X_train)

        err_grid = np.ones(len(phase_grid))
        for mjd in phase_grid:
            ind = np.where((X_train < mjd + phase_grid_space / 2) & (X_train > mjd - phase_grid_space / 2))[0]
            mags_at_this_phase = Y_train[ind]
            if len(mags_at_this_phase) > 1:
                std_mag = max(np.std(mags_at_this_phase), 0.01)
            elif len(mags_at_this_phase) == 1:
                std_mag = Z_train[ind]
            else:
                std_mag = 0.1
            err_grid[ind] = std_mag

        ### Run the GP
        gaussian_process = GaussianProcessRegressor(kernel=self.kernel, alpha=err_grid, n_restarts_optimizer=9)
        gaussian_process.fit(X_train, Y_train)

        self.gaussian_process = gaussian_process

        return gaussian_process, phases, mags, errs

    def predict_gp(self, filt, test_size, plot=False):

        gaussian_process, phases, mags, errs = self.run_gp(filt, test_size)

        mean_prediction, std_prediction = gaussian_process.predict(sorted(phases), return_std=True)

        if plot:
            Plot().plot_gp_predict_gp(
                gp_class=self,
                phases=phases,
                mean_prediction=mean_prediction,
                std_prediction=std_prediction,
                mags=mags,
                errs=errs,
                filt=filt,
            )