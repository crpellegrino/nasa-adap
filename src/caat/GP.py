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

    wle = {
        "u": 3560,
        "g": 4830,
        "r": 6260,
        "i": 7670,
        "z": 8890,
        "y": 9600,
        "w": 5985,
        "Y": 9600,
        "U": 3600,
        "B": 4380,
        "V": 5450,
        "R": 6410,
        "G": 6730,
        "E": 6730,
        "I": 7980,
        "J": 12200,
        "H": 16300,
        "K": 21900,
        "UVW2": 2030,
        "UVM2": 2231,
        "UVW1": 2634,
        "F": 1516,
        "N": 2267,
        "o": 6790,
        "c": 5330,
        "W": 33526,
        "Q": 46028,
    }

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

    def run_gp(self, filt, phasemin, phasemax, test_size, sn_set=None):

        phases, mags, errs, _ = self.process_dataset_for_gp(filt, phasemin, phasemax, log_transform=self.log_transform, sn_set=sn_set, use_fluxes=self.use_fluxes)
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

    def predict_gp(self, filt, phasemin, phasemax, test_size, plot=False, use_fluxes=False):

        gaussian_process, phases, mags, errs = self.run_gp(filt, phasemin, phasemax, test_size, use_fluxes=use_fluxes)

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
                use_fluxes=use_fluxes,
            )