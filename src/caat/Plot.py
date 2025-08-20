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
import logging
from caat.utils import colors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")


class Plot:
    """
    Plot class allowing any number of diagnostic / step-by-step plots to be created and displayed
    across the SN, SNCollection, GP, and GP3D classes
    """

    def create_empty_subplot(self):
        fig, ax = plt.subplots()
        return fig, ax

    def plot_sn_data(
        self,
        sn_class,
        data_to_plot,
        filts_to_plot,
        plot_fluxes=False,
    ):
        fig, ax = plt.subplots()

        for f in filts_to_plot:
            for filt, mag_list in data_to_plot.items():
                if f and f == filt:
                    if plot_fluxes:
                        mjds = np.asarray([phot["mjd"] for phot in mag_list if not phot.get("nondetection", False)])
                        fluxes = np.asarray([phot["flux"] for phot in mag_list if not phot.get("nondetection", False)])
                        errs = np.asarray([phot["fluxerr"] for phot in mag_list if not phot.get("nondetection", False)])
                        ax.errorbar(
                            mjds,
                            fluxes,
                            yerr=errs,
                            fmt="o",
                            mec="black",
                            color=colors.get(filt, "k"),
                            label=filt,
                        )

                        nondet_mjds = np.asarray([phot["mjd"] for phot in mag_list if phot.get("nondetection", False)])
                        nondet_fluxes = np.asarray([phot["flux"] for phot in mag_list if phot.get("nondetection", False)])
                        nondet_errs = np.asarray([phot["fluxerr"] for phot in mag_list if phot.get("nondetection", False)])
                        ax.errorbar(
                            nondet_mjds,
                            nondet_fluxes,
                            yerr=nondet_errs,
                            fmt="v",
                            alpha=0.5,
                            color=colors.get(filt, "k"),
                        )
                    else:
                        mjds = np.asarray([phot["mjd"] for phot in mag_list if not phot.get("nondetection", False)])
                        mags = np.asarray([phot["mag"] for phot in mag_list if not phot.get("nondetection", False)])
                        errs = np.asarray([phot["err"] for phot in mag_list if not phot.get("nondetection", False)])

                        ax.errorbar(
                            mjds,
                            mags,
                            yerr=errs,
                            fmt="o",
                            mec="black",
                            color=colors.get(filt, "k"),
                            label=filt,
                        )
        if not plot_fluxes:
            plt.gca().invert_yaxis()
            plt.ylabel("Apparent Magnitude")
        else:
            plt.ylabel("Flux")

        plt.legend()
        plt.xlabel("MJD")
        plt.title(sn_class.name)
        plt.minorticks_on()
        plt.show()

    def plot_fit_for_max(
        self,
        sn_class,
        mjd_array,
        mag_array,
        err_array,
        fit_mjds,
        fit_mags,
        fit_errs,
        inds_to_fit,
    ):
        """
        Takes as input arrays for MJD, mag, and err for a filter
        as well as the guess for the MJD of maximum and an array
        to shift the lightcurve over,
        and returns estimates of the peak MJD and mag at peak
        """
        fig, ax = plt.subplots()

        ax.errorbar(mjd_array, mag_array, yerr=err_array, fmt="o", color="black")
        ax.errorbar(
            fit_mjds,
            fit_mags,
            yerr=fit_errs,
            fmt="o",
            color="blue",
            label="Used in Fitting",
        )
        if len(mjd_array[inds_to_fit]) > 0:
            plt.ylim(min(mag_array[inds_to_fit]) - 0.5, max(mag_array[inds_to_fit]) + 0.5)
        plt.xlabel("MJD")
        plt.ylabel("Apparent Magnitude")
        plt.title(sn_class.name)
        plt.legend()
        # plt.gca().invert_yaxis()

        # plt.show()

    def plot_shift_to_max(self, sn_class, mjds, mags, errs, nondets, filt):
        sn = sn_class

        plt.errorbar(
            mjds[np.where((nondets == False))[0]],
            mags[np.where((nondets == False))[0]],
            yerr=errs[np.where((nondets == False))[0]],
            fmt="o",
            mec="black",
            color=colors.get(filt, "k"),
            label=filt + "-band",
        )
        plt.scatter(mjds[np.where((nondets == True))[0]], mags[np.where((nondets == True))[0]], marker="v", color=colors.get(filt, "k"), alpha=0.2)

        plt.xlabel("Shifted Time [days]")
        plt.ylabel("Shifted Magnitude")
        plt.title(sn.name + "-Shifted Data")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

    def plot_all_lcs(self, sn_class, filts=["all"], log_transform=False, plot_fluxes=False, ax=None, show=True):
        """plot all light curves of given subtype/collection
        can plot single, multiple or all bands"""
        sne = sn_class.sne
        logger.info(f"Plotting all {len(sne)} lightcurves in the collection")

        if not ax:
            fig, ax = plt.subplots()
        if filts[0] is not "all":
            filts_to_plot = filts
        else:
            filts_to_plot = colors.keys()

        for i, f in enumerate(filts_to_plot):
            for sn in sne:
                mjds, mags, errs, nondets = sn.shift_to_max(f, shift_fluxes=plot_fluxes)
                if len(mjds) > 0:
                    if log_transform is not False:
                        mjds = sn.log_transform_time(mjds, phase_start=log_transform)

                    if plot_fluxes:

                        nondet_inds = np.where((nondets == False))[0]
                        det_inds = np.where((nondets == True))[0]
                        ax.errorbar(
                            mjds[nondet_inds], mags[nondet_inds], yerr=errs[nondet_inds], fmt="o", mec="black", color=colors.get(f, "k")
                        )
                        ax.scatter(mjds[det_inds], mags[det_inds], marker="v", alpha=0.2, color=colors.get(f, "k"))
                    else:
                        ax.errorbar(mjds, mags, yerr=errs, fmt="o", mec="black", color=colors.get(f, "k"))
            ax.errorbar([], [], color=colors.get(f, "k"), label=f)
            if show:
                filtText = f + "\n"
                plt.figtext(0.95, 0.75 - (0.05 * i), filtText, fontsize=14, color=colors.get(f))

        if log_transform is False:
            ax.set_xlabel("Shifted Time [days]")
        else:
            ax.set_xlabel("Log(Shifted Time)")

        if plot_fluxes:
            ax.set_ylabel("Shifted Fluxes")
        else:
            ax.set_ylabel("Shifted Magnitudes")
            plt.gca().invert_yaxis()
        if show:
            plt.title("Lightcurves for collection of {} objects\nType:{}, Subtype:{}".format(len(sne), sn_class.type, sn_class.subtype))
            plt.show()

    def plot_gp_predict_gp(self, phases, mean_prediction, std_prediction, mags, errs, filt, use_fluxes=False):
        fig, ax = plt.subplots()
        ax.plot(sorted(phases), mean_prediction, color="k", label="GP fit", zorder=10)
        ax.errorbar(phases, mags.reshape(-1), errs.reshape(-1), fmt="o", color=colors.get(filt, "k"), alpha=0.2, label=filt, zorder=0)
        ax.fill_between(
            sorted(phases.ravel()),
            mean_prediction - 1.96 * std_prediction,
            mean_prediction + 1.96 * std_prediction,
            alpha=0.5,
            color="lightgray",
            label="96\% Confidence Interval",
            zorder=10,
        )

        plt.xlabel("Shifted Time [days]")
        if use_fluxes:
            plt.ylabel("Fluxes")
        else:
            plt.gca().invert_yaxis()
            plt.ylabel("Shifted Magnitude")
        plt.title("Single-Filter GP Fit")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def plot_construct_grid(
        self,
        gp_class,
        X,
        Y,
        Z,
        ax=None,
        Z_lower=None,
        Z_upper=None,
        grid_type=None,
        phase_grid=None,
        mag_grid=None,
        wl_grid=None,
        err_grid=None,
        filtlist=None,
    ):
        """
        :input grid_type: takes str object 'median' or 'poly', default=None
        """
        gpc = gp_class
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z)

        if Z_lower is not None:
            ax.plot_surface(X, Y, Z_lower, color='blue', alpha=0.2)
        if Z_upper is not None:
            ax.plot_surface(X, Y, Z_upper, color='blue', alpha=0.2)
        ax.set_xlabel("Phase Grid")
        ax.set_ylabel("Wavelengths [Angstroms]")

        ax.set_zlabel("Flux")

        if grid_type == "polynomial":
            ax.set_title("Polynomial Grid / Templates")
        elif grid_type == "median":
            ax.set_title("Median Grid / Templates")
        elif grid_type == "final":
            ax.set_title("Final Median GP Fit")
        else:
            ax.set_title("Grid")
        plt.tight_layout()
        # plt.show()

    def plot_subtract_data_from_grid(
        self,
        sn_class,
        phase_grid,
        mag_grid,
        wl_ind,
        filt,
        ax = None,
    ):
        sn = sn_class

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(phase_grid, mag_grid[:, wl_ind], color=colors.get(filt, "k"), label="template")

        plt.axhline(y=0, linestyle="--", color="gray")
        ax.errorbar([], [], yerr=[], marker="o", color="k", label="residuals", alpha=0.2)
        ax.errorbar(
            [],
            [],
            yerr=[],
            fmt="o",
            color=colors.get(filt, "k"),
            label="data",
            alpha=0.5,
        )
        ax.set_xlabel("Log(Time)")
        ax.set_ylabel("Flux relative to peak")
        ax.set_title("Template Subtraction for {} in {}-band".format(sn.name, filt))
        plt.legend()
        # plt.show()

    def plot_run_gp_overlay(
        self,
        ax,
        test_times,
        test_prediction,
        std_prediction,
        template_mags,
        residuals,
        log_transform,
        filt,
        sn=None,
    ):
        if sn is not None:
            # Convert between log fluxes to shifted magnitudes
            log_fluxes = test_prediction + template_mags
            shifted_peak_mag = np.log10(sn.zps[sn.info["peak_filt"]] * 1e-11 * 10 ** (-0.4 * sn.info["peak_mag"]))
            shifted_mags = -1 * ((np.log10(10**(log_fluxes + shifted_peak_mag) / (sn.zps[filt] * 1e-11)) / -0.4) - sn.info["peak_mag"])
            ax.plot(
                test_times,
                shifted_mags,
                label=filt,
                color=colors.get(filt, "k"),
            )
            ax.fill_between(
                test_times,
                shifted_mags - 1.96 * std_prediction,
                shifted_mags + 1.96 * std_prediction,
                alpha=0.2,
                color=colors.get(filt, "k"),
            )
        else:
            ax.plot(
                test_times,
                test_prediction + template_mags,
                label=filt,
                color=colors.get(filt, "k"),
            )
            ax.fill_between(
                test_times,
                test_prediction - 1.96 * std_prediction + template_mags,
                test_prediction + 1.96 * std_prediction + template_mags,
                alpha=0.2,
                color=colors.get(filt, "k"),
            )          

        ax.errorbar(
            np.exp(
                residuals["Phase"].values
            )
            - log_transform,
            residuals["Mag"].values,
            yerr=residuals["MagErr"].values,
            fmt="o",
            color=colors.get(filt, "k"),
            mec="k",
        )
        
        ax.set_xlabel("Normalized Time [days]")
        ax.set_ylabel("Flux Relative to Peak")
        if sn is not None:
            plt.title(sn.name)
        plt.legend()

    def plot_run_gp_surface(self, gp_class, x, y, test_prediction_reshaped):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, test_prediction_reshaped)
        ax.set_xlabel("Phase Grid")
        ax.set_ylabel("Wavelengths")
        ax.set_zlabel("Fluxes")
        plt.tight_layout()
        plt.show()
