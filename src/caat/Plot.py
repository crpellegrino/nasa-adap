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
        sn = sn_class

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
        sn = sn_class

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
        plt.title(sn.name)
        plt.legend()
        plt.gca().invert_yaxis()

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
        plt.legend()
        plt.show()

    def plot_all_lcs(self, sn_class, filts=["all"], log_transform=False, plot_fluxes=False):
        """plot all light curves of given subtype/collection
        can plot single, multiple or all bands"""
        sne = sn_class.sne
        logger.info(f"Plotting all {len(sne)} lightcurves in the collection")

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
                            mjds[nondet_inds], mags[nondet_inds], yerr=errs[nondet_inds], fmt="o", mec="black", color=colors.get(f, "k"), label=f
                        )
                        ax.scatter(mjds[det_inds], mags[det_inds], marker="v", alpha=0.2, color=colors.get(f, "k"))
                    else:
                        ax.errorbar(mjds, mags, yerr=errs, fmt="o", mec="black", color=colors.get(f, "k"), label=f)
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
        plt.title("Lightcurves for collection of {} objects\nType:{}, Subtype:{}".format(len(sne), sn_class.type, sn_class.subtype))
        plt.show()

    def plot_gp_predict_gp(self, gp_class, phases, mean_prediction, std_prediction, mags, errs, filt, use_fluxes=False):
        gp = gp_class

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
        Z_lower=None,
        Z_upper=None,
        grid_type=None,
        phase_grid=None,
        mag_grid=None,
        wl_grid=None,
        err_grid=None,
        log_transform=None,
        filtlist=None,
        use_fluxes=False,
    ):
        """
        :input grid_type: takes str object 'median' or 'poly', default=None
        """
        gpc = gp_class

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(X, Y, Z)

        if Z_lower is not None:
            ax.plot_surface(X, Y, Z_lower, color='blue', alpha=0.2)
        if Z_upper is not None:
            ax.plot_surface(X, Y, Z_upper, color='blue', alpha=0.2)
        ax.set_xlabel("Phase Grid")
        ax.set_ylabel("Wavelengths [Angstroms]")

        if use_fluxes:
            ax.set_zlabel("Flux")
        else:
            ax.invert_zaxis()
            ax.set_zlabel("Magnitude")

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

        if grid_type == "median":
            for filt in filtlist:
                if log_transform is not False:
                    wl_inds = np.where((abs(10**wl_grid - gpc.wle[filt]) <= 100))[0]
                else:
                    wl_inds = np.where((abs(wl_grid - gpc.wle[filt]) <= 100))[0]

                plt.errorbar(
                    phase_grid,
                    mag_grid[:, wl_inds[0]],
                    yerr=abs(err_grid[:, wl_inds[0]]),
                    fmt="o",
                )
                if not use_fluxes:
                    plt.gca().invert_yaxis()
                plt.title(filt)
                # plt.show()

        plt.show()

    def plot_subtract_data_from_grid(
        self,
        gp_class,
        sn_class,
        phase_grid,
        mag_grid,
        wl_ind,
        filt,
        log_transform,
        use_fluxes=False,
    ):
        gpc = gp_class
        sn = sn_class

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
        if log_transform is not None:
            ax.set_xlabel("Log(Time)")
        else:
            ax.set_xlabel("Time [days]")
        if use_fluxes:
            ax.set_ylabel("Flux relative to peak")
        else:
            plt.gca().invert_yaxis()
            ax.set_ylabel("Magnitude relative to Peak Mag")
        ax.set_title("Template Subtraction for {} in {}-band".format(sn.name, filt))
        plt.legend()
        # plt.show()

    def plot_run_gp_overlay(
        self,
        ax,
        sn_class,
        test_times,
        test_prediction,
        std_prediction,
        template_mags,
        residuals,
        phase_residuals,
        wl_residuals,
        err_residuals,
        inds_to_fit,
        log_transform,
        filt,
        use_fluxes,
        key_to_plot,
    ):
        sn = sn_class

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

        if log_transform is not False:
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

            ax.scatter(
                np.exp(
                    residuals["Phase"].values
                )
                - log_transform,
                residuals["Mag"].values,
                marker="v",
                color=colors.get(filt, "k"),
                alpha=0.5,
            )

        else:
            ax.errorbar(
                phase_residuals[wl_residuals == self.wle[filt] * (1 + sn.info.get("z", 0))],
                np.asarray([p[key_to_plot] for p in sn.shifted_data[filt]])[inds_to_fit],
                yerr=err_residuals[wl_residuals == self.wle[filt] * (1 + sn.info.get("z", 0))],
                fmt="o",
                color=colors.get(filt, "k"),
                mec="k",
            )

        if not use_fluxes:
            ax.invert_yaxis()
            ax.set_ylabel("Magnitude Relative to Peak")
        else:
            ax.set_ylabel("Flux Relative to Peak")

        ax.set_xlabel("Normalized Time [days]")
        plt.title(sn.name)
        plt.legend()

    def plot_run_gp_surface(self, gp_class, x, y, test_prediction_reshaped, title, use_fluxes=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, test_prediction_reshaped)
        ax.set_xlabel("Phase Grid")
        ax.set_ylabel("Wavelengths")
        if use_fluxes:
            ax.set_zlabel("Fluxes")
        else:
            ax.invert_zaxis()
            ax.set_zlabel("Magnitude")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_predict_gp(self, fig, ax, gp_class, test_prediction, std_prediction, log_transform, filt, use_fluxes=False):
        gpc = gp_class

        if log_transform is not False:
            test_times = np.exp(test_times) - log_transform
        ax.plot(test_times, test_prediction, label=filt)
        ax.fill_between(
            test_times,
            test_prediction - 1.96 * std_prediction,
            test_prediction + 1.96 * std_prediction,
            alpha=0.2,
        )

        if use_fluxes:
            ax.set_ylabel("Flux relative to peak")
        else:
            ax.invert_yaxis()
            ax.set_ylabel("Normalized Magnitude")
        ax.set_xlabel("Normalized Time [days]")
        plt.title("3D GP Fit")
        plt.legend()
        plt.show()