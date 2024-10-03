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

from .CAAT import CAAT
# from .GP import GP, Fitter
# from .GP3D import GP3D
# from .Kernels import RBFKernel, WhiteKernel, MaternKernel
from .Plot import Plot
# from .SNCollection import SNCollection, SNType
from caat.utils import ROOT_DIR, colors

warnings.filterwarnings("ignore")


class SN:
    """
    A Supernova object, taking a classification (i.e. SN II, SESNe, FBOT, etc.),
    a subtype (i.e., SN IIP, SN IIb, SN Ibn, etc.), and a name (i.e. SN2022acko)
    """

    base_path = os.path.join(ROOT_DIR, "data/")

    ### All ZPs for AB mags, in 1e-11 erg/s/cm**2/A
    zps = {}
    
    # zps = {
    #     "UVW2": 2502.2,#744.84,
    #     "UVM2": 2158.3,#785.58,
    #     "UVW1": 1510.9,#940.99,
    #     "U": 847.1,#1460.59,
    #     "B": 569.7,#4088.50,
    #     "V": 362.8,#3657.87,
    #     "g": 487.6,
    #     "r": 282.9,
    #     "i": 184.9,
    #     "o": 238.9,
    #     "c": 389.3,
    # }

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

    def __init__(self, name: str = None, data: dict = None):

        if isinstance(name, str):
            self.name = name
            self.data = {}

            found = False
            for typ in os.listdir(self.base_path):
                if os.path.isdir(os.path.join(self.base_path, typ)):
                    for subtyp in os.listdir(os.path.join(self.base_path, typ)):
                        if os.path.isdir(os.path.join(self.base_path, typ, subtyp)):
                            for snname in os.listdir(os.path.join(self.base_path, typ, subtyp)):
                                if name == snname:
                                    self.classification = typ
                                    self.subtype = subtyp
                                    found = True

            if not found:
                raise Exception(f"No SN named {name} found in our archives")

            self.read_info_from_caat_file()
            self.load_shifted_data()

        if isinstance(data, dict):
            self.name = ""
            self.classification = ""
            self.subtype = ""
            self.data = data
            self.info = {}
            self.shifted_data = {}

        for filt, wl in self.wle.items():
            self.zps[filt] = (10**-23 * 3e18 / wl) * 1e11

    def __repr__(self):
        return self.name

    def write_info_to_caat_file(self, force=False):

        caat = CAAT().caat
        row = caat[caat["Name"] == self.name]

        row["Tmax"] = self.info.get("peak_mjd", np.nan)
        row["Magmax"] = self.info.get("peak_mag", np.nan)
        row["Filtmax"] = self.info.get("peak_filt", "")

        caat[caat["Name"] == self.name] = row

        ### Save back to the csv file
        CAAT().save_db_file(os.path.join(ROOT_DIR, "data/", "caat.csv"), caat, force=force)

    def read_info_from_caat_file(self):

        caat = CAAT().caat
        row = caat[caat["Name"] == self.name]
        if np.isnan(row["Tmax"].values) or np.isnan(row["Magmax"].values) or not row["Filtmax"].values:
            self.info = {}

        else:
            info_dict = {}
            info_dict["peak_mjd"] = row["Tmax"].values[0]
            info_dict["peak_mag"] = row["Magmax"].values[0]
            info_dict["peak_filt"] = row["Filtmax"].values[0]
            info_dict["searched"] = True
            info_dict["z"] = row["Redshift"].values[0]
            info_dict["ra"] = row["RA"].values[0]
            info_dict["dec"] = row["Dec"].values[0]

            self.info = info_dict

    def load_swift_data(self):
        ### Load the Swift data for this object
        if not os.path.exists(
            os.path.join(
                self.base_path,
                self.classification,
                self.subtype,
                self.name,
                self.name + "_uvotB15.1.dat",
            )
        ):
            print("No Swift file for ", self.name)
            return

        ### Magnitudes in the SOUSA output file are in Vega mags
        ### We need to convert them to AB mags
        ### From here: https://swift.gsfc.nasa.gov/analysis/uvot_digest/zeropts.html 
        ab_minus_vega = {
            'V': -0.01,
            'B': -0.13,
            'U': 1.02,
            'UVW1': 1.51,
            'UVM2': 1.69,
            'UVW2': 1.73
        }

        df = pd.read_csv(
            os.path.join(
                self.base_path,
                self.classification,
                self.subtype,
                self.name,
                self.name + "_uvotB15.1.dat",
            ),
            delim_whitespace=True,
            comment="#",
            names=[
                "Filter",
                "MJD",
                "Mag",
                "MagErr",
                "3SigMagLim",
                "0.98SatLim",
                "Rate",
                "RateErr",
                "Ap",
                "Frametime",
                "Exp",
                "Telapse",
            ],
        )

        for i, row in df.iterrows():
            if not np.isnan(row["Mag"]):
                self.data.setdefault(row["Filter"], []).append(
                    {
                        "mag": row["Mag"] + ab_minus_vega[row["Filter"]], 
                        "err": row["MagErr"], 
                        "mjd": row["MJD"]
                    }
                )
            else:
                self.data.setdefault(row["Filter"], []).append(
                    {
                        "mag": row["3SigMagLim"] + ab_minus_vega[row["Filter"]],
                        "err": 0.01,
                        "mjd": row["MJD"],
                        "nondetection": True,
                    }
                )

    def load_json_data(self):

        ### Load data saved as a JSON file (ZTF, ATLAS, OpenSN, ASASSN)
        if not os.path.exists(os.path.join(self.base_path, self.classification, self.subtype, self.name)):
            print("No additional data files for ", self.name)
            return

        dirfiles = os.listdir(os.path.join(self.base_path, self.classification, self.subtype, self.name))

        for f in dirfiles:
            ### Trying to filter out info file and shifted data file, should do this better
            if ".json" in f and "_info.json" not in f and "_shifted_data.json" not in f:
                # print('Working with ', f)
                with open(
                    os.path.join(self.base_path, self.classification, self.subtype, self.name, f),
                    "r",
                ) as jsonf:
                    d = json.load(jsonf)

                for filt, mag_list in d.items():
                    self.data.setdefault(filt, []).extend([mag for mag in mag_list if mag["err"] < 9999])
                    self.data.setdefault(filt, []).extend(
                        [mag | {"err": 0.01, "nondetection": True} for mag in mag_list if mag["err"] == 9999 and not np.isnan(mag["mag"])]
                    )

    def write_shifted_data(self):

        with open(
            os.path.join(
                self.base_path,
                self.classification,
                self.subtype,
                self.name,
                self.name + "_shifted_data.json",
            ),
            "w+",
        ) as f:
            json.dump(self.shifted_data, f, indent=4)

    def load_shifted_data(self):

        ### Load shifted data that has been saved to a file

        if not os.path.exists(
            os.path.join(
                self.base_path,
                self.classification,
                self.subtype,
                self.name,
                self.name + "_shifted_data.json",
            )
        ):
            self.shifted_data = {}

        else:

            with open(
                os.path.join(
                    self.base_path,
                    self.classification,
                    self.subtype,
                    self.name,
                    self.name + "_shifted_data.json",
                ),
                "r",
            ) as f:
                shifted_data = json.load(f)

            self.shifted_data = shifted_data

    def convert_to_fluxes(self, phasemin=-20, phasemax=50):
        """
        Converts the saved photometric magnitudes to fluxes
        Converts both shifted and unshifted data
        """

        for filt in self.data:
            if filt in self.zps.keys():

                ### For right now, let's only care about the nondetection closest to
                ### both the first and last detection
                detection_mjds = np.asarray([phot["mjd"] for phot in self.data[filt] if not phot.get("nondetection", False)])
                if len(detection_mjds) > 0:
                    min_detection = min(detection_mjds)
                    max_detection = max(detection_mjds)

                    nondetection_mjds = np.asarray([phot["mjd"] for phot in self.data[filt] if phot.get("nondetection", False)])
                    if len(nondetection_mjds) == 0:
                        min_nondetection = 9e9
                        max_nondetection = 9e9
                    else:
                        min_nondetection = nondetection_mjds[np.argmin(abs(nondetection_mjds - min_detection))]
                        max_nondetection = nondetection_mjds[np.argmin(abs(nondetection_mjds - max_detection))]

                    new_phot = []
                    for i, phot in enumerate(self.data[filt]):
                        if phot.get("nondetection", False):
                            ### Check if this is the closest nondetection to either
                            ### the first or last detection in this filter
                            if abs(phot["mjd"] - min_nondetection) < 0.5 or abs(phot["mjd"] - max_nondetection) < 0.5:
                                phot["flux"] = np.log10(self.zps[filt] * 1e-11 * 10 ** (-0.4 * phot["mag"]))  # * 1e15
                                phot["fluxerr"] = phot["err"]  # 1.086 * phot['err'] * phot['flux']
                                new_phot.append(phot)
                        else:
                            phot["flux"] = np.log10(self.zps[filt] * 1e-11 * 10 ** (-0.4 * phot["mag"]))  # * 1e15
                            phot["fluxerr"] = phot["err"]  # 1.086 * phot['err'] * phot['flux']
                            new_phot.append(phot)

                    self.data[filt] = new_phot

                else:
                    self.data[filt] = []

        if self.shifted_data:
            ### Get the flux at peak, subtract it from the other fluxes
            for filt in self.shifted_data:
                if filt in self.zps.keys():

                    ### For right now, let's only care about the nondetection closest to
                    ### both the first and last detection

                    ### Here we can be a bit more careful and only pick the max/min detection within
                    ### a certain window around the time of peak, to avoid picking i.e. a max detection
                    ### that was spurious and occured a year after the SN
                    detection_mjds = np.asarray(
                        [
                            phot["mjd"]
                            for phot in self.shifted_data[filt]
                            if not phot.get("nondetection", False) and phot["mjd"] > phasemin and phot["mjd"] < phasemax
                        ]
                    )

                    detection_mags = np.asarray(
                        [
                            phot["mag"]
                            for phot in self.shifted_data[filt]
                            if not phot.get("nondetection", False) and phot["mjd"] > phasemin and phot["mjd"] < phasemax
                        ]
                    )
                    if len(detection_mjds) > 0:
                        min_detection = min(detection_mjds)
                        max_detection = max(detection_mjds)
                        min_detection_mag = detection_mags[np.where((detection_mjds==min_detection))[0]][0]
                        max_detection_mag = detection_mags[np.where((detection_mjds==max_detection))[0]][0]

                        # nondetection_mjds = np.asarray([phot["mjd"] for phot in self.shifted_data[filt] if phot.get("nondetection", False)])
                        # if len(nondetection_mjds) == 0:
                        #     min_nondetection = 9e9
                        #     max_nondetection = 9e9
                        # else:
                        #     min_nondetection = nondetection_mjds[np.argmin(abs(nondetection_mjds - min_detection))]
                        #     max_nondetection = nondetection_mjds[np.argmin(abs(nondetection_mjds - max_detection))]

                        new_phot = []

                        for i, phot in enumerate(self.shifted_data[filt]):
                            if phot.get("nondetection", False):
                                ### Check if this nondetection is close to either
                                ### the first or last nondetection in this filter
                                # #if (phot["mjd"] - min_nondetection < 0.0 and phot["mjd"] - min_nondetection > 0.5) or (phot["mjd"] - max_nondetection < 0.5 and phot["mjd"] - max_nondetection > 0.0):
                                # if (phot["mjd"] - min_nondetection < 0.0) or (phot["mjd"] - max_nondetection > 0.0):
                                #     unshifted_mag = phot["mag"] + self.info["peak_mag"]
                                #     shifted_flux = np.log10(self.zps[filt] * 1e-11 * 10 ** (-0.4 * unshifted_mag)) - np.log10(
                                #         self.zps[self.info["peak_filt"]] * 1e-11 * 10 ** (-0.4 * self.info["peak_mag"])
                                #     )  # * 1e15
                                #     phot["flux"] = shifted_flux
                                #     phot["fluxerr"] = phot["err"]
                                #     new_phot.append(phot)

                                if (phot["mjd"] - min_detection < 0.0 and phot["mag"] > min_detection_mag) or (phot["mjd"] - max_detection > 0.0 and phot["mag"] > max_detection_mag):
                                    #print(phot["mjd"], min_detection, phot['mag'], min_detection_mag, max_detection, max_detection_mag)
                                    unshifted_mag = phot["mag"] + self.info["peak_mag"]
                                    shifted_flux = np.log10(self.zps[filt] * 1e-11 * 10 ** (-0.4 * unshifted_mag)) - np.log10(
                                        self.zps[self.info["peak_filt"]] * 1e-11 * 10 ** (-0.4 * self.info["peak_mag"])
                                    )  # * 1e15
                                    phot["flux"] = shifted_flux
                                    phot["fluxerr"] = phot["err"]
                                    new_phot.append(phot)
                                
                                # else:
                                #     unshifted_mag = phot["mag"] + self.info["peak_mag"]
                                #     shifted_flux = np.log10(self.zps[filt] * 1e-11 * 10 ** (-0.4 * unshifted_mag)) - np.log10(
                                #         self.zps[self.info["peak_filt"]] * 1e-11 * 10 ** (-0.4 * self.info["peak_mag"])
                                #     )  # * 1e15
                                #     phot["flux"] = shifted_flux
                                #     phot["fluxerr"] = phot["err"] * 10
                                #     new_phot.append(phot)
                            else:
                                unshifted_mag = phot["mag"] + self.info["peak_mag"]
                                shifted_flux = np.log10(self.zps[filt] * 1e-11 * 10 ** (-0.4 * unshifted_mag)) - np.log10(
                                    self.zps[self.info["peak_filt"]] * 1e-11 * 10 ** (-0.4 * self.info["peak_mag"])
                                )  # * 1e15
                                phot["flux"] = shifted_flux
                                phot["fluxerr"] = phot["err"]
                                new_phot.append(phot)

                        self.shifted_data[filt] = new_phot

                    else:
                        self.shifted_data[filt] = []

    def correct_for_galactic_extinction(self):
        """
        Uses the coordinates of the SN from the CAAT file
        to find and correct for MW extinction
        NOTE: Must be run before convert_to_fluxes() is ran
        """
        sfd = SFDQuery()

        if not self.info.get("ra", "") or not self.info.get("dec", ""):
            print("No coordinates for this object")
            return

        coord = SkyCoord(ra=self.info["ra"] * u.deg, dec=self.info["dec"] * u.deg)
        exts = fm(
            np.asarray([self.wle[filt] * (1 + self.info.get("z", 0)) for filt in self.data.keys() if filt in self.wle.keys()]),
            sfd(coord),
        )

        i = 0
        for filt in self.data.keys():
            if filt in self.wle.keys():

                new_phot = []
                for phot in self.data[filt]:
                    if not phot.get("ext_corrected", False):
                        phot["mag"] -= exts[i]
                        phot["ext_corrected"] = True
                    new_phot.append(phot)

                self.data[filt] = new_phot
                i += 1

            else:
                self.data[filt] = []

        if self.shifted_data:
            i = 0
            for filt in self.shifted_data:
                if filt in self.wle.keys():

                    new_phot = []
                    for phot in self.shifted_data[filt]:
                        if not phot.get("ext_corrected", False):
                            phot["mag"] -= exts[i]
                            phot["ext_corrected"] = True
                        new_phot.append(phot)

                    self.shifted_data[filt] = new_phot
                    i += 1

                else:
                    self.shifted_data[filt] = []

    def plot_data(
        self,
        filts_to_plot=["all"],
        shifted_data_exists=False,
        view_shifted_data=False,
        offset=0,
        plot_fluxes=False,
    ):
        
        if filts_to_plot[0] == "all":  # if individual filters not specified, plot all by default
            filts_to_plot = colors.keys()
        
        if not self.data:  # check if data/SN has not been previously read in/initialized
            self.load_swift_data()
            self.load_json_data()

        if shifted_data_exists:
            if plot_fluxes:
                self.convert_to_fluxes()
            data_to_plot = self.shifted_data
        elif view_shifted_data:
            for f in filts_to_plot:
                self.shift_to_max(f, offset=offset)
            if plot_fluxes:
                self.convert_to_fluxes()
            data_to_plot = self.shifted_data
        else:
            if plot_fluxes:
                self.convert_to_fluxes()
            data_to_plot = self.data

        Plot().plot_sn_data(
            sn_class=self,
            data_to_plot=data_to_plot,
            filts_to_plot=filts_to_plot,
            plot_fluxes=plot_fluxes,
        )

    def fit_for_max(self, filt, shift_array=[-3, -2, -1, 0, 1, 2, 3], plot=False, offset=0):
        """
        Takes as input arrays for MJD, mag, and err for a filter
        as well as the guess for the MJD of maximum and an array
        to shift the lightcurve over,
        and returns estimates of the peak MJD and mag at peak
        """
        mjd_array = np.asarray([phot["mjd"] for phot in self.data[filt] if not phot.get("nondetection", False)])
        mag_array = np.asarray([phot["mag"] for phot in self.data[filt] if not phot.get("nondetection", False)])
        err_array = np.asarray([phot["err"] for phot in self.data[filt] if not phot.get("nondetection", False)])

        if len(mag_array) < 4:  # == 0:
            return None, None

        initial_guess_mjd_max = mjd_array[np.where((mag_array == min(mag_array)))[0]][0] + offset

        fit_inds = np.where((abs(mjd_array - initial_guess_mjd_max) < 30))[0]
        if len(fit_inds) < 4:

            return None, None

        fit_coeffs = np.polyfit(mjd_array[fit_inds], mag_array[fit_inds], 3)
        guess_phases = np.arange(min(mjd_array[fit_inds]), max(mjd_array[fit_inds]), 1)
        p = np.poly1d(fit_coeffs)
        guess_best_fit = p(guess_phases)

        if len(guess_best_fit) == 0:
            return None, None

        guess_mjd_max = guess_phases[np.where((guess_best_fit == min(guess_best_fit)))[0]][0]

        ### Do this because the array might not be ordered
        inds_to_fit = np.where((mjd_array > guess_mjd_max - 10) & (mjd_array < guess_mjd_max + 10))
        if len(inds_to_fit[0]) < 4:
            # print('Select a wider date range')
            return None, None

        numdata = len(mjd_array[inds_to_fit])
        numiter = max(int(numdata * np.log(numdata) ** 2), 200)

        fit_mjds = mjd_array[inds_to_fit]
        fit_mags = mag_array[inds_to_fit]
        fit_errs = err_array[inds_to_fit]

        if plot:
            Plot().plot_fit_for_max(
                sn_class=self,
                mjd_array=mjd_array,
                mag_array=mag_array,
                err_array=err_array,
                fit_mjds=fit_mjds,
                fit_mags=fit_mags,
                fit_errs=fit_errs,
                inds_to_fit=inds_to_fit,
            )

        peak_mags = []
        peak_mjds = []
        for num in range(numiter):
            simulated_points = []

            ### Shift by a certain number of days to randomly sample the light curve
            sim_shift = np.random.choice(shift_array)

            inds_to_fit = np.where((mjd_array > guess_mjd_max - 5 + sim_shift) & (mjd_array < guess_mjd_max + 5 + sim_shift))[0]
            if len(inds_to_fit) > 0:

                fit_mjds = mjd_array[inds_to_fit]
                fit_mags = mag_array[inds_to_fit]
                fit_errs = err_array[inds_to_fit]

                for i in range(len(fit_mjds)):
                    simulated_points.append(np.random.normal(fit_mags[i], fit_errs[i]))

                fit = np.polyfit(fit_mjds, simulated_points, 2)
                f = np.poly1d(fit)
                fit_time = np.linspace(min(fit_mjds), max(fit_mjds), 100)

                if num % 25 == 0 and plot:
                    plt.plot(fit_time, f(fit_time), color="black", linewidth=0.5)
                peak_mag = min(f(fit_time))
                peak_mags.append(peak_mag)
                peak_mjds.append(fit_time[np.argmin(f(fit_time))])

        if len(peak_mjds) == 0:
            return None, None

        if plot:
            plt.errorbar(
                mean(peak_mjds),
                mean(peak_mags),
                xerr=stdev(peak_mjds),
                yerr=stdev(peak_mags),
                color="red",
                fmt="o",
                label="Best Fit Peak",
            )
            plt.xlim(guess_mjd_max - 10, guess_mjd_max + 10)

        self.info["peak_mjd"] = mean(peak_mjds)
        self.info["peak_mag"] = mean(peak_mags)
        self.info["peak_filt"] = filt
        self.info["searched"] = True

    def shift_to_max(
        self,
        filt,
        shift_array=[-3, -2, -1, 0, 1, 2, 3],
        plot=False,
        offset=0,
        shift_fluxes=False,
        try_other_filts=True
    ):

        if not self.data:
            self.load_swift_data()
            self.load_json_data()

        if filt not in self.data.keys():
            return [], [], [], []

        if not self.info.get("peak_mjd") and not self.info.get("peak_mag"):
            self.fit_for_max(filt, shift_array=shift_array, plot=plot, offset=offset)

            if not self.info.get("peak_mjd", 0) > 0 and try_other_filts:
                for newfilt in ["V", "g", "c", "B", "r", "o", "U", "i", "UVW1"]:
                    if newfilt in self.data.keys() and newfilt != filt:
                        self.fit_for_max(newfilt, shift_array=shift_array, plot=plot, offset=offset)

                        if self.info.get("peak_mjd", 0) > 0:
                            break

                if newfilt == "UVW1" and not self.info.get("peak_mjd", 0) > 0:
                    print("Reached last filter and could not fit for peak for ", self.name)
                    self.info["searched"] = True

        if not self.info.get("peak_mag", 0) > 0:
            return [], [], [], []

        mjds = np.asarray([phot["mjd"] for phot in self.data[filt]]) - self.info["peak_mjd"]
        mags = np.asarray([phot["mag"] for phot in self.data[filt]]) - self.info["peak_mag"]
        errs = np.asarray([phot["err"] for phot in self.data[filt]])
        nondets = np.asarray([phot.get("nondetection", False) for phot in self.data[filt]])

        if plot:
            Plot().plot_shift_to_max(sn_class=self, mjds=mjds, mags=mags, errs=errs, nondets=nondets, filt=filt)

        self.shifted_data.setdefault(filt, []).extend(
            [
                {
                    "mjd": mjds[i],
                    "mag": mags[i],
                    "err": errs[i],
                    "nondetection": nondets[i],
                }
                for i in range(len(mjds))
            ]
        )

        if shift_fluxes:
            self.convert_to_fluxes()
            shifted_mjd = np.asarray([phot["mjd"] for phot in self.shifted_data[filt]])
            shifted_flux = np.asarray([phot["flux"] for phot in self.shifted_data[filt]])
            shifted_err = np.asarray([phot["fluxerr"] for phot in self.shifted_data[filt]])
            nondets = np.asarray([phot.get("nondetection", False) for phot in self.shifted_data[filt]])

            return shifted_mjd, shifted_flux, shifted_err, nondets

        return mjds, mags, errs, nondets

    def interactively_fit_for_max(
        self,
        filt="",
        shift_array=[-3, -2, -1, 0, 1, 2, 3],
        plot=True,
        offset=0,
        save_to_caat=False,
        force=False,
    ):

        self.load_json_data()
        self.load_swift_data()
        self.shifted_data = {}

        if not filt:
            self.plot_data()
            print("Data in filters {}\n".format(list(self.data.keys())))

            filt = input('Which filter would you like to use to fit for max? To skip, type "skip"\n')
            if filt == "skip":
                return

        mjds, _, _, _ = self.shift_to_max(filt, shift_array=shift_array, plot=plot, offset=offset)

        if len(mjds) == 0:
            refit = input("No photometry found for this filter. Try to refit? y/n \n")

        else:
            self.plot_data(view_shifted_data=True)
            refit = input("Refit the data with new filter or offset? y/n \n")

        if refit == "n" and save_to_caat:
            self.write_info_to_caat_file(force=force)

        elif refit == "n" and not save_to_caat:
            print('To save these parameters, rerun with "save_to_caat=True"')

        elif refit == "y":
            self.info = {}
            newfilt = input("Try fitting a new filter? If so, enter the filter here. If not, leave blank to pick new offset\n")

            if newfilt:
                self.interactively_fit_for_max(
                    newfilt,
                    shift_array=shift_array,
                    plot=plot,
                    offset=offset,
                    save_to_caat=save_to_caat,
                    force=force,
                )

            else:
                newoffset = input("Enter new offset here\n")

                if newoffset:
                    self.interactively_fit_for_max(
                        filt,
                        shift_array=shift_array,
                        plot=plot,
                        offset=float(newoffset),
                        save_to_caat=save_to_caat,
                        force=force,
                    )

    def log_transform_time(self, phases, phase_start=30):

        return np.log(phases + phase_start)