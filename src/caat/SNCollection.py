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
from .SN import SN

warnings.filterwarnings("ignore")


class SNCollection:
    """
    A SNCollection object, which holds an arbitrary number of SNe

    """

    base_path = "../data/"

    def __init__(
        self,
        names: Union[str, None] = None,
        sntype: Union[str, None] = None,
        snsubtype: Union[str, None] = None,
        SNe: Union[list[SN], None] = None,
        **kwargs,
    ):

        self.subtypes = list(kwargs.keys())

        if isinstance(SNe, SN):
            self.sne = SNe
        else:
            if isinstance(names, list):
                self.sne = [SN(name) for name in names]
            else:
                if type(sntype) is not None:
                    # convert this to a logger statement
                    print(f"Loading SN Type: {sntype}, Subtype: {snsubtype}")
                    caat = CAAT()
                    type_list = caat.get_sne_by_type(sntype, snsubtype)
                    print(type_list)
                    self.sne = [SN(name) for name in type_list]
                    self.type = sntype
                    self.subtype = snsubtype

    def __repr__(self):
        print("Collection of SN Objects")
        return self.sne

    # @property
    # def sne(self):
    #    return self.sne

    def get_type_list(self):
        # Maybe this lives in a separate class that handles the csv db file
        raise NotImplementedError

    def plot_all_lcs(self, filts=["all"], log_transform=False, plot_fluxes=False):
        """plot all light curves of given subtype/collection
        can plot single, multiple or all bands"""
        Plot().plot_all_lcs(sn_class=self, filts=filts, log_transform=log_transform, plot_fluxes=plot_fluxes)


class SNType(SNCollection):
    """
    A Type object, building a collection of all SNe of a given type (classification)
    """

    subtypes = []
    sne = []

    def __init__(self, type):

        self.type = type

        self.get_subtypes()
        self.build_object_list()

    def get_subtypes(self):

        caat = CAAT()
        subtype_list = caat.caat["Type"] == self.type
        self.subtypes = list(set(caat.caat[subtype_list].Subtype.values))

    def build_object_list(self):

        caat = CAAT()
        type_list = caat.get_sne_by_type(self.type)
        self.sne = [SN(name) for name in type_list]