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

warnings.filterwarnings("ignore")


class RBFKernel:  # pylint: disable=too-few-public-methods
    """
    An RBFKernel object, to be used in GP fitting
    Allows users to define Kernel parameters such as length scale, bounds, etc.
    """

    def __init__(self, length_scale, length_scale_bounds):

        self.kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)


class WhiteNoiseKernel:  # pylint: disable=too-few-public-methods
    """
    An RBFKernel object, to be used in GP fitting
    Allows users to define Kernel parameters such as noise level, bounds, etc.
    """

    def __init__(self, noise_level, noise_level_bounds):

        self.kernel = WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)


class MaternKernel:  # pylint: disable=too-few-public-methods
    """
    A MaternKernel, to be used in GP fitting
    Allows users to define Kernel parameters such as length scale, bounds, etc.
    """

    def __init__(self, length_scale, length_scale_bounds, nu):

        self.kernel = Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)