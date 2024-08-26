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

from caat.utils import ROOT_DIR

warnings.filterwarnings("ignore")


class CAAT:

    def __init__(self):
        # This might need to be replaced long term with some configuration parameters/config file
        base_path = os.path.join(ROOT_DIR, "data/")
        base_db_name = "caat.csv"
        db_loc = base_path + base_db_name

        if os.path.isfile(db_loc):
            # Chech to see if db file exists
            self.caat = pd.read_csv(db_loc)
        else:
            raise Warning("No database file found")

    def get_sne_by_type(self, sntype, snsubtype=None):
        if snsubtype is not None:
            sne_list = (self.caat["Type"] == sntype) & (self.caat["Subtype"] == snsubtype)
        else:
            sne_list = self.caat["Type"] == sntype
        return self.caat[sne_list].Name.values

    @staticmethod
    def save_db_file(db_loc, sndb, force=False):
        if not force and os.path.exists(db_loc):
            print("WARNING: CAAT file with this name already exists. To overwrite, use force=True")

        else:
            sndb.to_csv(db_loc, index=False)

    @staticmethod
    def read_info_from_tns_file(tns_file, sn_names, col_names):
        ### Read values from a TNS CSV file, given a list of SN names
        ### to query and a list of the names of columns to get

        # First, sanitize the names if not in TNS format (no SN or AT)
        sn_names = [sn_name.replace("SN", "").replace("AT", "") for sn_name in sn_names]

        ### This is really gross but works for now
        ### In the future, want to make this more pandas-esque and
        ### be able to handle multiple col_names at once
        tns_df = pd.read_csv(tns_file)
        tns_values = {}

        for col_name in col_names:
            tns_values[col_name] = []
            for name in sn_names:
                row = tns_df[tns_df["name"] == name]
                if len(row.values) == 0:
                    tns_values[col_name].append(np.nan)
                else:
                    tns_value = row[col_name].values[0]
                    if not tns_value:
                        tns_values[col_name].append(np.nan)
                    else:
                        tns_values[col_name].append(tns_value)

        return tns_values

    @classmethod
    def create_db_file(cls, CAAT, type_list=None, base_db_name="caat.csv", tns_file="", force=False):
        # This might need to be replaced long term with some configuration parameters/config file
        base_path = os.path.join(ROOT_DIR, "data/")
        db_loc = base_path + base_db_name

        # Create A List Of Folders To Parse
        if type_list is None:
            type_list = ["SESNe", "SLSN-I", "SLSN-II", "SNII", "SNIIn", "FBOT", "Other"]

        sndb_name = []
        sndb_type = []
        sndb_subtype = []
        sndb_z = []
        sndb_tmax = []
        sndb_mmax = []
        sndb_filtmax = []
        sndb_ra = []
        sndb_dec = []

        # etc
        for sntype in type_list:
            """For each folder:
            get a list of subfolders
            get a list of objects in each folder
            assign SN, subtypes to list
            """
            subtypes = os.listdir(base_path + sntype + "/")
            for snsubtype in subtypes:
                sn_names = os.listdir(base_path + sntype + "/" + snsubtype + "/")

                sndb_name.extend(sn_names)
                sndb_type.extend([sntype] * len(sn_names))
                sndb_subtype.extend([snsubtype] * len(sn_names))

                if tns_file:
                    tns_info = CAAT.read_info_from_tns_file(tns_file, sn_names, ["redshift", "ra", "declination"])
                    tns_z = tns_info["redshift"]
                    tns_ra = tns_info["ra"]
                    tns_dec = tns_info["declination"]
                else:
                    tns_z = [np.nan] * len(sn_names)
                    tns_ra = [np.nan] * len(sn_names)
                    tns_dec = [np.nan] * len(sn_names)

                sndb_z.extend(tns_z)
                sndb_ra.extend(tns_ra)
                sndb_dec.extend(tns_dec)

                sndb_tmax.extend([np.nan] * len(sn_names))
                sndb_mmax.extend([np.nan] * len(sn_names))
                sndb_filtmax.extend([""] * len(sn_names))

        sndb = pd.DataFrame(
            {
                "Name": sndb_name,
                "Type": sndb_type,
                "Subtype": sndb_subtype,
                "Redshift": sndb_z,
                "RA": sndb_ra,
                "Dec": sndb_dec,
                "Tmax": sndb_tmax,
                "Magmax": sndb_mmax,
                "Filtmax": sndb_filtmax,
            }
        )

        CAAT.save_db_file(db_loc, sndb, force=force)

    @classmethod
    def combine_db_files(CAAT, file1, file2, outfile):

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        merged = df1.combine_first(df2)

        ### Want to reorder columns in nicer way
        ### Here we are assuming df2 has all the columns in the merged df
        ### and in the correct order
        merged = merged[df2.columns.tolist()]
        CAAT.save_db_file(outfile, merged)

    @property
    def db(self):
        return self.caat

    def get_list_of_sne(self, type=None, year=None):  # etc, other filter parameters - # of detections in filer/wavelength regime?
        # parse the pandas db
        raise NotImplementedError