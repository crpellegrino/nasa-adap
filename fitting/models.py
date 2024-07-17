import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.model_selection import train_test_split
from statistics import mean, stdev
import warnings
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from extinction import fm07 as fm
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.sfd import SFDQuery

from typing import Union, Optional


warnings.filterwarnings('ignore')


colors = {'U': 'purple',
    'B': 'blue',
    'V': 'lime',
    'g': 'cyan',
    'r': 'orange',
    'i': 'red',
    'UVW2': '#FE0683',
    'UVM2': '#BF01BC',
    'UVW1': '#8B06FF',
    'c': 'turquoise',
    'o': 'salmon',
}

class CAAT:

    def __init__(self):
        #This might need to be replaced long term with some configuration parameters/config file    
        base_path = '../data/'
        base_db_name = 'caat.csv'
        db_loc = base_path + base_db_name

        if(os.path.isfile(db_loc)):
            #Chech to see if db file exists 
            self.caat=pd.read_csv(db_loc)
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
            print('WARNING: CAAT file with this name already exists. To overwrite, use force=True')
        
        else:
            sndb.to_csv(db_loc, index=False)


    @staticmethod
    def read_info_from_tns_file(tns_file, sn_names, col_names):
        ### Read values from a TNS CSV file, given a list of SN names 
        ### to query and a list of the names of columns to get

        #First, sanitize the names if not in TNS format (no SN or AT)
        sn_names = [sn_name.replace('SN', '').replace('AT', '') for sn_name in sn_names]

        ### This is really gross but works for now
        ### In the future, want to make this more pandas-esque and
        ### be able to handle multiple col_names at once
        tns_df = pd.read_csv(tns_file)
        tns_values = {}

        for col_name in col_names:
            tns_values[col_name] = []
            for name in sn_names:
                row = tns_df[tns_df['name']==name]
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
    def create_db_file(CAAT, type_list = None, base_db_name = 'caat.csv', tns_file = '', force = False):
        #This might need to be replaced long term with some configuration parameters/config file    
        base_path = '../data/'
        db_loc = base_path + base_db_name

        #Create A List Of Folders To Parse
        if type_list is None:
            type_list = ["SESNe", "SLSN-I", "SLSN-II", "SNII", "SNIIn", "FBOT"]

        sndb_name = []
        sndb_type = []
        sndb_subtype = []
        sndb_z = []
        sndb_tmax = []
        sndb_mmax = []
        sndb_filtmax = []
        sndb_ra = []
        sndb_dec = []

        #etc
        for sntype in type_list:
            """ For each folder:
            get a list of subfolders
            get a list of objects in each folder
            assign SN, subtypes to list
            """
            subtypes = os.listdir(base_path+sntype+"/")
            for snsubtype in subtypes:
                sn_names = os.listdir(base_path+sntype+"/"+snsubtype+"/")

                sndb_name.extend(sn_names)
                sndb_type.extend([sntype] * len(sn_names))
                sndb_subtype.extend([snsubtype] * len(sn_names))

                if tns_file:
                    tns_info = CAAT.read_info_from_tns_file(tns_file, sn_names, ['redshift', 'ra', 'declination'])
                    tns_z = tns_info['redshift']
                    tns_ra = tns_info['ra']
                    tns_dec = tns_info['declination']
                else:
                    tns_z = [np.nan] * len(sn_names)
                    tns_ra = [np.nan] * len(sn_names)
                    tns_dec = [np.nan] * len(sn_names)
                
                sndb_z.extend(tns_z)
                sndb_ra.extend(tns_ra)
                sndb_dec.extend(tns_dec)

                sndb_tmax.extend([np.nan] * len(sn_names))
                sndb_mmax.extend([np.nan]*len(sn_names))
                sndb_filtmax.extend([''] * len(sn_names))

        sndb = pd.DataFrame({"Name": sndb_name, "Type": sndb_type, "Subtype": sndb_subtype,
                             "Redshift": sndb_z, "RA": sndb_ra, "Dec": sndb_dec,
                             "Tmax": sndb_tmax, "Magmax": sndb_mmax, "Filtmax": sndb_filtmax})
        
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

    def get_list_of_SNe(self, Type = None,
                            Year = None):#etc, other filter parameters - # of detections in filer/wavelength regime?
        #parse the pandas db
        raise NotImplementedError


class SN:
    """
    A Supernova object, taking a classification (i.e. SN II, SESNe, FBOT, etc.),
    a subtype (i.e., SN IIP, SN IIb, SN Ibn, etc.), and a name (i.e. SN2022acko)
    """

    base_path = '../data/'

    ### All ZPs taken from SVO, in Jy and in the respective magnitude systems for those filts
    zps = {'UVW2': 744.84, 'UVM2': 785.58, 'UVW1': 940.99, 'U': 1460.59, 'B': 4088.50,
           'V': 3657.87, 'g': 487.6, 'r': 282.9, 'i': 184.9, 'o': 238.9, 'c': 389.3}

    wle = {'u': 3560,  'g': 4830, 'r': 6260, 'i': 7670, 'z': 8890, 'y': 9600, 'w':5985, 'Y': 9600,
           'U': 3600,  'B': 4380, 'V': 5450, 'R': 6410, 'G': 6730, 'E': 6730, 'I': 7980, 'J': 12200, 'H': 16300,
           'K': 21900, 'UVW2': 2030, 'UVM2': 2231, 'UVW1': 2634, 'F': 1516, 'N': 2267, 'o': 6790, 'c': 5330,
           'W': 33526, 'Q': 46028
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
                                    self.subtype  = subtyp
                                    found = True

            if not found:
                raise Exception(f'No SN named {name} found in our archives')
            
            self.read_info_from_caat_file()
            self.load_shifted_data()

        if isinstance(data, dict):
            self.name = ''
            self.classification = ''
            self.subtype = ''
            self.data = data
            self.info = {}
            self.shifted_data = {}


    def __repr__(self):
        return self.name


    def write_info_to_caat_file(self, force=False):

        Caat = CAAT()
        caat = Caat.caat
        row = caat[caat["Name"] == self.name]

        row['Tmax'] = self.info.get('peak_mjd', np.nan)
        row['Magmax'] = self.info.get('peak_mag', np.nan)
        row['Filtmax'] = self.info.get('peak_filt', '')
        
        caat[caat["Name"] == self.name] = row

        ### Save back to the csv file
        Caat.save_db_file('../data/caat.csv', caat, force=force)


    def read_info_from_caat_file(self):

        caat = CAAT().caat
        row = caat[caat["Name"] == self.name]
        if np.isnan(row['Tmax'].values) or np.isnan(row['Magmax'].values) or not row['Filtmax'].values:
            self.info = {}

        else:
            info_dict = {}
            info_dict['peak_mjd'] = row['Tmax'].values[0]
            info_dict['peak_mag'] = row['Magmax'].values[0]
            info_dict['peak_filt'] = row['Filtmax'].values[0]
            info_dict['searched'] = True
            info_dict['z'] = row['Redshift'].values[0]
            info_dict['ra'] = row['RA'].values[0]
            info_dict['dec'] = row['Dec'].values[0]

            self.info = info_dict

    
    def load_swift_data(self):
        ### Load the Swift data for this object
        if not os.path.exists(os.path.join(self.base_path, self.classification, self.subtype, self.name, self.name+'_uvotB15.1.dat')):
            print('No Swift file for ', self.name)
            return

        df = pd.read_csv(os.path.join(self.base_path, self.classification, self.subtype, self.name, self.name+'_uvotB15.1.dat'), delim_whitespace=True, comment='#', names=['Filter', 'MJD', 'Mag', 'MagErr', '3SigMagLim', '0.98SatLim', 'Rate', 'RateErr', 'Ap', 'Frametime', 'Exp', 'Telapse'])

        for i, row in df.iterrows():
            if not np.isnan(row['Mag']):
                self.data.setdefault(row['Filter'], []).append({'mag': row['Mag'], 'err': row['MagErr'], 'mjd': row['MJD']})
            else:
                self.data.setdefault(row['Filter'], []).append({'mag': row['3SigMagLim'], 'err': 0.01, 'mjd': row['MJD'], 'nondetection': True})


    def load_json_data(self):

        ### Load data saved as a JSON file (ZTF, ATLAS, OpenSN, ASASSN)
        if not os.path.exists(os.path.join(self.base_path, self.classification, self.subtype, self.name)):
            print('No additional data files for ', self.name)
            return 

        dirfiles = os.listdir(os.path.join(self.base_path, self.classification, self.subtype, self.name))

        for f in dirfiles:
            ### Trying to filter out info file and shifted data file, should do this better
            if '.json' in f and '_info.json' not in f and '_shifted_data.json' not in f: 
                # print('Working with ', f)
                with open(os.path.join(self.base_path, self.classification, self.subtype, self.name, f), 'r') as jsonf:
                    d = json.load(jsonf)

                for filt, mag_list in d.items():
                    self.data.setdefault(filt, []).extend([mag for mag in mag_list if mag['err'] < 9999])
                    self.data.setdefault(filt, []).extend([mag | {'err': 0.01, 'nondetection': True} for mag in mag_list if mag['err'] == 9999 and not np.isnan(mag['mag'])])


    def write_shifted_data(self):

        with open(os.path.join(self.base_path, self.classification, self.subtype, self.name, self.name+'_shifted_data.json'), 'w+') as f:
            json.dump(self.shifted_data, f, indent=4)
    

    def load_shifted_data(self):

        ### Load shifted data that has been saved to a file

        if not os.path.exists(os.path.join(self.base_path, self.classification, self.subtype, self.name, self.name+'_shifted_data.json')):
            self.shifted_data = {}

        else:

            with open(os.path.join(self.base_path, self.classification, self.subtype, self.name, self.name+'_shifted_data.json'), 'r') as f:
                shifted_data = json.load(f)

            self.shifted_data = shifted_data


    def convert_to_fluxes(self):
        """
        Converts the saved photometric magnitudes to fluxes
        Converts both shifted and unshifted data
        """

        for filt in self.data:
            if filt in self.zps.keys():

                ### For right now, let's only care about the nondetection closest to
                ### both the first and last detection
                detection_mjds = np.asarray([phot['mjd'] for phot in self.data[filt] if not phot.get('nondetection', False)])
                if len(detection_mjds) > 0:
                    min_detection = min(detection_mjds)
                    max_detection = max(detection_mjds)

                    nondetection_mjds = np.asarray([phot['mjd'] for phot in self.data[filt] if phot.get('nondetection', False)])
                    if len(nondetection_mjds) == 0:
                        min_nondetection = 9e9
                        max_nondetection = 9e9
                    else:
                        min_nondetection = nondetection_mjds[np.argmin(abs(nondetection_mjds - min_detection))]
                        max_nondetection = nondetection_mjds[np.argmin(abs(nondetection_mjds - max_detection))]

                    new_phot = []
                    for i, phot in enumerate(self.data[filt]):
                        if phot.get('nondetection', False):
                            ### Check if this is the closest nondetection to either 
                            ### the first or last detection in this filter
                            if abs(phot['mjd'] - min_nondetection) < 0.5 or abs(phot['mjd'] - max_nondetection) < 0.5:
                                phot['flux'] = np.log10(self.zps[filt] * 1e-11 * 10**(-0.4*phot['mag']))# * 1e15
                                phot['fluxerr'] = phot['err']#1.086 * phot['err'] * phot['flux']
                                new_phot.append(phot)
                        else:
                            phot['flux'] = np.log10(self.zps[filt] * 1e-11 * 10**(-0.4*phot['mag']))# * 1e15
                            phot['fluxerr'] = phot['err']#1.086 * phot['err'] * phot['flux']
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
                    detection_mjds = np.asarray([phot['mjd'] for phot in self.shifted_data[filt] if not phot.get('nondetection', False) and phot['mjd'] > -20 and phot['mjd'] < 50])
                    if len(detection_mjds) > 0:
                        min_detection = min(detection_mjds)
                        max_detection = max(detection_mjds)

                        nondetection_mjds = np.asarray([phot['mjd'] for phot in self.shifted_data[filt] if phot.get('nondetection', False)])
                        if len(nondetection_mjds) == 0:
                            min_nondetection = 9e9
                            max_nondetection = 9e9
                        else:
                            min_nondetection = nondetection_mjds[np.argmin(abs(nondetection_mjds - min_detection))]
                            max_nondetection = nondetection_mjds[np.argmin(abs(nondetection_mjds - max_detection))]

                        new_phot = []

                        for i, phot in enumerate(self.shifted_data[filt]):
                            if phot.get('nondetection', False):
                                ### Check if this nondetection is close to either 
                                ### the first or last nondetection in this filter
                                if abs(phot['mjd'] - min_nondetection) < 10 or abs(phot['mjd'] - max_nondetection) < 10:
                                    unshifted_mag = phot['mag'] + self.info['peak_mag']
                                    shifted_flux = np.log10(self.zps[filt] * 1e-11 * 10**(-0.4*unshifted_mag)) - np.log10(self.zps[self.info['peak_filt']] * 1e-11 * 10**(-0.4*self.info['peak_mag'])) #* 1e15
                                    phot['flux'] = shifted_flux
                                    phot['fluxerr'] = phot['err']
                                    new_phot.append(phot)
                            else:
                                unshifted_mag = phot['mag'] + self.info['peak_mag']
                                shifted_flux = np.log10(self.zps[filt] * 1e-11 * 10**(-0.4*unshifted_mag)) - np.log10(self.zps[self.info['peak_filt']] * 1e-11 * 10**(-0.4*self.info['peak_mag'])) #* 1e15
                                phot['flux'] = shifted_flux
                                phot['fluxerr'] = phot['err']
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
        
        if not self.info.get('ra', '') or not self.info.get('dec', ''):
            print('No coordinates for this object')
            return 
        
        coord = SkyCoord(ra=self.info['ra']*u.deg, dec=self.info['dec']*u.deg)
        exts = fm(np.asarray([self.wle[filt] * (1 + self.info.get('z', 0)) for filt in self.data.keys() if filt in self.wle.keys()]), sfd(coord))
        
        i = 0
        for filt in self.data.keys():
            if filt in self.wle.keys():

                new_phot = []
                for phot in self.data[filt]:
                    if not phot.get('ext_corrected', False):
                        phot['mag'] -= exts[i]
                        phot['ext_corrected'] = True
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
                        if not phot.get('ext_corrected', False):
                            phot['mag'] -= exts[i]
                            phot['ext_corrected'] = True
                        new_phot.append(phot)

                    self.shifted_data[filt] = new_phot
                    i += 1

                else:
                    self.shifted_data[filt] = []


    def plot_data(self, filts_to_plot=['all'], shifted_data_exists=False, view_shifted_data=False, offset=0, plot_fluxes=False):
        if not self.data: # check if data/SN has not been previously read in/initialized
            self.load_swift_data()
            self.load_json_data()
        
        fig, ax = plt.subplots()

        if filts_to_plot[0] == 'all': # if individual filters not specified, plot all by default
            filts_to_plot = colors.keys()

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
        
        for f in filts_to_plot:
            for filt, mag_list in data_to_plot.items():
                if f and f == filt:
                    if plot_fluxes:
                        mjds = np.asarray([phot['mjd'] for phot in mag_list if not phot.get('nondetection', False)])
                        fluxes = np.asarray([phot['flux'] for phot in mag_list if not phot.get('nondetection', False)])
                        errs = np.asarray([phot['fluxerr'] for phot in mag_list if not phot.get('nondetection', False)])
                        ax.errorbar(mjds, fluxes, yerr=errs, fmt='o', mec='black', color=colors.get(filt, 'k'), label=filt)

                        nondet_mjds = np.asarray([phot['mjd'] for phot in mag_list if phot.get('nondetection', False)])
                        nondet_fluxes = np.asarray([phot['flux'] for phot in mag_list if phot.get('nondetection', False)])
                        nondet_errs = np.asarray([phot['fluxerr'] for phot in mag_list if phot.get('nondetection', False)])
                        ax.errorbar(nondet_mjds, nondet_fluxes, yerr=nondet_errs, fmt='v', alpha=0.5, color=colors.get(filt, 'k'))                    
                    else:
                        mjds = np.asarray([phot['mjd'] for phot in mag_list if not phot.get('nondetection', False)])
                        mags = np.asarray([phot['mag'] for phot in mag_list if not phot.get('nondetection', False)])
                        errs = np.asarray([phot['err'] for phot in mag_list if not phot.get('nondetection', False)])

                        ax.errorbar(mjds, mags, yerr=errs, fmt='o', mec='black', color=colors.get(filt, 'k'), label=filt)
        if not plot_fluxes:
            plt.gca().invert_yaxis()
            plt.ylabel('Apparent Magnitude')
        else:
            plt.ylabel('Flux')

        plt.legend()
        plt.xlabel('MJD')
        plt.title(self.name)
        plt.minorticks_on()
        plt.show()


    def fit_for_max(self, filt, shift_array=[-3, -2, -1, 0, 1, 2, 3], plot=False, offset=0):
        """
        Takes as input arrays for MJD, mag, and err for a filter
        as well as the guess for the MJD of maximum and an array
        to shift the lightcurve over,
        and returns estimates of the peak MJD and mag at peak
        """
        mjd_array = np.asarray([phot['mjd'] for phot in self.data[filt] if not phot.get('nondetection', False)])
        mag_array = np.asarray([phot['mag'] for phot in self.data[filt] if not phot.get('nondetection', False)])
        err_array = np.asarray([phot['err'] for phot in self.data[filt] if not phot.get('nondetection', False)])

        if len(mag_array) < 4:#== 0:
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
        
        guess_mjd_max = guess_phases[np.where((guess_best_fit==min(guess_best_fit)))[0]][0]

        ### Do this because the array might not be ordered
        inds_to_fit = np.where((mjd_array > guess_mjd_max - 10) & (mjd_array < guess_mjd_max + 10))
        if len(inds_to_fit[0]) < 4:
            #print('Select a wider date range')
            return None, None
        
        if plot:
            fig, ax = plt.subplots()

        numdata = len(mjd_array[inds_to_fit])
        numiter = max(int(numdata * np.log(numdata)**2), 200)

        fit_mjds = mjd_array[inds_to_fit]
        fit_mags = mag_array[inds_to_fit]
        fit_errs = err_array[inds_to_fit]

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
                    ax.plot(fit_time, f(fit_time), color='black', linewidth=0.5)
                peak_mag = min(f(fit_time))
                peak_mags.append(peak_mag)
                peak_mjds.append(fit_time[np.argmin(f(fit_time))])

        if len(peak_mjds) == 0:
            return None, None

        if plot:
            ax.errorbar(mjd_array, mag_array, yerr=err_array, fmt='o', color='black')
            ax.errorbar(fit_mjds, fit_mags, yerr=fit_errs, fmt='o', color='blue', label='Used in Fitting')
            ax.errorbar(mean(peak_mjds), mean(peak_mags), xerr=stdev(peak_mjds), yerr=stdev(peak_mags), color='red', fmt='o', label='Best Fit Peak')
            plt.xlim(guess_mjd_max-10, guess_mjd_max+10)
            if len(mjd_array[inds_to_fit]) > 0:
                plt.ylim(min(mag_array[inds_to_fit])-0.5, max(mag_array[inds_to_fit])+0.5)
            plt.xlabel('MJD')
            plt.ylabel('Apparent Magnitude')
            plt.title(self.name)
            plt.legend()
            plt.gca().invert_yaxis()

            plt.show()

        self.info['peak_mjd'] = mean(peak_mjds)
        self.info['peak_mag'] = mean(peak_mags)
        self.info['peak_filt'] = filt
        self.info['searched'] = True


    def shift_to_max(self, filt, shift_array=[-3, -2, -1, 0, 1, 2, 3], plot=False, offset=0, shift_fluxes=False):

        if not self.data:
            self.load_swift_data()
            self.load_json_data()

        if filt not in self.data.keys():
            return [], [], [], []

        if not self.info.get('peak_mjd') and not self.info.get('peak_mag'):
            self.fit_for_max(filt, shift_array=shift_array, plot=plot, offset=offset)

            if not self.info.get('peak_mjd', 0) > 0:
                for newfilt in ['V', 'g', 'c', 'B', 'r', 'o', 'U', 'i', 'UVW1']:
                    if newfilt in self.data.keys() and newfilt != filt:
                        self.fit_for_max(newfilt, shift_array=shift_array, plot=plot, offset=offset)

                        if self.info.get('peak_mjd', 0) > 0:
                            break

                if newfilt == 'UVW1' and not self.info.get('peak_mjd', 0) > 0:
                    print('Reached last filter and could not fit for peak for ', self.name)
                    self.info['searched'] = True
        
        if not self.info.get('peak_mag', 0) > 0:
            return [], [], [], []

        mjds = np.asarray([phot['mjd'] for phot in self.data[filt]]) - self.info['peak_mjd']
        mags = np.asarray([phot['mag'] for phot in self.data[filt]]) - self.info['peak_mag']
        errs = np.asarray([phot['err'] for phot in self.data[filt]])
        nondets = np.asarray([phot.get('nondetection', False) for phot in self.data[filt]])

        if plot:
            plt.errorbar(mjds[np.where((nondets==False))[0]], mags[np.where((nondets==False))[0]], yerr=errs[np.where((nondets==False))[0]], fmt='o', mec='black', color=colors.get(filt, 'k'), label=filt+'-band')
            plt.scatter(mjds[np.where((nondets==True))[0]], mags[np.where((nondets==True))[0]], marker='v', color=colors.get(filt, 'k'), alpha=0.2)

            plt.xlabel('Shifted Time [days]')
            plt.ylabel('Shifted Magnitude')
            plt.title(self.name+'-Shifted Data')
            plt.legend()
            plt.gca().invert_yaxis()

            plt.show()

        self.shifted_data.setdefault(filt, []).extend(
            [{'mjd': mjds[i], 'mag': mags[i], 'err': errs[i], 'nondetection': nondets[i]} for i in range(len(mjds))]
        )

        if shift_fluxes:
            self.convert_to_fluxes()
            shifted_mjd = np.asarray([phot['mjd'] for phot in self.shifted_data[filt]])
            shifted_flux = np.asarray([phot['flux'] for phot in self.shifted_data[filt]])
            shifted_err = np.asarray([phot['fluxerr'] for phot in self.shifted_data[filt]])
            nondets = np.asarray([phot.get('nondetection', False) for phot in self.shifted_data[filt]])

            return shifted_mjd, shifted_flux, shifted_err, nondets

        return mjds, mags, errs, nondets
    

    def interactively_fit_for_max(self, filt='', shift_array=[-3, -2, -1, 0, 1, 2, 3], plot=True, offset=0, save_to_caat=False, force=False):

        self.load_json_data()
        self.load_swift_data()
        self.shifted_data = {}

        if not filt:
            self.plot_data()
            print('Data in filters {}\n'.format(list(self.data.keys())))

            filt = input('Which filter would you like to use to fit for max? To skip, type "skip"\n')
            if filt == 'skip':
                return 
            
        mjds, _, _, _ = self.shift_to_max(filt, shift_array=shift_array, plot=plot, offset=offset)

        if len(mjds) == 0:
            refit = input('No photometry found for this filter. Try to refit? y/n \n')

        else:
            self.plot_data(view_shifted_data=True)
            refit = input('Refit the data with new filter or offset? y/n \n')

        if refit == 'n' and save_to_caat:
            self.write_info_to_caat_file(force=force)

        elif refit == 'n' and not save_to_caat:
            print('To save these parameters, rerun with "save_to_caat=True"')

        elif refit == 'y':
            self.info = {}
            newfilt = input('Try fitting a new filter? If so, enter the filter here. If not, leave blank to pick new offset\n')

            if newfilt:
                self.interactively_fit_for_max(newfilt, shift_array=shift_array, plot=plot, offset=offset, save_to_caat=save_to_caat, force=force)
                
            else:
                newoffset = input('Enter new offset here\n')

                if newoffset:
                    self.interactively_fit_for_max(filt, shift_array=shift_array, plot=plot, offset=float(newoffset), save_to_caat=save_to_caat, force=force)


    def log_transform_time(self, phases, phase_start=30):

        return np.log(phases + phase_start)
    

class SNCollection:

    """
    A SNCollection object, which holds an arbitrary number of SNe
    
    """
    base_path = '../data/'


    def __init__(self, names: Union[str, None] = None, 
                 sntype: Union[str, None] = None, 
                 snsubtype: Union[str, None] = None, 
                 SNe: Union[list[SN], None] = None,
                **kwargs):
        
        self.subtypes = list(kwargs.keys())
        
        if(isinstance(SNe, SN)):
            self.sne = SNe
        else:
            if(isinstance(names, list)):
                self.sne = [SN(name) for name in names]
            else:
                if(type(sntype) is not None):
                    #convert this to a logger statement
                    print(f"Loading SN Type: {sntype}, Subtype: {snsubtype}")
                    caat = CAAT()
                    type_list = caat.get_sne_by_type(sntype, snsubtype)
                    print(type_list)
                    self.sne = [SN(name) for name in type_list]
                    self.type=sntype
                    self.subtype=snsubtype

    def __repr__(self):
        print("Collection of SN Objects")
        return self.sne

    #@property
    #def sne(self):
    #    return self.sne
    
    def get_type_list(self):
        #Maybe this lives in a separate class that handles the csv db file
        raise NotImplementedError

    def plot_all_lcs(self, filts=['all'], log_transform=False, plot_fluxes=False):
        """plot all light curves of given subtype/collection
            can plot single, multiple or all bands"""
        sne = self.sne
        print(f"Plotting all {len(sne)} lightcurves in the collection")

        fig, ax = plt.subplots()
        if filts[0] is not 'all':
            filts_to_plot = filts
        else:
            print(f"BEWARE -- plotting ALL bands of ALL objects in the collection -- plot will be messy.\n")
            filts_to_plot = colors.keys()

        for i,f in enumerate(filts_to_plot):
            for sn in sne:
                mjds, mags, errs, nondets = sn.shift_to_max(f, shift_fluxes=plot_fluxes)
                if len(mjds) > 0:
                    if log_transform is not False:
                        mjds = sn.log_transform_time(mjds, phase_start=log_transform)

                    if plot_fluxes:

                        nondet_inds = np.where((nondets==False))[0]
                        det_inds = np.where((nondets==True))[0]
                        ax.errorbar(mjds[nondet_inds], mags[nondet_inds], yerr=errs[nondet_inds], fmt='o', mec='black', color=colors.get(f, 'k'), label=f)
                        ax.scatter(mjds[det_inds], mags[det_inds], marker='v', alpha=0.2, color=colors.get(f, 'k'))
                    else:
                        ax.errorbar(mjds, mags, yerr=errs, fmt='o', mec='black', color=colors.get(f, 'k'), label=f)
            filtText = f+'\n'
            plt.figtext(0.95, 0.75-(0.05*i), filtText, fontsize=14,color=colors.get(f))

        # cannot figure out how to display only unique labels (filters) in order w/ matching handles 
        # plt.legend()
        
        if log_transform is False:
            ax.set_xlabel('Shifted Time [days]')
        else:
            ax.set_xlabel('Log(Shifted Time)')

        if plot_fluxes:
            ax.set_ylabel('Shifted Fluxes')
        else:
            ax.set_ylabel('Shifted Magnitudes')
            plt.gca().invert_yaxis()
        plt.title('Lightcurves for collection of {} objects\nType:{}, Subtype:{}'.format(len(sne),self.type,self.subtype))
        plt.show()


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
        subtype_list = (caat.caat["Type"] == self.type)
        self.subtypes = list(set(caat.caat[subtype_list].Subtype.values))


    def build_object_list(self):
    
        caat = CAAT()
        type_list = caat.get_sne_by_type(self.type)
        self.sne = [SN(name) for name in type_list]
        

class Fitter:

    """
    A Fitter object, fitting the light curves of a class (Type) of supernovae
    """
    
    def __init__(self, collection):
        # collection long term could be a individual SN or a SNCollection, 
        # make sure this works at some point
        self.collection = collection


class RBFKernel:

    """
    An RBFKernel object, to be used in GP fitting
    Allows users to define Kernel parameters such as length scale, bounds, etc.
    """

    def __init__(self, length_scale, length_scale_bounds):

        self.kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)


class WhiteNoiseKernel:

    """
    An RBFKernel object, to be used in GP fitting
    Allows users to define Kernel parameters such as noise level, bounds, etc.
    """

    def __init__(self, noise_level, noise_level_bounds):

        self.kernel = WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
    

class MaternKernel:

    """
    A MaternKernel, to be used in GP fitting
    Allows users to define Kernel parameters such as length scale, bounds, etc.
    """

    def __init__(self, length_scale, length_scale_bounds, nu):

        self.kernel = Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)


class GP(Fitter):

    """
    GP fit to a single band
    """
    wle = {'u': 3560,  'g': 4830, 'r': 6260, 'i': 7670, 'z': 8890, 'y': 9600, 'w':5985, 'Y': 9600,
           'U': 3600,  'B': 4380, 'V': 5450, 'R': 6410, 'G': 6730, 'E': 6730, 'I': 7980, 'J': 12200, 'H': 16300,
           'K': 21900, 'UVW2': 2030, 'UVM2': 2231, 'UVW1': 2634, 'F': 1516, 'N': 2267, 'o': 6790, 'c': 5330,
           'W': 33526, 'Q': 46028
    }

    def __init__(self, sne_collection, kernel):
        super().__init__(sne_collection)
        self.kernel = kernel


    def process_dataset_for_gp(self, filt, phasemin, phasemax, log_transform=False, sn_set=None, use_fluxes=False):
        """
        Loads all the data, shifts the data to peak,
        and concatenates the data for the object's SN collection
        or a provided SN set
        """
        
        phases, mags, errs, wls = np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])

        if sn_set is None:
            sn_set = self.collection.sne

        for sn in sn_set:

            z = sn.info.get('z', 0)

            if len(sn.data) == 0:
                sn.load_swift_data()
                sn.load_json_data()

            if len(sn.shifted_data) == 0:
                ### Check to see if we've already tried to fit for maximum
                if not sn.info:
                    shifted_mjd = []
                else:
                    shifted_mjd, shifted_mag, err, nondets = sn.shift_to_max(filt, shift_fluxes=use_fluxes)

            else:
                ### We already successfully fit for peak, so get the shifted photometry for this filter
                shifted_mjd, shifted_mag, err, nondets = sn.shift_to_max(filt, shift_fluxes=use_fluxes)
                    
            if len(shifted_mjd) > 0:

                if log_transform is not False:
                    shifted_mjd = sn.log_transform_time(shifted_mjd, phase_start=log_transform)

                if log_transform is not False:
                    inds_to_fit = np.where((shifted_mjd > np.log(phasemin+log_transform)) & (shifted_mjd < np.log(phasemax+log_transform)))[0]
                else:
                    inds_to_fit = np.where((shifted_mjd > phasemin) & (shifted_mjd < phasemax))[0]
                
                phases = np.concatenate((phases, shifted_mjd[inds_to_fit]))
                mags = np.concatenate((mags, shifted_mag[inds_to_fit]))
                errs = np.concatenate((errs, err[inds_to_fit]))
                if log_transform is not False:
                    wls = np.concatenate((wls, np.ones(len(inds_to_fit)) * np.log10(self.wle[filt] * (1 + z))))
                else:
                    wls = np.concatenate((wls, np.ones(len(inds_to_fit)) * self.wle[filt] * (1 + z)))

        return phases.reshape(-1, 1), mags.reshape(-1, 1), errs.reshape(-1, 1), wls.reshape(-1, 1)


    def run_gp(self, filt, phasemin, phasemax, test_size, use_fluxes=False):

        phases, mags, errs, _ = self.process_dataset_for_gp(filt, phasemin, phasemax, use_fluxes=use_fluxes)
        X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(phases, mags, errs, test_size=test_size)

        ### Get array of errors at each timestep
        min_phase, max_phase = sorted(X_train)[0], sorted(X_train)[-1]
        phase_grid = np.linspace(min_phase, max_phase, len(X_train))
        phase_grid_space = (max_phase - min_phase)/len(X_train)

        err_grid = np.ones(len(phase_grid))
        for mjd in phase_grid:
            ind = np.where((X_train < mjd + phase_grid_space/2) & (X_train > mjd - phase_grid_space/2))[0]
            mags_at_this_phase = Y_train[ind]
            if len(mags_at_this_phase) > 1:
                std_mag = max(np.std(mags_at_this_phase), 0.01)
            elif len(mags_at_this_phase) == 1:
                std_mag = Z_train[ind]
            else:
                std_mag = 0.1
            err_grid[ind] = std_mag

        ### Run the GP
        gaussian_process = GaussianProcessRegressor(
            kernel=self.kernel, alpha=err_grid, n_restarts_optimizer=9
        )
        gaussian_process.fit(X_train, Y_train)

        self.gaussian_process = gaussian_process

        return gaussian_process, phases, mags, errs, err_grid


    def predict_gp(self, filt, phasemin, phasemax, test_size, plot=False, use_fluxes=False):

        gaussian_process, phases, mags, errs, err_grid = self.run_gp(filt, phasemin, phasemax, test_size, use_fluxes=use_fluxes)

        mean_prediction, std_prediction = gaussian_process.predict(sorted(phases), return_std=True)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(sorted(phases), mean_prediction, color='k', label='GP fit',zorder=10)
            ax.errorbar(phases, mags.reshape(-1), errs.reshape(-1), fmt='o', color=colors.get(filt, 'k'), alpha=0.2, label=filt,zorder=0)
            ax.fill_between(
                    sorted(phases.ravel()),
                    mean_prediction - 1.96*std_prediction,
                    mean_prediction + 1.96*std_prediction,
                    alpha=0.5,
                    color='lightgray',
                    label='95\% confidence region',
                    zorder=10
            )
            plt.gca().invert_yaxis()
            plt.xlabel('Shifted Time [days]')
            plt.ylabel('Shifted Magnitude')
            plt.title('Single-Filter GP Fit')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()


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
                interp = interp1d(interp_array[notnan_inds],
                                    row[notnan_inds],
                                    'cubic')

                interp_row = interp(interp_array[min(notnan_inds):max(notnan_inds)])
                savgol_l = savgol_filter(interp_row, filter_window, 3, mode='mirror')
                row[min(notnan_inds):max(notnan_inds)] = savgol_l
                grid[i] = row
        return grid   
      

    def build_samples_3d(self, filt, phasemin, phasemax, log_transform=False, sn_set=None, use_fluxes=False):
        """
        Builds the data set from the SN collection for a given filter
        and returns, along with the phases, wls, and mags,
        the uncertainty in the measurements as the standard deviation
        of the photometry at each phase step
        """

        phases, mags, errs, wls = self.process_dataset_for_gp(filt, phasemin, phasemax, log_transform=log_transform, sn_set=sn_set, use_fluxes=use_fluxes)

        min_phase, max_phase = sorted(phases)[0], sorted(phases)[-1]
        phase_grid = np.linspace(min_phase, max_phase, len(phases))
        phase_grid_space = (max_phase - min_phase)/len(phases)

        err_grid = np.ones(len(phase_grid))
        for mjd in phase_grid:
            ind = np.where((phases < mjd + phase_grid_space/2) & (phases > mjd - phase_grid_space/2))[0]
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


    def process_dataset_for_gp_3d(self, filtlist, phasemin, phasemax, log_transform=False, plot=False, fit_residuals=False, set_to_normalize=None, use_fluxes=False):
        """
        Processes the data set for the GP3D object's SN collection and
        (optionally) a SN set filter-by-filter and returns 
        arrays of the SN collection's photometric details
        as well as the photometric details of the SN set to normalize to
        """

        all_phases, all_wls, all_mags, all_errs = [], [], [], []

        for filt in filtlist:
            phases, wl_grid, mags, err_grid = self.build_samples_3d(filt, phasemin, phasemax, log_transform=log_transform, use_fluxes=use_fluxes)

            all_phases = np.concatenate((all_phases, phases.flatten()))
            all_wls = np.concatenate((all_wls, wl_grid.flatten()))
            all_mags = np.concatenate((all_mags, mags.flatten()))
            all_errs = np.concatenate((all_errs, err_grid.flatten()))

        if not fit_residuals:
            return all_phases, all_wls, all_mags, all_errs

        ### Create the template grid from the observations
        if set_to_normalize is not None:
            all_template_phases, all_template_wls, all_template_mags, all_template_errs = [], [], [], []
            for filt in filtlist:
                phases, wl_grid, mags, err_grid = self.build_samples_3d(filt, phasemin, phasemax, log_transform=log_transform, sn_set=set_to_normalize, use_fluxes=use_fluxes)

                all_template_phases = np.concatenate((all_template_phases, phases.flatten()))
                all_template_wls = np.concatenate((all_template_wls, wl_grid.flatten()))
                all_template_mags = np.concatenate((all_template_mags, mags.flatten()))
                all_template_errs = np.concatenate((all_template_errs, err_grid.flatten()))               
        else:
            all_template_phases = all_phases
            all_template_wls = all_wls
            all_template_mags = all_mags
            all_template_errs = all_errs

        return all_phases, all_wls, all_mags, all_errs, all_template_phases, all_template_wls, all_template_mags, all_template_errs


    def construct_median_grid(self, phasemin, phasemax, filtlist, all_template_phases, all_template_wls, all_template_mags, all_template_errs, log_transform=False, plot=False):
        """
        Takes as input the photometry from the sn set to normalize
        and constructs a 2D template grid consisting of the median photometry
        at each phase and wl step
        """

        if log_transform is not False:
            phase_grid_linear = np.arange(phasemin, phasemax, 1/24.) # Grid of phases to iterate over, by hour
            phase_grid = np.log(phase_grid_linear + log_transform) # Grid of phases in log space

            wl_grid_linear = np.arange(min([self.wle[f] for f in filtlist])-500, max([self.wle[f] for f in filtlist])+500, 99.5) # Grid of wavelengths to iterate over, by 100 A
            wl_grid = np.log10(wl_grid_linear)

        else:
            phase_grid = np.arange(phasemin, phasemax, 1/24.) # Grid of phases to iterate over, by hour
            wl_grid = np.arange(min([self.wle[f] for f in filtlist])-500, max([self.wle[f] for f in filtlist])+500, 99.5) # Grid of wavelengths to iterate over, by 100 A
        
        mag_grid = np.empty((len(phase_grid), len(wl_grid)))
        mag_grid[:] = np.nan
        err_grid = np.copy(mag_grid)

        for i in range(len(phase_grid)):
            for j in range(len(wl_grid)):

                ### Get all data that falls within this phase + 5 days, and this wl +- 100 A
                if log_transform is not False:
                    inds = np.where((np.exp(all_template_phases) - np.exp(phase_grid[i]) <= 5.0) & (np.exp(all_template_phases) - np.exp(phase_grid[i]) > 0.0) & (abs(10**all_template_wls - 10**wl_grid[j]) <= 500))[0]
                else:
                    inds = np.where((all_template_phases - phase_grid[i] <= 5.0) & (all_template_phases - phase_grid[i] > 0.0) & (abs(all_template_wls - wl_grid[j]) <= 500))[0]
                
                if len(inds) > 0:
                
                    median_mag = np.median(all_template_mags[inds])
                    iqr = np.subtract(*np.percentile(all_template_mags[inds], [75, 25]))
                    
                    mag_grid[i,j] = median_mag
                    err_grid[i,j] = iqr

        mag_grid = self.interpolate_grid(mag_grid.T, phase_grid)
        mag_grid = mag_grid.T
        mag_grid = self.interpolate_grid(mag_grid, wl_grid, filter_window=31)
        err_grid = self.interpolate_grid(err_grid.T, phase_grid)
        err_grid = err_grid.T
        err_grid = self.interpolate_grid(err_grid, wl_grid, filter_window=31)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            if log_transform is not False:
                X, Y = np.meshgrid(np.exp(phase_grid) - log_transform, 10**wl_grid)
            else:
                X, Y = np.meshgrid(phase_grid, wl_grid)

            Z = mag_grid.T

            ax.plot_surface(X, Y, Z)
            ax.invert_zaxis()
            ax.set_xlabel('Phase Grid')
            ax.set_ylabel('Wavelengths [nm]')
            #ax.set_zlabel('Magnitude')
            plt.show()

            for filt in filtlist:
                if log_transform is not False:
                    wl_inds = np.where((abs(10**wl_grid - self.wle[filt]) <= 100))[0]
                else:
                    wl_inds = np.where((abs(wl_grid - self.wle[filt]) <= 100))[0]

                plt.errorbar(phase_grid, mag_grid[:,wl_inds[0]], yerr=abs(err_grid[:,wl_inds[0]]), fmt='o')
                plt.gca().invert_yaxis()
                plt.title(filt)
                plt.show()

        return phase_grid, wl_grid, mag_grid, err_grid
    

    def construct_polynomial_grid(self, phasemin, phasemax, filtlist, all_template_phases, all_template_wls, all_template_mags, all_template_errs, log_transform=False, plot=False):
        """
        Takes as input the photometry from the sn set to normalize
        and constructs a 2D template grid consisting of the polynomial fit
        to the SN set to normalize photometry at each phase and wl step
        """    

        if log_transform is not False:
            phase_grid_linear = np.arange(phasemin, phasemax, 1/24.) # Grid of phases to iterate over, by hour
            phase_grid = np.log(phase_grid_linear + log_transform) # Grid of phases in log space

            wl_grid_linear = np.arange(min([self.wle[f] for f in filtlist])-500, max([self.wle[f] for f in filtlist])+500, 99.5) # Grid of wavelengths to iterate over, by 100 A
            wl_grid = np.log10(wl_grid_linear)

        else:
            phase_grid = np.arange(phasemin, phasemax, 1/24.) # Grid of phases to iterate over, by hour
            wl_grid = np.arange(min([self.wle[f] for f in filtlist])-500, max([self.wle[f] for f in filtlist])+500, 99.5) # Grid of wavelengths to iterate over, by 100 A
        
        mag_grid = np.empty((len(phase_grid), len(wl_grid)))
        mag_grid[:] = np.nan
        err_grid = np.copy(mag_grid)

        for j in range(len(wl_grid)):

            ### Get all data that falls within this wl +- 100 A
            if log_transform is not False:
                inds = np.where((abs(10**all_template_wls - 10**wl_grid[j]) <= 100))[0]
                ### Add an array of fake measurements to anchor the ends of the fit
                anchor_phases = np.asarray([np.log(phasemin + log_transform), np.log(phasemin + 2.5 + log_transform), np.log(phasemax + log_transform)])
            else:
                inds = np.where((abs(all_template_wls - wl_grid[j]) <= 100))[0]
                anchor_phases = np.asarray([phasemin, phasemax])
            
            if len(inds) > 0:

                fit_coeffs = np.polyfit(
                    np.concatenate((all_template_phases[inds], anchor_phases)), 
                    np.concatenate((all_template_mags[inds], np.asarray([-5.0, -4.0, -5.0]))),#np.ones(len(anchor_phases)) * -5.0)), 
                    3, 
                    w=1/(np.sqrt(
                        (np.concatenate((all_template_errs[inds], np.ones(len(anchor_phases)) * 0.05)))**2 + (np.ones(len(all_template_errs[inds]) + len(anchor_phases)) * 0.1)**2))
                )
                fit = np.poly1d(fit_coeffs)
                grid_mags = fit(phase_grid)
            
                mag_grid[:,j] = grid_mags
                err_grid[:,j] = np.ones(len(phase_grid)) * np.median(abs(all_template_mags[inds] - fit(all_template_phases[inds])))

        ### Interpolate over the wavelengths to get a complete 2D grid
        mag_grid = self.interpolate_grid(mag_grid, wl_grid, filter_window=31)
        err_grid = self.interpolate_grid(err_grid, wl_grid, filter_window=31)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            if log_transform is not False:
                X, Y = np.meshgrid(np.exp(phase_grid) - log_transform, 10**wl_grid)
            else:
                X, Y = np.meshgrid(phase_grid, wl_grid)

            Z = mag_grid.T

            ax.plot_surface(X, Y, Z)
            ax.invert_zaxis()
            ax.set_xlabel('Phase Grid')
            ax.set_ylabel('Wavelengths [Angstroms]')
            #ax.set_zlabel('Magnitude')
            plt.show()

            # for filt in filtlist:
            #     if log_transform is not False:
            #         wl_inds = np.where((abs(10**wl_grid - self.wle[filt]) <= 100))[0]
            #     else:
            #         wl_inds = np.where((abs(wl_grid - self.wle[filt]) <= 100))[0]

            #     plt.errorbar(phase_grid, mag_grid[:,wl_inds[0]], yerr=abs(err_grid[:,wl_inds[0]]), fmt='o')
            #     plt.gca().invert_yaxis()
            #     plt.title(filt)
            #     plt.show()

        return phase_grid, wl_grid, mag_grid, err_grid
    

    def subtract_data_from_grid(self, sn, phasemin, phasemax, filtlist, phase_grid, wl_grid, mag_grid, err_grid, log_transform=False, plot=False, use_fluxes=False):
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
                if sn.info.get('searched', False):
                    shifted_mjd, shifted_mag, err, nondets = [], [], [], []
                else:
                    sn.correct_for_galactic_extinction()
                    shifted_mjd, shifted_mag, err, nondets = sn.shift_to_max(filt, shift_fluxes=use_fluxes)
            else:
                if filt in sn.shifted_data.keys():
                    if use_fluxes:
                        sn.correct_for_galactic_extinction()
                        sn.convert_to_fluxes()
                        shifted_mag = np.asarray([phot['flux'] for phot in sn.shifted_data[filt]])
                        err = np.asarray([phot['fluxerr'] for phot in sn.shifted_data[filt]])
                        nondets = np.asarray([phot.get('nondetection', False) for phot in sn.shifted_data[filt]])
                    else:
                        sn.correct_for_galactic_extinction()
                        shifted_mag = np.asarray([phot['mag'] for phot in sn.shifted_data[filt]])
                        err = np.asarray([phot['err'] for phot in sn.shifted_data[filt]])
                        nondets = np.asarray([phot.get('nondetection', False) for phot in sn.shifted_data[filt]])
                    
                    shifted_mjd = np.asarray([phot['mjd'] for phot in sn.shifted_data[filt]])
                else:
                    shifted_mjd = []
            
            if len(shifted_mjd) > 0:

                if log_transform is not False:
                    shifted_mjd = sn.log_transform_time(shifted_mjd, phase_start=log_transform)
                    inds_to_fit = np.where((shifted_mjd > np.log(phasemin+log_transform)) & (shifted_mjd < np.log(phasemax+log_transform)))[0]
                    ### The wl corresponding to wl_ind is no more than the wl grid spacing away from the true wl being measured
                    wl_ind = np.argmin(abs(10**wl_grid - self.wle[filt] * (1 + sn.info.get('z', 0))))
                else:
                    inds_to_fit = np.where((shifted_mjd > phasemin) & (shifted_mjd < phasemax))[0]
                    wl_ind = np.argmin(abs(10**wl_grid - self.wle[filt] * (1 + sn.info.get('z', 0))))

                phases = shifted_mjd[inds_to_fit]
                mags = shifted_mag[inds_to_fit]
                errs = err[inds_to_fit]
                current_nondets = nondets[inds_to_fit]

                if plot and len(phases) > 0:
                    print('Plotting for filter ', filt)
                    fig, ax = plt.subplots()
                    ax.plot(phase_grid, mag_grid[:, wl_ind], color=colors.get(filt, 'k'), label='template')
                
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
                        wl_residual = np.log10(self.wle[filt]*(1 + sn.info.get('z', 0)))
                    else:
                        wl_residual = self.wle[filt]*(1 + sn.info.get('z', 0))

                    residuals.setdefault(filt, []).extend([{'phase_residual': phase,
                                                            'wl_residual': wl_residual,
                                                            'mag_residual': mags[i] - mag_grid[phase_ind, wl_ind],
                                                            'err_residual': errs[i],
                                                            'mag': mags[i],
                                                            'nondetection': current_nondets[i]}])

                    if plot:
                        ax.errorbar(phase, mags[i] - mag_grid[phase_ind, wl_ind], yerr=np.sqrt(errs[i]**2 + err_grid[phase_ind, wl_ind]**2), marker='o', color='k')
                        ax.errorbar(phase, mags[i], yerr=errs[i], fmt='o', color=colors.get(filt, 'k'))
                                
                if plot and len(phases) > 0:
                    plt.axhline(y=0, linestyle='--', color='gray')
                    ax.errorbar([], [], yerr=[], marker='o', color='k', label='residuals', alpha=0.2)
                    ax.errorbar([], [], yerr=[], fmt='o', color=colors.get(filt, 'k'), label='data', alpha=0.5)
                    plt.gca().invert_yaxis()
                    plt.legend()
                    plt.show()

        return residuals


    def run_gp(self, filtlist, phasemin, phasemax, test_size=0.9, plot=False, log_transform=False, fit_residuals=False, set_to_normalize=None, subtract_median=False, subtract_polynomial=False, interactive=False, use_fluxes=False):
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
        if fit_residuals:
            all_phases, all_wls, all_mags, all_errs, all_template_phases, all_template_wls, all_template_mags, all_template_errs = self.process_dataset_for_gp_3d(filtlist, phasemin, phasemax, log_transform=log_transform, plot=False, fit_residuals=True, set_to_normalize=set_to_normalize, use_fluxes=use_fluxes)
            kernel_params = []
            gaussian_processes = []
            if subtract_polynomial:
                phase_grid, wl_grid, mag_grid, err_grid = self.construct_polynomial_grid(phasemin, phasemax, filtlist, all_template_phases, all_template_wls, all_template_mags, all_template_errs, log_transform=log_transform, plot=plot)
            elif subtract_median:
                phase_grid, wl_grid, mag_grid, err_grid = self.construct_median_grid(phasemin, phasemax, filtlist, all_template_phases, all_template_wls, all_template_mags, all_template_errs, log_transform=log_transform, plot=plot)
            else:
                raise Exception("Must toggle either subtract_median or subtract_polynomial as True to run GP3D")
            for sn in self.collection.sne:
                print(sn.name)
                residuals = self.subtract_data_from_grid(sn, phasemin, phasemax, filtlist, phase_grid, wl_grid, mag_grid, err_grid, log_transform=log_transform, plot=False, use_fluxes=use_fluxes)

                phase_residuals = np.asarray([p['phase_residual'] for phot_list in residuals.values() for p in phot_list])
                wl_residuals = np.asarray([p['wl_residual'] for phot_list in residuals.values() for p in phot_list])
                mag_residuals = np.asarray([p['mag_residual'] for phot_list in residuals.values() for p in phot_list])
                err_residuals = np.asarray([p['err_residual'] for phot_list in residuals.values() for p in phot_list])

                x = np.vstack((phase_residuals, wl_residuals)).T
                y = mag_residuals
                if len(y) > 1:
                    # We have enough points to fit
                    err = err_residuals

                    gaussian_process = GaussianProcessRegressor(
                        kernel=self.kernel, alpha=err, n_restarts_optimizer=10
                    )
                    gaussian_process.fit(x, y)

                    if plot:
                        fig, ax = plt.subplots()
                    for filt in filtlist:
                        
                        if log_transform is not False:
                            if len(wl_residuals[abs(10**wl_residuals - self.wle[filt]*(1 + sn.info.get('z', 0))) < 1]) == 0:
                                continue
                            test_times_linear = np.arange(phasemin, phasemax, 1./24)
                            test_times = np.log(test_times_linear + log_transform)
                            test_waves = np.ones(len(test_times)) * np.log10(self.wle[filt]*(1 + sn.info.get('z', 0)))
                        else:
                            if len(wl_residuals[wl_residuals==self.wle[filt]]*(1 + sn.info.get('z', 0))) == 0:
                                continue
                            test_times = np.arange(phasemin, phasemax, 1./24)
                            test_waves = np.ones(len(test_times)) * self.wle[filt]*(1 + sn.info.get('z', 0))

                        ### Trying to convert back to normalized magnitudes here
                        if log_transform is not None:
                            wl_ind = np.argmin(abs(10**wl_grid - self.wle[filt]*(1 + sn.info.get('z', 0))))
                        else:
                            wl_ind = np.argmin(abs(wl_grid - self.wle[filt]*(1 + sn.info.get('z', 0))))
                        template_mags = []
                        for i in range(len(phase_grid)):
                            template_mags.append(mag_grid[i, wl_ind])

                        template_mags = np.asarray(template_mags)

                        test_prediction, std_prediction = gaussian_process.predict(np.vstack((test_times, test_waves)).T, return_std=True)
                        if log_transform is not False:
                            test_times = np.exp(test_times) - log_transform

                        if plot:
                            ax.plot(test_times, test_prediction+template_mags, label=filt, color=colors.get(filt, 'k'))
                            ax.fill_between(
                                    test_times,
                                    test_prediction - 1.96*std_prediction + template_mags,
                                    test_prediction + 1.96*std_prediction + template_mags,
                                    alpha=0.2,
                                    color=colors.get(filt, 'k')
                            )

                        # Plot the SN photometry
                        shifted_mjd = np.asarray([phot['mjd'] for phot in sn.shifted_data[filt]])
                        if log_transform is not False:
                            shifted_mjd = sn.log_transform_time(shifted_mjd, phase_start=log_transform)

                        if log_transform is not False:
                            inds_to_fit = np.where((shifted_mjd > np.log(phasemin+log_transform)) & (shifted_mjd < np.log(phasemax+log_transform)))[0]
                        else:
                            inds_to_fit = np.where((shifted_mjd > phasemin) & (shifted_mjd < phasemax))[0]

                        key_to_plot = 'flux' if use_fluxes else 'mag'
                        if log_transform is not False and plot:

                            ax.errorbar(np.exp(np.asarray([p['phase_residual'] for p in residuals[filt] if p['phase_residual'] > np.log(phasemin+log_transform) and p['phase_residual'] < np.log(phasemax+log_transform) and not p['nondetection']])) - log_transform,
                                        [p['mag'] for p in residuals[filt] if p['phase_residual'] > np.log(phasemin+log_transform) and p['phase_residual'] < np.log(phasemax+log_transform) and not p['nondetection']],
                                        yerr=[p['err_residual'] for p in residuals[filt] if p['phase_residual'] > np.log(phasemin+log_transform) and p['phase_residual'] < np.log(phasemax+log_transform) and not p['nondetection']],
                                        fmt='o',
                                        color=colors.get(filt, 'k'),
                                        mec='k')
                            
                            ax.scatter(np.exp(np.asarray([p['phase_residual'] for p in residuals[filt] if p['phase_residual'] > np.log(phasemin+log_transform) and p['phase_residual'] < np.log(phasemax+log_transform) and p['nondetection']])) - log_transform,
                                        [p['mag'] for p in residuals[filt] if p['phase_residual'] > np.log(phasemin+log_transform) and p['phase_residual'] < np.log(phasemax+log_transform) and p['nondetection']],
                                        marker='v',
                                        color=colors.get(filt, 'k'),
                                        alpha=0.5)

                        elif plot:
                            ax.errorbar(phase_residuals[wl_residuals==self.wle[filt]*(1 + sn.info.get('z', 0))], 
                                        np.asarray([p[key_to_plot] for p in sn.shifted_data[filt]])[inds_to_fit],
                                        yerr=err_residuals[wl_residuals==self.wle[filt]*(1 + sn.info.get('z', 0))], 
                                        fmt='o',
                                        color=colors.get(filt, 'k'),
                                        mec='k')
                        
                    if plot:
                        if not use_fluxes:
                            ax.invert_yaxis()
                            ax.set_ylabel('Magnitude Relative to Peak')
                        else:
                            ax.set_ylabel('Flux Relative to Peak')

                        ax.set_xlabel('Normalized Time [days]')
                        plt.title(sn.name)
                        plt.legend()
                        plt.show()

                        if (subtract_median or subtract_polynomial) and interactive:
                            use_for_template = input('Use this fit to construct a template? y/n')

                    if subtract_median or subtract_polynomial:

                        if log_transform is not None:
                            waves_to_predict = np.unique(wl_residuals)
                            diffs = abs(np.subtract.outer(10**wl_grid, 10**waves_to_predict)) # The difference between our measurement wavelengths and the wl grid

                        else:
                            waves_to_predict = np.unique(wl_residuals)
                            diffs = abs(np.subtract.outer(wl_grid, waves_to_predict))

                        ### Compare the wavelengths of our measured filters to those in the wl grid
                        ### and fit for those grid wls that are within 500 A of one of our measurements
                        wl_inds_fitted = np.unique(np.where((diffs < 500.))[0])
                        
                        x, y = np.meshgrid(phase_grid, wl_grid[wl_inds_fitted])
                        test_prediction, std_prediction = gaussian_process.predict(np.vstack((x.ravel(), y.ravel())).T, return_std=True)
                        test_prediction = np.asarray(test_prediction)

                        template_mags = []

                        for wl_ind in wl_inds_fitted:
                            
                            for i in range(len(phase_grid)):
                                template_mags.append(mag_grid[i, wl_ind])
                                ###NOTE: Some of these template mags are NaNs

                        template_mags = np.asarray(template_mags).reshape((len(x), -1))
                                            
                        ### Put the fitted wavelengths back in the right spot on the grid
                        ### and append to the gaussian processes array
                        test_prediction_reshaped = test_prediction.reshape((len(x), -1)) + template_mags
                        gp_grid = np.empty((len(wl_grid), len(phase_grid)))
                        gp_grid[:] = np.nan
                        for i, col in enumerate(test_prediction_reshaped[:,]):
                            current_wl_grid_ind = wl_inds_fitted[i]
                            gp_grid[current_wl_grid_ind, :] = col

                        if plot:
                            
                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            ax.plot_surface(x, y, test_prediction_reshaped)
                            if use_fluxes:
                                ax.set_zlabel('Fluxes')
                            else:
                                ax.invert_zaxis()
                                ax.set_zlabel('Magnitude')

                            ax.set_xlabel('Phase Grid')
                            ax.set_ylabel('Wavelengths')
                            plt.show()

                        if not plot:
                            use_for_template = 'y'
                        elif not interactive:
                            use_for_template = 'y'
                        if use_for_template == 'y':
                            gaussian_processes.append(gp_grid)
                    kernel_params.append(gaussian_process.kernel_.theta)

            #if not subtract_median and not subtract_polynomial: 
            #    return gaussian_processes, phase_grid, wl_grid
            #return None, phase_residuals, kernel_params
            return gaussian_processes, phase_grid, kernel_params, wl_grid

        else:
            all_phases, all_wls, all_mags, all_errs = self.process_dataset_for_gp_3d(filtlist, phasemin, phasemax, log_transform=log_transform, plot=plot, use_fluxes=use_fluxes)
            x = np.vstack((all_phases, all_wls)).T
            y = all_mags
            err = all_errs

            X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(x, y, err, test_size=test_size)
        
            ### Run the GP
            gaussian_process = GaussianProcessRegressor(
                kernel=self.kernel, alpha=Z_train, n_restarts_optimizer=10
            )
            gaussian_process.fit(X_train, Y_train)
            
            self.gaussian_process = gaussian_process
            
            return gaussian_process, X_test, None


    def predict_gp(self, filtlist, phasemin, phasemax, test_size=0.9, plot=False, log_transform=False, fit_residuals=False, set_to_normalize=False, subtract_median=False, subtract_polynomial=False, use_fluxes=False):

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
        if not subtract_median and not subtract_polynomial:#test_size is not None:
            ### Fitting sample of SNe altogether
            gaussian_process, X_test, kernel_params, _ = self.run_gp(filtlist, phasemin, phasemax, test_size=test_size, plot=plot, log_transform=log_transform, fit_residuals=fit_residuals, subtract_median=subtract_median, subtract_polynomial=subtract_polynomial, use_fluxes=use_fluxes)
            if plot:
                fig, ax = plt.subplots()
                
            if test_size is not None:
                for filt in filtlist:

                    test_times = np.linspace(min(X_test[:,0]), max(X_test[:,0]), 60)
                    if log_transform is not None:
                        test_waves = np.ones(len(test_times)) * np.log10(self.wle[filt])

                    else:
                        test_waves = np.ones(len(test_times)) * self.wle[filt]

                    test_prediction, std_prediction = gaussian_process.predict(np.vstack((test_times, test_waves)).T, return_std=True)

                    if plot:
                        if log_transform is not False:
                            test_times = np.exp(test_times) - log_transform
                        ax.plot(test_times, test_prediction, label=filt)
                        ax.fill_between(
                                test_times,
                                test_prediction - 1.96*std_prediction,
                                test_prediction + 1.96*std_prediction,
                                alpha=0.2,
                        )

                if plot:
                    ax.invert_yaxis()
                    ax.set_xlabel('Normalized Time [days]')
                    ax.set_ylabel('Normalized Magnitude')
                    plt.title('3D GP Fit')
                    plt.legend()
                    plt.show()

        else:
            ### We're fitting each SN individually and then median combining the full 2D GP
            print('Running GP')
            gaussian_processes, phase_grid, _, wl_grid = self.run_gp(filtlist, phasemin, 
                                                                  phasemax, plot=plot, 
                                                                  log_transform=log_transform, 
                                                                  fit_residuals=fit_residuals, 
                                                                  set_to_normalize=set_to_normalize, 
                                                                  subtract_median=subtract_median,
                                                                  subtract_polynomial=subtract_polynomial,
                                                                  use_fluxes=use_fluxes)
            print('Done running')
            median_gp = np.nanmedian(np.dstack(gaussian_processes), -1)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            if log_transform is not False:
                X, Y = np.meshgrid(np.exp(phase_grid) - log_transform, 10**wl_grid)
            else:
                X, Y = np.meshgrid(phase_grid, wl_grid)

            Z = median_gp

            ax.plot_surface(X, Y, Z)

            if use_fluxes:
                ax.set_zlabel('Fluxes')
            else:
                ax.invert_zaxis()
                ax.set_zlabel('Magnitude')

            ax.set_xlabel('Phase Grid')
            ax.set_ylabel('Wavelengths')
            plt.title('Final Median GP Fit')
            plt.show()

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