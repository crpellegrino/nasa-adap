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

warnings.filterwarnings('ignore')


class SN():
    """
    A Supernova object, taking a classification (i.e. SN II, SESN, FBOT, etc.),
    a subtype (i.e., SN IIP, SN IIb, SN Ibn, etc.), and a name (i.e. SN2022acko)
    """

    base_path = '/home/cmp5cr/nasa_adap/data/'


    def __init__(self, classification, subtype, name):

        self.classification = classification
        self.subtype  = subtype
        self.name = name
        self.data = {}
        self.peak_mjd = 0
        self.peak_mag = 0


    def __repr__(self):
        return self.name

    
    def load_swift_data(self):

        ### Load the Swift data for this object
        if not os.path.exists(os.path.join(self.base_path, self.classification, self.subtype, self.name, self.name+'_uvotB15.1.dat')):
            print('No Swift file for ', self.name)
            return

        df = pd.read_csv(os.path.join(self.base_path, self.classification, self.subtype, self.name, self.name+'_uvotB15.1.dat'), delim_whitespace=True, comment='#', names=['Filter', 'MJD', 'Mag', 'MagErr', '3SigMagLim', '0.98SatLim', 'Rate', 'RateErr', 'Ap', 'Frametime', 'Exp', 'Telapse'])

        for i, row in df.iterrows():
            if not np.isnan(row['Mag']):
                self.data.setdefault(row['Filter'], []).append({'mag': row['Mag'], 'err': row['MagErr'], 'mjd': row['MJD']})


    def load_json_data(self):

        ### Load data saved as a JSON file (ZTF, ATLAS, OpenSN, ASASSN)
        if not os.path.exists(os.path.join(self.base_path, self.classification, self.subtype, self.name)):
            print('No additional data files for ', self.name)
            return 

        dirfiles = os.listdir(os.path.join(self.base_path, self.classification, self.subtype, self.name))

        for f in dirfiles:
            if '.json' in f:
                print('Working with ', f)
                with open(os.path.join(self.base_path, self.classification, self.subtype, self.name, f), 'r') as jsonf:
                    d = json.load(jsonf)

                for filt, mag_list in d.items():
                    self.data.setdefault(filt, []).extend([mag for mag in mag_list if mag['err'] < 9999])


    def plot_data(self, only_this_filt='', shift=False):

        fig, ax = plt.subplots()

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

        for filt, mag_list in self.data.items():
            if only_this_filt and only_this_filt != filt:
                continue
            else:
                mjds = np.asarray([phot['mjd'] for phot in mag_list])
                mags = np.asarray([phot['mag'] for phot in mag_list])
                errs = np.asarray([phot['err'] for phot in mag_list])

                if shift:
                    mjds -= self.peak_mjd
                    mags -= self.peak_mag

                ax.errorbar(mjds, mags, yerr=errs, fmt='o', mec='black', color=colors.get(filt, 'k'), label=filt)

        plt.gca().invert_yaxis()
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('Apparent magnitude')
        plt.minorticks_on()
        plt.show()


    def fit_for_max(self, filt, shift_array=[-3, -2, -1, 0, 1, 2, 3], plot=False):
        """
        Takes as input arrays for MJD, mag, and err for a filter
        as well as the guess for the MJD of maximum and an array
        to shift the lightcurve over,
        and returns estimates of the peak MJD and mag at peak
        """
        mjd_array = np.asarray([phot['mjd'] for phot in self.data[filt]])
        mag_array = np.asarray([phot['mag'] for phot in self.data[filt]])
        err_array = np.asarray([phot['err'] for phot in self.data[filt]])

        if len(mag_array) == 0:
            return None, None

        guess_mjd_max = mjd_array[np.where((mag_array == min(mag_array)))[0]][0]

        ### Do this because the array might not be ordered
        inds_to_fit = np.where((mjd_array > guess_mjd_max - 10) & (mjd_array < guess_mjd_max + 10))

        if len(inds_to_fit[0]) < 4:
            #print('Select a wider date range')
            return None, None

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

            inds_to_fit = np.where((mjd_array > guess_mjd_max - 5 + sim_shift) & (mjd_array < guess_mjd_max + 5 + sim_shift))
            fit_mjds = mjd_array[inds_to_fit]
            fit_mags = mag_array[inds_to_fit]
            fit_errs = err_array[inds_to_fit]

            for i in range(len(fit_mjds)):
                simulated_points.append(np.random.normal(fit_mags[i], fit_errs[i]))
            fit = np.polyfit(fit_mjds, simulated_points, 2)
            f = np.poly1d(fit)
            fit_time = np.linspace(min(fit_mjds), max(fit_mjds), 100)

            if num % 25 == 0 and plot:
                plt.plot(fit_time, f(fit_time), color='black', linewidth=0.5)
            peak_mag = min(f(fit_time))
            peak_mags.append(peak_mag)
            peak_mjds.append(fit_time[np.argmin(f(fit_time))])

        if plot:
            plt.errorbar(mjd_array, mag_array, yerr=err_array, fmt='o', color='black')
            plt.errorbar(fit_mjds, fit_mags, yerr=fit_errs, fmt='o', color='blue')
            plt.errorbar(mean(peak_mjds), mean(peak_mags), xerr=stdev(peak_mjds), yerr=stdev(peak_mags), color='red', fmt='o')
            #plt.xlim(guess_mjd_max-10, guess_mjd_max+10)
            #plt.ylim(min(mag_array[inds_to_fit])-0.5, max(mag_array[inds_to_fit])+0.5)
            plt.gca().invert_yaxis()
            plt.show()

        self.peak_mjd = mean(peak_mjds)
        self.peak_mag = mean(peak_mags)
        self.peak_filt = filt

        return mean(peak_mjds), mean(peak_mags)


    def shift_to_max(self, filt, shift_array=[-3, -2, -1, 0, 1, 2, 3], plot=False):

        if not self.data:
            self.load_swift_data()
            self.load_json_data()

        if filt not in self.data.keys():
            return [], [], []


        if not self.peak_mjd > 0 and not self.peak_mag > 0:

            peak_mjd, peak_mag = self.fit_for_max(filt, shift_array=shift_array, plot=plot)
            while not peak_mjd:
                # Fitting didn't work, try another filter
                for newfilt in ['V', 'g', 'c', 'B', 'r', 'o', 'U', 'i', 'UVW1']:
                    if newfilt in self.data.keys() and newfilt != filt:
                        peak_mjd, peak_mag = self.fit_for_max(newfilt, shift_array=shift_array, plot=plot)
            
                if newfilt == 'UVW1' and not peak_mjd:
                    print('Reached last filter and could not fit for peak')
                    break
        
        if not self.peak_mag > 0:
            return [], [], []

        mjds = np.asarray([phot['mjd'] for phot in self.data[filt]]) - self.peak_mjd
        mags = np.asarray([phot['mag'] for phot in self.data[filt]]) - self.peak_mag
        errs = np.asarray([phot['err'] for phot in self.data[filt]])

        return mjds, mags, errs


    def log_transform_time(self, phases, phase_start=30):

        return np.log(phases + phase_start)


class Type():

    """
    A Type object, building a collection of all SNe of a given type (classification)
    """
    base_path = '/home/cmp5cr/nasa_adap/data/'
    sne = {}
    subtypes = []


    def __init__(self, classification):
        
        self.classification = classification

        self.get_subtypes()


    def get_subtypes(self):

        for d in os.listdir(os.path.join(self.base_path, self.classification)):
            if os.path.isdir(os.path.join(self.base_path, self.classification, d)):
                self.subtypes.append(d)


    def build_object_list(self):
    
        for subtype in self.subtypes:
            self.sne[subtype] = []
            for name in os.listdir(os.path.join(self.base_path, self.classification, subtype)):
                if os.path.isdir(os.path.join(self.base_path, self.classification, subtype, name)):
                    sn = SN(classification=self.classification, subtype=subtype, name=name)

                    self.sne[subtype].append(sn)


    def plot_all_lcs(self, filt):

        for subtype in self.subtypes:

            fig, ax = plt.subplots()
            
            ### First find max values for given filt
            for sn in self.sne[subtype]:

                mjds, mags, errs = sn.shift_to_max(filt)
                mjds = sn.log_transform_time(mjds)
                ax.errorbar(mjds, mags, yerr=errs, fmt='o', color='k')
            
            plt.gca().invert_yaxis()
            plt.title('{} + {}'.format(subtype, filt))
            plt.show()
        

class Fitter():

    """
    A Fitter object, fitting the light curves of a class (Type) of supernovae
    """
    
    def __init__(self, classification):

        self.classification = classification
        self.type = Type(classification)
        self.type.build_object_list()


class RBFKernel():

    """
    An RBFKernel object, to be used in GP fitting
    Allows users to define Kernel parameters such as length scale, bounds, etc.
    """

    def __init__(self, length_scale, length_scale_bounds):

        self.kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)


class WhiteNoiseKernel():

    """
    An RBFKernel object, to be used in GP fitting
    Allows users to define Kernel parameters such as noise level, bounds, etc.
    """

    def __init__(self, noise_level, noise_level_bounds):

        self.kernel = WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
    

def MaternKernel():

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

    def __init__(self, classification, subtype, kernel):
        super().__init__(classification)
        self.subtype = subtype
        self.kernel = kernel


    def process_dataset_for_gp(self, filt, phasemin, phasemax, log_transform=False):
        phases, mags, errs = np.asarray([]), np.asarray([]), np.asarray([])

        for sn in self.type.sne[self.subtype]:

            shifted_mjd, shifted_mag, err = sn.shift_to_max(filt)
            if len(shifted_mjd) == 0:
                continue
            if log_transform is not False:
                shifted_mjd = sn.log_transform_time(shifted_mjd, phase_start=log_transform)

            if log_transform is not False:
                inds_to_fit = np.where((shifted_mjd > np.log(phasemin+log_transform)) & (shifted_mjd < np.log(phasemax+log_transform)))[0]
            else:
                inds_to_fit = np.where((shifted_mjd > phasemin) & (shifted_mjd < phasemax))[0]
            
            phases = np.concatenate((phases, shifted_mjd[inds_to_fit]))
            mags = np.concatenate((mags, shifted_mag[inds_to_fit]))
            errs = np.concatenate((errs, err[inds_to_fit]))

        return phases.reshape(-1, 1), mags.reshape(-1, 1), errs.reshape(-1, 1)


    def run_gp(self, filt, phasemin, phasemax, test_size):

        phases, mags, errs = self.process_dataset_for_gp(filt, phasemin, phasemax)
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
        #kernel = 1 * RBF(length_scale=10.0, length_scale_bounds=(1.0, 1e2)) + WhiteKernel(noise_level=1., noise_level_bounds=(1e-10, 10.))
        gaussian_process = GaussianProcessRegressor(
            kernel=self.kernel, alpha=err_grid, n_restarts_optimizer=9
        )
        gaussian_process.fit(X_train, Y_train)

        self.gaussian_process = gaussian_process

        return gaussian_process, phases, mags, errs, err_grid


    def predict_gp(self, filt, phasemin, phasemax, test_size, plot=False):

        gaussian_process, phases, mags, errs, err_grid = self.run_gp(filt, phasemin, phasemax, test_size)

        mean_prediction, std_prediction = gaussian_process.predict(sorted(phases), return_std=True)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(sorted(phases), mean_prediction)
            ax.errorbar(phases, mags.reshape(-1), errs.reshape(-1), fmt='o', alpha=0.2)
            ax.fill_between(
                    sorted(phases.ravel()),
                    mean_prediction - 1.96*std_prediction,
                    mean_prediction + 1.96*std_prediction,
                    alpha=0.5
            )
            plt.gca().invert_yaxis()
            plt.show()


class GP3D(GP):

    """
    GP fit to all bands and epochs
    """
    wle = {'u': 3560,  'g': 4830, 'r': 6260, 'i': 7670, 'z': 8890, 'y': 9600, 'w':5985, 'Y': 9600,
           'U': 3600,  'B': 4380, 'V': 5450, 'R': 6410, 'G': 6730, 'E': 6730, 'I': 7980, 'J': 12200, 'H': 16300,
           'K': 21900, 'UVW2': 2030, 'UVM2': 2231, 'UVW1': 2634, 'F': 1516, 'N': 2267, 'o': 6790, 'c': 5330,
           'W': 33526, 'Q': 46028
    }

    def build_samples_3d(self, filt, phasemin, phasemax, log_transform=False):

        phases, mags, errs = self.process_dataset_for_gp(filt, phasemin, phasemax, log_transform=log_transform)

        min_phase, max_phase = sorted(phases)[0], sorted(phases)[-1]
        phase_grid = np.linspace(min_phase, max_phase, len(phases))
        phase_grid_space = (max_phase - min_phase)/len(phases)

        wl_grid = np.ones(len(phase_grid))
        err_grid = np.ones(len(phase_grid))
        for mjd in phase_grid:
            ind = np.where((phases < mjd + phase_grid_space/2) & (phases > mjd - phase_grid_space/2))[0]
            mags_at_this_phase = mags[ind]
            if len(mags_at_this_phase) > 1:
                std_mag = max(np.std(mags_at_this_phase), 0.01)
            elif len(mags_at_this_phase) == 1:
                std_mag = errs[ind]
            else:
                std_mag = 0.1

            err_grid[ind] = std_mag

            wl_grid[ind] = self.wle[filt]

        return phases, wl_grid, mags, err_grid


    def process_dataset_for_gp_3d(self, filtlist, phasemin, phasemax, log_transform=False):

        all_phases, all_wls, all_mags, all_errs = [], [], [], []

        for filt in filtlist:
            phases, wl_grid, mags, err_grid = self.build_samples_3d(filt, phasemin, phasemax, log_transform=log_transform)

            all_phases = np.concatenate((all_phases, phases.flatten()))
            all_wls = np.concatenate((all_wls, wl_grid.flatten()))
            all_mags = np.concatenate((all_mags, mags.flatten()))
            all_errs = np.concatenate((all_errs, err_grid.flatten()))

        return all_phases, all_wls, all_mags, all_errs


    def run_gp(self, filtlist, phasemin, phasemax, test_size, log_transform=False):

        all_phases, all_wls, all_mags, all_errs = self.process_dataset_for_gp_3d(filtlist, phasemin, phasemax, log_transform=log_transform)
        x = np.vstack((all_phases, all_wls)).T
        y = all_mags
        err = all_errs

        X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(x, y, err, test_size=test_size)
    
        ### Run the GP
        #kernel = 1 * RBF(length_scale=[10.0, 500.0], length_scale_bounds=(1.0, 2e3)) + WhiteKernel(noise_level=1., noise_level_bounds=(1e-10, 10.))
        gaussian_process = GaussianProcessRegressor(
            kernel=self.kernel, alpha=Z_train, n_restarts_optimizer=10, normalize_y=True
        )
        gaussian_process.fit(X_train, Y_train)
        
        self.gaussian_process = gaussian_process
        
        return gaussian_process, X_train, X_test, Y_train, Y_test, Z_train, Z_test


    def predict_gp(self, filtlist, phasemin, phasemax, test_size, plot=False, log_transform=False):

        gaussian_process, X_train, X_test, Y_train, Y_test, Z_train, Z_test = self.run_gp(filtlist, phasemin, phasemax, test_size, log_transform=log_transform)

        if plot:
            fig, ax = plt.subplots()

        for filt in filtlist:

            test_times = np.linspace(min(X_test[:,0]), max(X_test[:,0]), 60)
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
                        alpha=0.5,
                )

        if plot:
            ax.invert_yaxis()
            plt.legend()
            plt.show()



kernel = RBFKernel([np.log(10.0), 500.0], (0.1, 2.0e3)).kernel + WhiteNoiseKernel(1., (1e-10, 10.)).kernel
#kernel = Matern([10.0, 500.0], (1., 2.0e3), 2.5)
gp = GP3D('SNII', 'SNIIP', kernel)
gp.predict_gp(['UVW2', 'UVM2', 'UVW1', 'U', 'B', 'V'], -20, 50, 0.9, plot=True, log_transform=30)

print(gp.gaussian_process.kernel_)
