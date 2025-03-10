import numpy as np
import matplotlib.pyplot as plt
from caat import SN
from caat.utils import query_svo_service, bin_spec
import logging
from scipy.interpolate import interp1d
import pandas as pd
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataCube:
    """"
    Class with routines to handle pre-processing of data
    Runs SN methods such as `load_<>_data`, `shift_to_max`,
    and `convert_to_fluxes` to initialize data set.
    Also optionally performs iterative warping on SED at each
    epoch to better match observed photometry.
    Constructs a Pandas dataframe with this information to be
    saved, read, and manipulated as part of the fitting routine
    """
    def __init__(
            self, 
            sn: SN = None, 
            name: str | None = None, 
            data: dict | None = None, 
        ):

        if sn:
            self.sn = sn
        else:
            sn = SN(name=name, data=data)
            self.sn = sn

        if not self.sn.data:
            self.sn.load_json_data()
            self.sn.load_swift_data()
        self.sn.correct_for_galactic_extinction()
        for filt in list(self.sn.data.keys()):
            self.sn.shift_to_max(filt)
            if filt not in self.sn.wle.keys():
                del self.sn.data[filt]
        self.sn.convert_all_mags_to_fluxes()

    def construct_cube(self):

        if (
            not any(
                [
                    filt for filt in list({filt for filt in list(self.sn.data.keys()) + list(self.sn.shifted_data.keys())})
                ]
            )
        or (
            not self.sn.data or not self.sn.shifted_data
            )
        ):
            cube = np.asarray(
                [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    []
                ]
            )

        else:       
            cube = np.array(
                [
                    np.hstack(
                        [[d['mjd'] for d in self.sn.data[filt]] for filt in self.sn.data.keys()]
                    ),
                    np.hstack(
                        [[d['mjd'] for d in self.sn.shifted_data[filt]] for filt in self.sn.shifted_data.keys()]
                    ),
                    np.hstack(
                        [np.repeat([filt], len(self.sn.data[filt])) for filt in self.sn.data.keys()]
                    ),
                    np.hstack(
                        [np.repeat([filt], len(self.sn.shifted_data[filt])) for filt in self.sn.shifted_data.keys()]
                    ),
                    np.hstack(
                        [np.repeat([self.sn.wle[filt] * (1 + self.sn.info.get('z', 0.0))], len(self.sn.data[filt])) for filt in self.sn.data.keys()]
                    ),
                    np.hstack(
                        [np.repeat([self.sn.wle[filt] * (1 + self.sn.info.get('z', 0.0))], len(self.sn.shifted_data[filt])) for filt in self.sn.shifted_data.keys()]
                    ),
                    np.hstack(
                        [[d['flux'] for d in self.sn.data[filt]] for filt in self.sn.data.keys()]
                    ),
                    np.hstack(
                        [[d['flux'] for d in self.sn.shifted_data[filt]] for filt in self.sn.shifted_data.keys()]
                    ),
                    np.hstack(
                        [[d['fluxerr']*d['flux'] for d in self.sn.data[filt]] for filt in self.sn.data.keys()]
                    ),
                    np.hstack(
                        [[d['fluxerr'] for d in self.sn.shifted_data[filt]] for filt in self.sn.shifted_data.keys()]
                    ),
                    np.hstack(
                        [[d['mag'] for d in self.sn.data[filt]] for filt in self.sn.data.keys()]
                    ),
                    np.hstack(
                        [[d['mag'] for d in self.sn.shifted_data[filt]] for filt in self.sn.shifted_data.keys()]
                    ),
                    np.hstack(
                        [[d['err'] for d in self.sn.data[filt]] for filt in self.sn.data.keys()]
                    ),
                    np.hstack(
                        [[d.get('nondetection', False) for d in self.sn.data[filt]] for filt in self.sn.data.keys()]
                    )
                ], dtype=object
            )

        if len(cube.shape) != 2:
            logger.warning(f"WARNING: Construct cube failed for {self.sn.name}")
            cube = np.asarray(
                [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    []
                ]
            )

        self.cube = pd.DataFrame(
            data=cube.T, 
            columns=[
                "MJD", 
                "Phase", 
                "Filter", 
                "ShiftedFilter", 
                "Wavelength",
                "ShiftedWavelength", 
                "Flux", 
                "ShiftedFlux", 
                "Fluxerr",
                "ShiftedFluxerr", 
                "Mag", 
                "ShiftedMag", 
                "Magerr", 
                "Nondetection"
            ]
        ).dropna()

    def deconstruct_cube(self):
        """
        Takes as input a data cube and turns it into a dictionary
        of photometry to be used in GP fitting
        Assumes a data cube of size (n, 5)
        """
        #TODO: Make this compatible with change to pandas df
        pass
        # data = {}
        # for i in range(len(self.cube[0,:])):
        #     data.setdefault(self.cube[4][i], []).append(
        #         {
        #             'mjd': self.cube[0][i], 
        #             'wle': self.cube[1][i], 
        #             'flux': np.log10(self.cube[2][i]), 
        #             'fluxerr': self.cube[3][i]/10**(self.cube[2][i]),
        #             'mag': self.cube[5][i],
        #             'err': self.cube[6][i],
        #             'nondetection': self.cube[7][i]
        #         }
        #     )
        
        # if self.shift:
        #     self.sn.shifted_data = data
        # else:
        #     self.sn.data = data

    def plot_cube(self):

        if not hasattr(self, 'cube'):
            self.construct_cube()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.errorbar(self.cube['Phase'], self.cube['Wavelength'], self.cube['Flux'], zerr=self.cube['Fluxerr'], fmt='o')
        ax.set_xlabel("Phase Grid")
        ax.set_ylabel("Wavelengths")
        ax.set_zlabel("Fluxes")

        plt.tight_layout()
        plt.show()

    def measure_flux_in_filter(
            self, 
            niter: int = 100,
            plot: bool = False, 
            verbose: bool = False,
            save: bool = False
        ):

        #TODO: Encorporate niter
        #TODO: Diagnostics on observed photometry versus mangled photometry
        #      and trends across filters

        if save and os.path.exists(
            os.path.join(
                    self.sn.base_path,
                    self.sn.classification,
                    self.sn.subtype,
                    self.sn.name,
                    self.sn.name + "_datacube_mangled.csv",
            )
        ):
            logger.info(f"Already saved mangled datacube for {self.sn.name}, skipping")
            return

        filt_tel_conversion = {'UVW2': 'Swift',
                               'UVM2': 'Swift',
                               'UVW1': 'Swift',
                               'U': 'Swift',
                               'c': 'Atlas',
                               'o': 'Atlas',
                               'B': 'Swift',
                               'V': 'Swift',
                               'g': 'ZTF',
                               'r': 'ZTF',
                               'i': 'ZTF',
                               'R': 'CTIO',
                               'I': 'CTIO',
                               'J': 'CTIO',
                               'H': 'CTIO',
                               'K': 'CTIO',
                               'Y': 'DECam',
                               'u': 'DECam',
                               'G': 'GAIA',
                               'y': 'PAN-STARRS',
                               'z': 'PAN-STARRS',
                               'w': 'PAN-STARRS'
                            }

        self.construct_cube()

        if len(self.cube['MJD']) == 0: # No data, so return nothing
            return

        trans_fns = {}
        filts_to_ignore = []
        for filt in np.unique(self.cube["Filter"]):
            trans_wl, trans_eff = query_svo_service(filt_tel_conversion[filt.replace("'", "")], filt.replace("'", ""))
            trans_eff /= max(trans_eff)
            # Get min and max wavelength for this filter, let's define it as where eff < 10%
            center_of_filt = trans_wl[np.argmax(trans_eff)]
            tail_wls = trans_wl[np.where((trans_eff < 0.025))[0]]
            try:
                min_trans_wl = np.max(tail_wls[np.where((tail_wls < center_of_filt))[0]])
                max_trans_wl = np.min(tail_wls[np.where((tail_wls > center_of_filt))[0]])
                trans_fns[filt] = {
                    'wl': trans_wl, 
                    'eff': trans_eff,
                    'min_wl': min_trans_wl - 500.,
                    'max_wl': max_trans_wl + 500.
                }
            except:
                logger.warning(f"Warning: transmission function failed for {filt}, ignoring")
                filts_to_ignore.append(filt)

        inds_to_drop = self.cube.loc[self.cube['Filter'].isin(filts_to_ignore)].index
        self.cube = self.cube.drop(inds_to_drop).reset_index(drop=True)

        for phase in np.arange(min(self.cube['Phase']), max(self.cube['Phase']), 1.0):
            current_lc_inds = np.where((abs(self.cube['Phase'] - phase) <= 0.5))[0]

            if len(current_lc_inds) > 0 and len({filt for filt in self.cube['Filter'][current_lc_inds]}) > 1: # Have data in at least two filters at this epoch
                current_lc_cube = self.cube[abs(self.cube['Phase'] - phase) <= 0.5]

                bluest_wavelength = np.min(current_lc_cube['Wavelength'].values)
                reddest_wavelength = np.max(current_lc_cube['Wavelength'].values)
                bluest_filt = np.unique(current_lc_cube[current_lc_cube['Wavelength'] == bluest_wavelength]['Filter'])[0]
                reddest_filt = np.unique(current_lc_cube[current_lc_cube['Wavelength'] == reddest_wavelength]['Filter'])[0]

                current_lc = current_lc_cube['Flux'].values
                current_lc = np.concatenate(([0.0], current_lc, [0.0]))
                current_lc_err = np.concatenate(([0.0], current_lc_cube['Fluxerr'], [0.0]))
                current_lc_wls = np.concatenate((
                    [trans_fns[bluest_filt]['min_wl'] * (1 + self.sn.info.get('z', 0.0))],
                    current_lc_cube['Wavelength'],
                    [trans_fns[reddest_filt]['max_wl'] * (1 + self.sn.info.get('z', 0.0))]
                ))
                wl_grid = np.linspace(current_lc_wls[0], current_lc_wls[-1], 50)

                errors = np.ones(len(current_lc_inds)) * 100.0
                n = 0
                #central_wls = np.copy(current_lc_wls[1:-1])
                measured_wls = np.copy(current_lc_wls)
                measured_flux = np.copy(current_lc)

                ### Construct SED by interpolating over this LC
                interp = interp1d(measured_wls, measured_flux, kind='linear')
                binned_sed = interp(wl_grid)

                #while any(errors > 1.5) and n < 100:
                for i in range(100):

                    if all(errors <= 1.5):
                        break

                    residuals = []

                    for i, filt in enumerate(current_lc_cube['Filter']):

                        ### Bin the transmission curve and SED to common resolution
                        binned_trans_wl, binned_trans_eff = bin_spec(trans_fns[filt]['wl'], trans_fns[filt]['eff'], wl_grid)

                        if n == 0 and plot:
                            plt.plot(binned_trans_wl, binned_trans_eff*max(interp(wl_grid)))

                        ### Get overlap between the filter and the SED
                        sed_inds = np.where((wl_grid >= min(binned_trans_wl)) & (wl_grid <= max(binned_trans_wl)))[0]
                        if len(sed_inds) > 0:
                            interp_filt = interp1d(binned_trans_wl, binned_trans_eff)
                            interp_trans_wl = np.linspace(wl_grid[sed_inds[0]], wl_grid[sed_inds[-1]], len(sed_inds))
                            interp_trans_eff = interp_filt(interp_trans_wl)
                            
                            flux = np.nansum(binned_sed[sed_inds] *
                                            interp_trans_eff
                            ) / len(interp_trans_eff)
                            implied_central_wl = interp_trans_wl[np.argmax(binned_sed[sed_inds] * interp_trans_eff)]
                            #convolved_sed = binned_sed[sed_inds] * interp_trans_eff
                            #implied_central_wl = interp_trans_wl[np.argsort(convolved_sed)[len(convolved_sed)//2]]
                            real_flux_inds = np.where((current_lc_cube['Filter'] == filt))[0]+1

                            if len(real_flux_inds) > 1:
                                real_flux = np.average(current_lc[real_flux_inds])
                            else:
                                real_flux = current_lc[real_flux_inds][0]

                            try:
                                error = max(flux/real_flux, real_flux/flux)
                            except ZeroDivisionError:
                                error = 100

                            try:
                                resid = flux / real_flux
                            except ZeroDivisionError:
                                resid = 100

                            if verbose:
                                print(f'Filter: {filt}, convolved flux: {flux}, measured flux: {real_flux}, error: {error}')
                                print(f'Filter: {filt}, real wavelength: {current_lc_wls[i+1]}, warped wl: {implied_central_wl}')
                            errors[i] = error
                            residuals.append(resid)
                            measured_flux[i+1] = flux
                            measured_wls[i+1] = implied_central_wl

                        else:
                            ### No overlap between SED and filter, so break
                            n = 100


                    if any(errors > 1.5):

                        ### Make a residual interpolated SED from the convolved fluxes at the 
                        ### implied wavelengths, warp the SED using this residual, and rerun the loop
                        if n < 100:
                            residual_interp = interp1d(measured_wls, np.concatenate(([0.0], residuals, [0.0])))
                            residual = residual_interp(wl_grid)
                            binned_sed /= residual
                            
                            n += 1
                            if any(errors > 1e3):
                                n = 100

                        if n == 100 and verbose:
                            print('Couldnt iterate to match flux!')

                if plot:
                    plt.errorbar(current_lc_wls, current_lc, yerr=current_lc_err, fmt='o', alpha=0.3)
                    plt.errorbar(measured_wls, measured_flux, yerr=current_lc_err, fmt='o')
                    plt.plot(wl_grid, interp1d(measured_wls, measured_flux, kind='linear')(wl_grid))
                    plt.show()

                if n < 100:
                    ### Put the new effective wavelengths back into the data cube in the right spots
                    current_lc_cube['Wavelength'] = measured_wls[1:-1]
                    current_lc_cube['ShiftedWavelength'] = measured_wls[1:-1]
                    self.cube.update(current_lc_cube)

        if verbose:
            print(f'Done warping SED for {self.sn.name}')

        if save:
            self.cube.to_csv(
                os.path.join(
                    self.sn.base_path,
                    self.sn.classification,
                    self.sn.subtype,
                    self.sn.name,
                    self.sn.name + "_datacube_mangled.csv",
                ),
            )
