import numpy as np
import matplotlib.pyplot as plt
from caat import SN
from caat.utils import query_svo_service, bin_spec
import logging
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataCube:

    def __init__(self, name: str = None, data: dict = None, shift: bool = False):

        self.sn = SN(name=name, data=data)
        self.sn.load_json_data()
        self.sn.load_swift_data()
        if shift:
            for filt in self.sn.data.keys():
                self.sn.shift_to_max(filt)
        self.sn.convert_to_fluxes()
        self.shift = shift

    def construct_cube(self):

        if self.shift:
            data = self.sn.shifted_data
        else:
            data = self.sn.data

        self.cube = np.asarray(
            [
                np.hstack(
                    [[d['mjd'] for d in data[filt]] for filt in data.keys()]
                ),
                np.hstack(
                    [np.repeat([self.sn.wle[filt] * (1 + self.sn.info.get('z', 0.0))], len(data[filt])) for filt in data.keys()]
                ),
                np.hstack(
                    [[10**d['flux'] for d in data[filt]] for filt in data.keys()]
                ),
                np.hstack(
                    [[d['fluxerr']*10**d['flux'] for d in data[filt]] for filt in data.keys()]
                ),
                np.hstack(
                    [np.repeat([filt], len(data[filt])) for filt in data.keys()]
                )
            ], dtype=object
        )

    def deconstruct_cube(self):
        """
        Takes as input a data cube and turns it into a dictionary
        of photometry to be used in GP fitting
        Assumes a data cube of size (n, 5)
        """
        data = {}
        for i in range(len(self.cube[:])):
            data.setdefault(self.cube[4][i], []).append(
                {
                    'mjd': self.cube[0][i], 
                    'wle': self.cube[1][i], 
                    'flux': self.cube[2][i], 
                    'fluxerr': self.cube[3][i]
                }
            )
        
        if self.shift:
            self.sn.shifted_data = data
        else:
            self.sn.data = data

    def plot_cube(self):

        if not hasattr(self, 'cube'):
            self.construct_cube()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.errorbar(self.cube[0], self.cube[1], self.cube[2], zerr=self.cube[3], fmt='o')
        ax.set_xlabel("Phase Grid")
        ax.set_ylabel("Wavelengths")
        ax.set_zlabel("Fluxes")

        plt.tight_layout()
        plt.show()

    def measure_flux_in_filter(self, plot: bool = False):

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
                               'i': 'ZTF'
                            }

        self.construct_cube()

        trans_fns = {}
        for filt in np.unique(self.cube[4]):
            trans_wl, trans_eff = query_svo_service(filt_tel_conversion[filt], filt)
            trans_eff /= max(trans_eff)
            # Get min and max wavelength for this filter, let's define it as where eff < 10%
            center_of_filt = trans_wl[np.argmax(trans_eff)]
            tail_wls = trans_wl[np.where((trans_eff < 0.025))[0]]
            min_trans_wl = np.max(tail_wls[np.where((tail_wls < center_of_filt))[0]])
            max_trans_wl = np.min(tail_wls[np.where((tail_wls > center_of_filt))[0]])
            trans_fns[filt] = {
                'wl': trans_wl, 
                'eff': trans_eff,
                'min_wl': min_trans_wl - 500.,
                'max_wl': max_trans_wl + 500.
            }

        for phase in np.arange(min(self.cube[0]), max(self.cube[0]), 1.0):
            current_lc_inds = np.where((abs(self.cube[0] - phase) <= 0.5))[0]

            if len(current_lc_inds) > 0 and len(self.cube[4][current_lc_inds]) > 1: # Have data in at least two filters at this epoch
                bluest_filt = np.unique(self.cube[4][current_lc_inds][np.argmin(self.cube[1][current_lc_inds])])[0]
                reddest_filt = np.unique(self.cube[4][current_lc_inds][np.argmax(self.cube[1][current_lc_inds])])[0]

                current_lc = self.cube[2][current_lc_inds]
                current_lc = np.concatenate(([0.0], current_lc, [0.0]))
                current_lc_err = np.concatenate(([0.0], self.cube[3][current_lc_inds], [0.0]))
                current_lc_wls = np.concatenate((
                    [trans_fns[bluest_filt]['min_wl'] * (1 + self.sn.info.get('z', 0.0))],
                    self.cube[1][current_lc_inds],
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

                while any(errors > 1.5) and n < 100:

                    residuals = []

                    for i, filt in enumerate(self.cube[4][current_lc_inds]):

                        ### Bin the transmission curve and SED to common resolution
                        binned_trans_wl, binned_trans_eff = bin_spec(trans_fns[filt]['wl'], trans_fns[filt]['eff'], wl_grid)

                        if n == 0 and plot:
                            plt.plot(binned_trans_wl, binned_trans_eff*max(interp(wl_grid)))

                        ### Get overlap between the filter and the SED
                        sed_inds = np.where((wl_grid >= min(binned_trans_wl)) & (wl_grid <= max(binned_trans_wl)))[0]
                        interp_filt = interp1d(binned_trans_wl, binned_trans_eff)
                        interp_trans_wl = np.linspace(wl_grid[sed_inds[0]], wl_grid[sed_inds[-1]], len(sed_inds))
                        interp_trans_eff = interp_filt(interp_trans_wl)
                        
                        flux = np.nansum(binned_sed[sed_inds] *
                                        interp_trans_eff
                        ) / len(interp_trans_eff)
                        implied_central_wl = interp_trans_wl[np.argmax(binned_sed[sed_inds] * interp_trans_eff)]
                        #convolved_sed = binned_sed[sed_inds] * interp_trans_eff
                        #implied_central_wl = interp_trans_wl[np.argsort(convolved_sed)[len(convolved_sed)//2]]
                        real_flux_inds = np.where((self.cube[4][current_lc_inds] == filt))[0]+1

                        if len(real_flux_inds) > 1:
                            real_flux = np.average(current_lc[real_flux_inds])
                        else:
                            real_flux = current_lc[real_flux_inds][0]

                        error = max(flux/real_flux, real_flux/flux)
                        resid = flux / real_flux
                        print(f'Filter: {filt}, convolved flux: {flux}, measured flux: {real_flux}, error: {error}')
                        print(f'Filter: {filt}, real wavelength: {current_lc_wls[i+1]}, warped wl: {implied_central_wl}')
                        errors[i] = error
                        residuals.append(resid)
                        measured_flux[i+1] = flux
                        measured_wls[i+1] = implied_central_wl

                    if any(errors > 1.5):

                        ### Make a residual interpolated SED from the convolved fluxes at the 
                        ### implied wavelengths, warp the SED using this residual, and rerun the loop

                        residual_interp = interp1d(measured_wls, np.concatenate(([0.0], residuals, [0.0])))
                        residual = residual_interp(wl_grid)
                        binned_sed /= residual
                        
                        n += 1
                        if n == 100:
                            print('Couldnt iterate to match flux!')

                if plot:
                    plt.errorbar(current_lc_wls, current_lc, yerr=current_lc_err, fmt='o', alpha=0.3)
                    plt.errorbar(measured_wls, measured_flux, yerr=current_lc_err, fmt='o')
                    plt.plot(wl_grid, interp1d(measured_wls, measured_flux, kind='linear')(wl_grid))
                    plt.show()

                if n < 100:
                    ### Put the new effective wavelengths back into the data cube in the right spots
                    self.cube[1][current_lc_inds] = measured_wls[1:-1]
