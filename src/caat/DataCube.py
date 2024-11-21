import numpy as np
import matplotlib.pyplot as plt
from caat import SN
from caat.utils import query_svo_service, bin_spec
import logging
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataCube:

    def __init__(self, name: str = None, data: dict = None):

        self.sn = SN(name=name, data=data)
        self.sn.load_json_data()
        self.sn.load_swift_data()
        #for filt in self.sn.data.keys():
        #    self.sn.shift_to_max(filt)
        self.sn.convert_to_fluxes()

    def construct_cube(self):

        data = self.sn.data #self.sn.shifted_data

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

    def plot_cube(self):

        self.construct_cube()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.errorbar(self.cube[0], self.cube[1], self.cube[2], zerr=self.cube[3], fmt='o')
        ax.set_xlabel("Phase Grid")
        ax.set_ylabel("Wavelengths")
        ax.set_zlabel("Fluxes")

        plt.tight_layout()
        plt.show()

    def measure_flux_in_filter(self):

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

        for phase in np.arange(min(self.cube[0]), max(self.cube[0]), 1.0):
            current_lc_inds = np.where((abs(self.cube[0] - phase) <= 0.5))[0]

            trans_fns = {}

            if len(current_lc_inds) > 0:
                bluest_filt = np.unique(self.cube[4][current_lc_inds][np.argmin(self.cube[1][current_lc_inds])])[0]
                reddest_filt = np.unique(self.cube[4][current_lc_inds][np.argmax(self.cube[1][current_lc_inds])])[0]

                current_lc = self.cube[2][current_lc_inds]
                current_lc = np.concatenate(([0.0], current_lc, [0.0]))
                current_lc_err = np.concatenate(([0.0], self.cube[3][current_lc_inds], [0.0]))
                current_lc_wls = np.concatenate((
                    [self.sn.wle[bluest_filt] * (1 + self.sn.info.get('z', 0.0)) - 500.0], 
                    self.cube[1][current_lc_inds],
                    [self.sn.wle[reddest_filt] * (1 + self.sn.info.get('z', 0.0)) + 500.0]   
                ))
                wl_grid = np.linspace(current_lc_wls[0], current_lc_wls[-1], 50)
                
                for i, filt in enumerate(self.cube[4][current_lc_inds]):
                    trans_wl, trans_eff = query_svo_service(filt_tel_conversion[filt], filt)
                    trans_eff /= max(trans_eff)
                    trans_fns[filt] = {'wl': trans_wl, 'eff': trans_eff}

                    error = 100.0
                    n = 0

                    #TODO: All filters at same time
                    central_wl = current_lc_wls[i+1]

                    while error > 2.5 and n < 100:

                        ### Construct SED by interpolating over this LC
                        interp = interp1d(current_lc_wls, current_lc, kind='linear')

                        ### Bin the transmission curve and SED to common resolution
                        binned_trans_wl, binned_trans_eff = bin_spec(trans_wl, trans_eff, wl_grid)
                        binned_sed = interp(wl_grid)

                        ### Get overlap between the filter and the SED
                        sed_inds = np.where((wl_grid >= min(binned_trans_wl)) & (wl_grid <= max(binned_trans_wl)))[0]
                        interp_filt = interp1d(binned_trans_wl, binned_trans_eff)
                        interp_trans_wl = np.linspace(wl_grid[sed_inds[0]], wl_grid[sed_inds[-1]], len(sed_inds))
                        interp_trans_eff = interp_filt(interp_trans_wl)
                            
                        flux = np.nansum(binned_sed[sed_inds] *
                                        interp_trans_eff
                        ) / len(interp_trans_eff)
                        real_flux_inds = np.where((self.cube[4][current_lc_inds] == filt))[0]+1

                        if len(real_flux_inds) > 1:
                            real_flux = np.average(current_lc[real_flux_inds])
                        else:
                            real_flux = current_lc[real_flux_inds][0]

                        error = max(flux/real_flux, real_flux/flux)
                        print(f'Filter: {filt}, convolved flux: {flux}, measured flux: {current_lc[real_flux_inds]}, error: {error}')

                        if error > 2.5:
                            current_lc_wls[i+1] = central_wl + np.random.randint(-400, 400)
                            n += 1
                            if n == 100:
                                print('Couldnt iterate to match flux!')
                                plt.plot(binned_trans_wl, binned_trans_eff*max(interp(wl_grid)))
                
                        else:
                            plt.plot(binned_trans_wl, binned_trans_eff*max(interp(wl_grid)))

                plt.errorbar(current_lc_wls, current_lc, yerr=current_lc_err, fmt='o')
                plt.plot(wl_grid, interp1d(current_lc_wls, current_lc, kind='linear')(wl_grid))
                plt.show()
