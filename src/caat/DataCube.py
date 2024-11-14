import numpy as np
import matplotlib.pyplot as plt
from caat import SN
import logging
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataCube:

    def __init__(self, name: str = None, data: dict = None):

        self.sn = SN(name=name, data=data)
        self.sn.load_json_data()
        self.sn.load_swift_data()
        for filt in self.sn.data.keys():
            self.sn.shift_to_max(filt)
        self.sn.convert_to_fluxes()

    def construct_cube(self):

        data = self.sn.shifted_data

        self.cube = np.asarray(
            [
                np.hstack(
                    [[d['mjd'] for d in data[filt]] for filt in data.keys()]
                ),
                np.hstack(
                    [np.repeat([self.sn.wle[filt] * (1 + self.sn.info.get('z', 0.0))], len(data[filt])) for filt in data.keys()]
                ),
                np.hstack(
                    [[d['flux'] for d in data[filt]] for filt in data.keys()]
                ),
                np.hstack(
                    [[d['fluxerr'] for d in data[filt]] for filt in data.keys()]
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

        self.construct_cube()

        for phase in np.arange(min(self.cube[0]), max(self.cube[0]), 1.0):
            current_lc_inds = np.where((abs(self.cube[0] - phase) <= 0.5))[0]
            print(phase, self.cube[4][current_lc_inds])

            if len(current_lc_inds) > 0:
                bluest_filt = np.unique(self.cube[4][current_lc_inds][np.argmin(self.cube[1][current_lc_inds])])[0]
                reddest_filt = np.unique(self.cube[4][current_lc_inds][np.argmax(self.cube[1][current_lc_inds])])[0]
                
                current_lc = np.concatenate(([0.0], self.cube[2][current_lc_inds], [0.0]))
                current_lc_err = np.concatenate(([0.0], self.cube[3][current_lc_inds], [0.0]))
                current_lc_wls = np.concatenate((
                    [self.sn.wle[bluest_filt] * (1 + self.sn.info.get('z', 0.0)) - 500.0], 
                    self.cube[1][current_lc_inds],
                    [self.sn.wle[reddest_filt] * (1 + self.sn.info.get('z', 0.0)) + 500.0]   
                ))

                ### Construct SED by interpolating over this LC
                interp = interp1d(current_lc_wls, current_lc, kind='linear')

                plt.errorbar(current_lc_wls, current_lc, yerr=current_lc_err, fmt='o')
                interp_wls = np.linspace(min(current_lc_wls), max(current_lc_wls), 20)
                plt.plot(interp_wls, interp(interp_wls))
                plt.show()