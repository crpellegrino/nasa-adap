import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from .DataCube import DataCube
from .SN import SN
from .SNCollection import SNCollection, SNType
from caat.utils import ROOT_DIR, WLE, FILT_TEL_CONVERSION, bin_spec, query_svo_service

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SNModel:
    """
    A Supernova Model object. Saves the outputted 3D SED surface from the
    Gaussian process regression routine for a SN object or a collection of SNe.
    Provides routines for saving and loading the final GP fits, as well as 
    for extracting 2D light curves or SEDs from the interpolated surfaces.
    """
    def __init__(
        self, 
        surface: GaussianProcessRegressor | str | None = None, 
        template_mags: np.ndarray | str | None = None,
        phase_grid: np.ndarray | None = None,
        phase_bounds: tuple | None = None,
        wl_grid: np.ndarray| None = None,
        filters_fit: list[str] | None = None,
        sn: SN | None = None, 
        sncollection: SNType | SNCollection | None = None,
        norm_set: SNType | SNCollection | None = None,
        log_transform: int | float | bool = False
    ):
        if not sn and not sncollection:
            raise ValueError("Need to specify either a SN or SNCollection for this model!")
        
        self.base_path = os.path.join(ROOT_DIR, "data/final_models/")

        if sn:
            self.sn = sn
        if sncollection:
            self.collection = sncollection
        if norm_set:
            self.norm_set = norm_set
        
        if type(surface) == str:
            self.load_gp(surface)
        else:
            self.surface = surface

        if type(template_mags) == str:
            self.load_template(template_mags)
        else:
            self.template = template_mags
        
        if phase_grid is None and phase_bounds is not None and self.template is not None:
            self.phase_grid = np.linspace(phase_bounds[0], phase_bounds[1], self.template.shape[0])
            self.min_phase, self.max_phase = phase_bounds
        else:
            self.phase_grid = phase_grid
            self.min_phase, self.max_phase = int(min(phase_grid)), int(max(phase_grid))

        if wl_grid is None and filters_fit is not None and self.template is not None:
            self.wl_grid = np.linspace(
                min(WLE[f] for f in filters_fit) - 500,
                max(WLE[f] for f in filters_fit) + 500,
                self.template.shape[1],
            )
        else:
            self.wl_grid = wl_grid
        self.min_wl, self.max_wl = int(min(self.wl_grid)), int(max(self.wl_grid))
        self.filters = filters_fit
        
        self.log_transform = log_transform

    def save_gp(self, filename: str = None):
        if not self.surface:
            logger.warning("Need to instantiate the SNModel class with a GP model to save it!")
            return
        if not filename:
            try:
                filename = f"{self.sn.name}_GP_model_{self.min_phase}_{self.max_phase}_{''.join(filt for filt in self.filters)}.pkl"
            except:
                filename = ''
                if hasattr(self.collection, "type"):
                    filename += self.collection.type
                    if hasattr(self.collection, "subtype"):
                        filename += f"_{self.collection.subtype}"
                else:
                    filename += f"{', '.join(sn.name for sn in self.collection.sne)}"
                
                filename += f"_GP_model_{self.min_phase}_{self.max_phase}_{','.join(filt for filt in self.filters)}.pkl"

        with open(os.path.join(self.base_path, filename), 'wb') as f:
            pickle.dump(self.surface, f)

    def save_template(self, filename: str = None):
        if self.template is None:
            logger.warning("Need to instantiate the SNModel class with template mags to save it!")
            return
        if not filename:
            filename = ''
            if hasattr(self.norm_set, "type"):
                filename += self.norm_set.type
                if hasattr(self.norm_set, "subtype"):
                    filename += f"_{self.norm_set.subtype}"
            else:
                filename += f"{', '.join(sn.name for sn in self.norm_set.sne)}"
            
            filename += f"_template_{self.min_phase}_{self.max_phase}_{','.join(filt for filt in self.filters)}.csv"

        np.savetxt(os.path.join(self.base_path, filename), self.template, delimiter=",")

    def load_gp(self, filename: str):
        """
        Load a GPR object from a saved file
        """
        if not os.path.exists(os.path.join(self.base_path, filename)):
            raise ValueError("No model file exists by that name!")
        
        with open(os.path.join(self.base_path, filename), 'rb') as f:
            surface = pickle.load(f)

        if hasattr(self, "surface") and self.surface is not None:
            # Warn that we're overwriting the existing model
            logger.warning("Overwriting existing GP model")

        self.surface = surface

    def load_template(self, filename: str):
        """
        Load a grid of template magnitudes from a saved file
        """
        if not os.path.exists(os.path.join(self.base_path, filename)):
            raise ValueError("No model file exists by that name!")
        
        template = np.genfromtxt(os.path.join(self.base_path, filename), delimiter=",")

        if hasattr(self, "template") and self.template is not None:
            # Warn that we're overwriting the existing template
            logger.warning("Overwriting existing tempalte grid")

        self.template = template
    
    def predict_lightcurve(self, phase_min, phase_max, wavelength, show=True):
        # Predict a light curve
        if phase_max > self.max_phase or phase_min < self.min_phase:
            raise ValueError("Phases need to be within the bounds of the GP")
        if wavelength < self.min_wl or wavelength > self.max_wl:
            raise ValueError("Wavelength needs to be within the bounds of the GP")
        
        if self.log_transform is not False:
            linear_phases = np.linspace(phase_min, phase_max, int(min(len(self.phase_grid)/2, len(self.wl_grid)/2)))
            phases = np.log(linear_phases + self.log_transform)
            waves = np.ones(len(phases)) * np.log10(wavelength)

        else:
            phases = np.linspace(phase_min, phase_max, int(min(len(self.phase_grid)/2, len(self.wl_grid)/2)))
            linear_phases = phases
            waves = np.ones(len(phases)) * wavelength

        prediction, dev = self.surface.predict(np.vstack((phases, waves)).T, return_std=True)
        
        # Add back on template mag for the correct phase and wavelength inds
        template_lc = []
        wl_ind = np.argmin((abs(self.wl_grid - wavelength)))
        for i in range(len(phases)):
            phase_ind = np.argmin((abs(self.phase_grid - linear_phases[i])))
            template_lc.append(self.template[phase_ind, wl_ind])
        template_lc = np.asarray(template_lc)

        plt.plot(linear_phases, prediction + template_lc)
        plt.plot(linear_phases, prediction + template_lc - dev, alpha=0.2, color='blue')
        plt.plot(linear_phases, prediction + template_lc + dev, alpha=0.2, color='blue')
        plt.xlabel("Phase (days)")
        plt.ylabel("Log10(Flux) Relative to Peak")
        plt.title(f"Light curve at {wavelength} Angstroms")
        if show:
            plt.show()

    def predict_lightcurve_over_filter(self, phase_min, phase_max, filt, show=True):
        # Predict a light curve over a filter
        if phase_max > self.max_phase or phase_min < self.min_phase:
            raise ValueError("Phases need to be within the bounds of the GP")
        
        ### Try to integrate over the SED
        trans_wl, trans_eff = query_svo_service(FILT_TEL_CONVERSION[filt.replace("'", "")], filt.replace("'", ""))
        trans_eff /= max(trans_eff)
        # Get min and max wavelength for this filter, let's define it as where eff < 10%
        center_of_filt = trans_wl[np.argmax(trans_eff)]
        center_inds = np.where((trans_eff >= 0.1))[0]
        tail_wls = trans_wl[np.where((trans_eff < 0.1))[0]]
        try:
            min_trans_wl = np.max(tail_wls[np.where((tail_wls < center_of_filt))[0]])
            max_trans_wl = np.min(tail_wls[np.where((tail_wls > center_of_filt))[0]])
        except:
            logger.error(f"Warning: transmission function failed for {filt}")
            return
        
        if max_trans_wl < self.min_wl or min_trans_wl > self.max_wl:
            raise ValueError("Wavelength needs to be within the bounds of the GP")
        
        if self.log_transform is not False:
            linear_phases = np.linspace(phase_min, phase_max, int(min(len(self.phase_grid)/2, len(self.wl_grid)/2)))
            phases = np.log(linear_phases + self.log_transform)
            waves = np.linspace(np.log10(min_trans_wl), np.log10(max_trans_wl), len(self.wl_grid))

        else:
            phases = np.linspace(phase_min, phase_max, int(min(len(self.phase_grid)/2, len(self.wl_grid)/2)))
            linear_phases = phases
            waves = np.linspace(min_trans_wl, max_trans_wl, len(self.wl_grid))

        # Interpolate the transmission function onto the template grid
        _, binned_trans_eff = bin_spec(trans_wl[center_inds], trans_eff[center_inds], 10**waves)
        center_inds = np.where((binned_trans_eff >= 0.1))[0]

        x, y = np.meshgrid(phases, waves)

        prediction, dev = self.surface.predict(np.vstack((x.ravel(), y.ravel())).T, return_std=True)
        prediction = prediction.reshape((len(phases), -1))
        dev = dev.reshape((len(phases)), -1)

        # Integrate over the SED
        # TODO: We have to add the template mag grid onto this before integrating
        lc = []
        lc_upper = []
        lc_lower = []
        for i, sed in enumerate(prediction[:]):
            sed = 10**(sed + self.template[i]) # Put back into linear units for the integration
            synthetic_phot = np.nansum(sed*binned_trans_eff[center_inds]) / len(sed)
            lc.append(synthetic_phot)

            sed_upper = 10**(sed + dev[i] + self.template[i])
            synthetic_phot_upper = np.nansum(sed_upper*binned_trans_eff[center_inds]) / len(sed)
            lc_upper.append(synthetic_phot_upper)

            sed_lower = 10**(sed - dev[i] + self.template[i])
            synthetic_phot_lower = np.nansum(sed_lower*binned_trans_eff[center_inds]) / len(sed)
            lc_lower.append(synthetic_phot_lower)

        lc = np.log10(np.asarray(lc))
        lc_upper = np.log10(np.asarray(lc_upper))
        lc_lower = np.log10(np.asarray(lc_lower))
        
        # Add back on template mag for the correct phase and wavelength inds
        # TODO: Is this correct?
        template_lc = []
        wl_ind = np.argmin((abs(self.wl_grid - center_of_filt)))
        for i in range(len(phases)):
            phase_ind = np.argmin((abs(self.phase_grid - linear_phases[i])))
            template_lc.append(self.template[phase_ind, wl_ind])
        template_lc = np.asarray(template_lc)

        plt.plot(linear_phases, lc + template_lc)
        plt.plot(linear_phases, lc_lower + template_lc, alpha=0.2, color='blue')
        plt.plot(linear_phases, lc_upper + template_lc, alpha=0.2, color='blue')
        plt.xlabel("Phase (days)")
        plt.ylabel("Log10(Flux) Relative to Peak")
        plt.title(f"{filt} Light Curve")
        if show:
            plt.show()
    
    def predict_sed(self, wavelength_min, wavelength_max, phase, show=True):

        # Predict an SED
        if wavelength_max > self.max_wl or wavelength_min < self.min_wl:
            raise ValueError("Wavelengths need to be within the bounds of the GP")
        if phase < self.min_phase or phase > self.max_phase:
            raise ValueError("Phase needs to be within the bounds of the GP")
        
        if self.log_transform is not False:
            linear_waves = np.linspace(wavelength_min, wavelength_max, int(min(len(self.phase_grid)/2, len(self.wl_grid)/2)))
            waves = np.log10(linear_waves)
            phases = np.ones(len(waves)) * np.log(phase + self.log_transform)

        else:
            waves = np.linspace(wavelength_min, wavelength_max, int(min(len(self.phase_grid)/2, len(self.wl_grid)/2)))
            linear_waves = waves
            phases = np.ones(len(waves)) * phase

        prediction, dev = self.surface.predict(np.vstack((phases, waves)).T, return_std=True)

        # Add back on template mag for the correct phase and wavelength inds
        template_lc = []
        phase_ind = np.argmin((abs(self.phase_grid - phase)))
        for i in range(len(waves)):
            wl_ind = np.argmin((abs(self.wl_grid - linear_waves[i])))
            template_lc.append(self.template[phase_ind, wl_ind])
        template_lc = np.asarray(template_lc)

        plt.plot(linear_waves, prediction + template_lc)
        plt.plot(linear_waves, prediction + template_lc - dev, alpha=0.2, color='blue')
        plt.plot(linear_waves, prediction + template_lc + dev, alpha=0.2, color='blue')
        plt.xlabel("Phase (days)")
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Log10(Flux) Relative to Peak")
        plt.title(f"SED at {phase} days")
        if show:
            plt.show()

    def compare_lightcurve_with_photometry(self, sn: SN, filt: str):
        datacube = DataCube(sn=sn)
        datacube.construct_cube()  
        cube = datacube.cube
        
        filtered_cube = cube[(cube["ShiftedFilter"]==filt) & (cube["Phase"] > self.min_phase) & (cube["Phase"] < self.max_phase)]

        observed_phases = filtered_cube["Phase"].values
        observed_fluxes = filtered_cube["ShiftedFlux"].values
        observed_flux_errs = filtered_cube["ShiftedFluxerr"].values

        plt.errorbar(observed_phases, observed_fluxes, yerr=observed_flux_errs, fmt='o', label=sn.name)
        plt.legend()
        plt.show()
