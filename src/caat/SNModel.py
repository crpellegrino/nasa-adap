import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor

from .DataCube import DataCube
from .SN import SN
from .SNCollection import SNCollection, SNType
from caat.utils import ROOT_DIR, WLE, FILT_TEL_CONVERSION, bin_spec, query_svo_service

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SurfaceArray:
    """ 
    Class to handle a final SED surface that is a numpy array,
    rather than a GaussianProcessRegressor object.
    This is to accomodate the user story of median combining multiple
    GP SEDs generated from individual objects.

    Contains a `predict` method that is called with the same inputs as the 
    `GaussianProcessRegressor.predict` method, so that both can objects
    can be used interchangeably.
    """
    def __init__(self, surface: np.ndarray, phase_grid: np.ndarray, wl_grid: np.ndarray):
        self.surface = surface[0]
        self.iqr = surface[1]

        self.phase_grid = phase_grid
        self.wl_grid = wl_grid

    def predict(self, input: np.ndarray, return_std: bool = False):
        """
        Predict a light curve or SED from the SurfaceArray.
        Analogous to the `GaussianProcessRegressor.predict` method
        """
        # Interpolate over the surface
        interp = RegularGridInterpolator((self.phase_grid, self.wl_grid), self.surface.T)
        result = interp(input)

        if not return_std:
            return result
        
        interp_std = RegularGridInterpolator((self.phase_grid, self.wl_grid), self.iqr.T)
        std = interp_std(input)

        return result, std


class SNModel:
    """
    A Supernova Model object. Saves the outputted 3D SED surface from the
    Gaussian process regression routine for a SN object or a collection of SNe.
    Provides routines for saving and loading the final GP fits, as well as 
    for extracting 2D light curves or SEDs from the interpolated surfaces.
    """
    def __init__(
        self, 
        surface: GaussianProcessRegressor | str | SurfaceArray | None = None, 
        template_mags: np.ndarray | None = None,
        phase_grid: np.ndarray | None = None,
        phase_bounds: tuple | None = None,
        wl_grid: np.ndarray| None = None,
        filters_fit: list[str] | None = None,
        sn: SN | None = None, 
        sncollection: SNType | SNCollection | None = None,
        norm_set: SNType | SNCollection | None = None,
        log_transform: int | float | bool = False
    ):  
        self.base_path = os.path.join(ROOT_DIR, "data/final_models/")
        
        if type(surface) == str:
            ### This will load everything from the fits file
            self.load_from_fits(surface)
        
        else:
            if not sn and not sncollection:
                raise ValueError("Need to specify either a SN or SNCollection for this model!")
            
            if sn:
                self.sn = sn
            if sncollection:
                self.collection = sncollection
            if norm_set:
                self.norm_set = norm_set
            
            self.surface = surface
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

    def save_fits(self, filename: str = None, force: bool = False):
        if not self.surface:
            logger.warning("Need to instantiate the SNModel class with a GP model to save it!")
            return
        
        if not filename:
            try:
                filename = f"{self.sn.name}_GP_model.fits"
            except:
                filename = ''
                if hasattr(self.collection, "type"):
                    filename += self.collection.type
                    if hasattr(self.collection, "subtype"):
                        filename += f"_{self.collection.subtype}"
                else:
                    filename += f"{', '.join(sn.name for sn in self.collection.sne)}"
                
                filename += "_GP_model.fits"

        # Save the GP model, template grid, phase grid, and wavelength grid  
        # as fits HDUs
        model_bytes = pickle.dumps(self.surface)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        model_array = model_array.reshape((1, -1))
        model_hdu = fits.PrimaryHDU(model_array)
        model_hdu.header["FILTERS_FIT"] = ",".join(self.filters)
        model_hdu.header["LOG_TRANSFORM"] = self.log_transform
        if hasattr(self, "sn"):
            model_hdu.header["OBJECTS"] = self.sn.name 
        else:
            model_hdu.header["OBJECTS"] = ",".join([sn.name for sn in self.collection.sne])
        model_hdu.header["NORM_SET"] = ",".join([sn.name for sn in self.norm_set.sne])
        if isinstance(self.surface, GaussianProcessRegressor):
            model_hdu.header["TYPE"] = "Gaussian Process Regressor"
        elif isinstance(self.surface, SurfaceArray):
            model_hdu.header["TYPE"] = "Numpy Array"

        template_hdu = fits.ImageHDU(self.template, name="TEMPLATE")

        phase_array = self.phase_grid
        phase_hdu = fits.ImageHDU(phase_array, name="PHASE ARRAY")

        wavelength_array = self.wl_grid
        wavelength_hdu = fits.ImageHDU(wavelength_array, name="WAVELENGTH ARRAY")
        
        hdul = fits.HDUList([model_hdu, template_hdu, phase_hdu, wavelength_hdu])
        hdul.writeto(os.path.join(self.base_path, filename), overwrite=force)
        hdul.close()

    def load_from_fits(self, filename: str):
        with fits.open(os.path.join(self.base_path, filename)) as hdul:
            surface = pickle.loads(hdul[0].data)
            log_transform = hdul[0].header["LOG_TRANSFORM"]
            filters_fit = hdul[0].header["FILTERS_FIT"]
            object_names = hdul[0].header["OBJECTS"]
            norm_set_names = hdul[0].header["NORM_SET"]
            
            template = hdul[1].data
            phase_grid = hdul[2].data
            wl_grid = hdul[3].data

        self.surface = surface
        self.template = template
        self.phase_grid = phase_grid
        self.min_phase, self.max_phase = min(phase_grid), max(phase_grid)
        self.wl_grid = wl_grid
        self.min_wl, self.max_wl = min(wl_grid), max(wl_grid)

        self.log_transform = log_transform
        self.filters = filters_fit
        if "," not in object_names:
            # Only one object in our sample, so load it as a SN object
            self.sn = SN(name=object_names)
        else:
            self.collection = SNCollection(names=object_names.split(","))
        
        self.norm_set = SNCollection(names=norm_set_names.split(","))
    
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
        if self.template is not None:
            template_lc = []
            wl_ind = np.argmin((abs(self.wl_grid - wavelength)))
            for i in range(len(phases)):
                phase_ind = np.argmin((abs(self.phase_grid - linear_phases[i])))
                template_lc.append(self.template[phase_ind, wl_ind])
            template_lc = np.asarray(template_lc)

        else:
            template_lc = np.zeros(len(prediction))

        plt.plot(linear_phases, prediction + template_lc)
        plt.plot(linear_phases, prediction + template_lc - 1.96*dev, alpha=0.2, color='blue')
        plt.plot(linear_phases, prediction + template_lc + 1.96*dev, alpha=0.2, color='blue')
        plt.xlabel("Phase (days)")
        plt.ylabel("Log10(Flux) Relative to Peak")
        plt.title(f"Light curve at {wavelength} Angstroms")
        if show:
            plt.show()

    # def predict_lightcurve_over_filter(self, phase_min, phase_max, filt, show=True):
    #     # Predict a light curve over a filter
    #     if phase_max > self.max_phase or phase_min < self.min_phase:
    #         raise ValueError("Phases need to be within the bounds of the GP")
        
    #     ### Try to integrate over the SED
    #     trans_wl, trans_eff = query_svo_service(FILT_TEL_CONVERSION[filt.replace("'", "")], filt.replace("'", ""))
    #     trans_eff /= max(trans_eff)
    #     # Get min and max wavelength for this filter, let's define it as where eff < 10%
    #     center_of_filt = trans_wl[np.argmax(trans_eff)]
    #     center_inds = np.where((trans_eff >= 0.1))[0]
    #     tail_wls = trans_wl[np.where((trans_eff < 0.1))[0]]
    #     try:
    #         min_trans_wl = np.max(tail_wls[np.where((tail_wls < center_of_filt))[0]])
    #         max_trans_wl = np.min(tail_wls[np.where((tail_wls > center_of_filt))[0]])
    #     except:
    #         logger.error(f"Warning: transmission function failed for {filt}")
    #         return
        
    #     if max_trans_wl < self.min_wl or min_trans_wl > self.max_wl:
    #         raise ValueError("Wavelength needs to be within the bounds of the GP")
        
    #     if self.log_transform is not False:
    #         linear_phases = np.linspace(phase_min, phase_max, int(min(len(self.phase_grid)/2, len(self.wl_grid)/2)))
    #         phases = np.log(linear_phases + self.log_transform)
    #         waves = np.linspace(np.log10(min_trans_wl), np.log10(max_trans_wl), len(self.wl_grid))

    #     else:
    #         phases = np.linspace(phase_min, phase_max, int(min(len(self.phase_grid)/2, len(self.wl_grid)/2)))
    #         linear_phases = phases
    #         waves = np.linspace(min_trans_wl, max_trans_wl, len(self.wl_grid))

    #     # Interpolate the transmission function onto the template grid
    #     _, binned_trans_eff = bin_spec(trans_wl[center_inds], trans_eff[center_inds], 10**waves)
    #     center_inds = np.where((binned_trans_eff >= 0.1))[0]

    #     x, y = np.meshgrid(phases, waves)

    #     prediction, dev = self.surface.predict(np.vstack((x.ravel(), y.ravel())).T, return_std=True)
    #     prediction = prediction.reshape((len(phases), -1))
    #     dev = dev.reshape((len(phases)), -1)

    #     # Integrate over the SED
    #     # TODO: We have to add the template mag grid onto this before integrating
    #     lc = []
    #     lc_upper = []
    #     lc_lower = []
    #     for i, sed in enumerate(prediction[:]):
    #         sed = 10**(sed + self.template[i]) # Put back into linear units for the integration
    #         synthetic_phot = np.nansum(sed*binned_trans_eff[center_inds]) / len(sed)
    #         lc.append(synthetic_phot)

    #         sed_upper = 10**(sed + dev[i] + self.template[i])
    #         synthetic_phot_upper = np.nansum(sed_upper*binned_trans_eff[center_inds]) / len(sed)
    #         lc_upper.append(synthetic_phot_upper)

    #         sed_lower = 10**(sed - dev[i] + self.template[i])
    #         synthetic_phot_lower = np.nansum(sed_lower*binned_trans_eff[center_inds]) / len(sed)
    #         lc_lower.append(synthetic_phot_lower)

    #     lc = np.log10(np.asarray(lc))
    #     lc_upper = np.log10(np.asarray(lc_upper))
    #     lc_lower = np.log10(np.asarray(lc_lower))
        
    #     # Add back on template mag for the correct phase and wavelength inds
    #     # TODO: Is this correct?
    #     template_lc = []
    #     wl_ind = np.argmin((abs(self.wl_grid - center_of_filt)))
    #     for i in range(len(phases)):
    #         phase_ind = np.argmin((abs(self.phase_grid - linear_phases[i])))
    #         template_lc.append(self.template[phase_ind, wl_ind])
    #     template_lc = np.asarray(template_lc)

    #     plt.plot(linear_phases, lc + template_lc)
    #     plt.plot(linear_phases, lc_lower + template_lc, alpha=0.2, color='blue')
    #     plt.plot(linear_phases, lc_upper + template_lc, alpha=0.2, color='blue')
    #     plt.xlabel("Phase (days)")
    #     plt.ylabel("Log10(Flux) Relative to Peak")
    #     plt.title(f"{filt} Light Curve")
    #     if show:
    #         plt.show()
    
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
        if self.template is not None:
            template_lc = []
            phase_ind = np.argmin((abs(self.phase_grid - phase)))
            for i in range(len(waves)):
                wl_ind = np.argmin((abs(self.wl_grid - linear_waves[i])))
                template_lc.append(self.template[phase_ind, wl_ind])
            template_lc = np.asarray(template_lc)
        else:
            template_lc = np.zeros(len(prediction))

        plt.plot(linear_waves, prediction + template_lc)
        plt.plot(linear_waves, prediction + template_lc - 1.96*dev, alpha=0.2, color='blue')
        plt.plot(linear_waves, prediction + template_lc + 1.96*dev, alpha=0.2, color='blue')
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
