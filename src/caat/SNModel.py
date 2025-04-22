import os
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor

from .SN import SN
from .SNCollection import SNCollection, SNType
from caat.utils import ROOT_DIR

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
        surface: GaussianProcessRegressor | None = None, 
        phase_bounds: tuple[str, str] | None = None,
        filters_fit: list[str] | None = None,
        sn: SN | None = None, 
        sncollection: SNType | SNCollection | None = None,
    ):
        if not sn and not sncollection:
            raise ValueError("Need to specify either a SN or SNCollection for this model!")

        if sn:
            self.sn = sn
        if sncollection:
            self.collection = sncollection
        
        self.surface = surface
        self.min_phase, self.max_phase = phase_bounds
        self.filters = filters_fit

        self.base_path = os.path.join(ROOT_DIR, "data/final_models/")

    def save(self, filename: str = None):
        if not self.surface:
            logger.warning("Need to instantiate the SNModel class with a GP model to save it!")
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
                
                filename += f"_GP_model_{self.min_phase}_{self.max_phase}_{''.join(filt for filt in self.filters)}.pkl"

        with open(os.path.join(self.base_path, filename), 'wb') as f:
            pickle.dump(self.surface, f)

    def load(self, filename: str):
        if not os.path.exists(os.path.join(self.base_path, filename)):
            raise ValueError("No model file exists by that name!")
        
        with open(os.path.join(self.base_path, filename), 'rb') as f:
            surface = pickle.load(f)

        if self.surface:
            # Warn that we're overwriting the existing model
            logger.warning("Overwriting existing GP model")

        self.surface = surface
    