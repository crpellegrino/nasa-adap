import numpy as np
import pytest
import pandas as pd
from sklearn.gaussian_process.kernels import RBF

from caat.SN import SN
from caat.SNCollection import SNCollection


@pytest.fixture
def mock_data() -> dict:
    return {
        "B": [
            {
            "mjd": 60000.0,
            "mag": 18.0,
            "err": 0.1,
            "nondetection": False,
            }
        ]
    }

@pytest.fixture
def mock_datacube() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Filter": np.asarray(["B", "B", "B"]),
            "ShiftedFilter": np.asarray(["B", "B", "B"]),
            "Phase": np.asarray([-15.0, 0.0, 15.0]),
            "LogPhase": np.asarray([0.5, 3.0, 3.5]),
            "Wavelength": np.asarray([5000.0, 5000.0, 5000.0]),
            "Mag": np.asarray([18.0, 16.0, 17.0]),
            "ShiftedMag": np.asarray([0.2, 0.0, 0.1]),
            "Magerr": np.asarray([0.1, 0.1, 0.1]),
            "Flux": np.asarray([0.2, 0.01, 0.1]),
            "ShiftedFlux": np.asarray([0.2, 0.01, 0.1]),
            "FluxErr": np.asarray([0.1, 0.1, 0.1]),
            "ShiftedFluxerr": np.asarray([0.1, 0.1, 0.1]),
            "Nondetection": np.asarray([False, False, False]),
            "ShiftedWavelength": np.asarray([5050.0, 5000.0, 5000.0]),
            "LogShiftedWavelength": np.log10(np.asarray([5050.0, 5000.0, 5000.0])),
            "MagFromPeak": np.asarray([-2.0, 0.0, -1.0]),
        }
    )

@pytest.fixture
def mock_sn(mock_data, mock_datacube) -> SN:
    sn = SN(
        data=mock_data
    )
    sn.cube = mock_datacube
    
    return sn

@pytest.fixture
def mock_sncollection(mock_sn) -> SNCollection:
    return SNCollection(SNe=mock_sn)

@pytest.fixture
def mock_kernel() -> RBF:
    return RBF()
