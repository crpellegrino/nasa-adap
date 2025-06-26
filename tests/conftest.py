import numpy as np
import pytest
import pandas as pd

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
            "Filter": np.asarray(["B"]),
            "ShiftedFilter": np.asarray(["B"]),
            "Phase": np.asarray([-15.0]),
            "LogPhase": np.asarray([0.5]),
            "Wavelength": np.asarray([5000.0]),
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