import pytest
from caat.GP3D import GP3D
from unittest.mock import Mock


@pytest.fixture
def mock_gp3d(mock_sncollection, mock_kernel) -> GP3D:
    mock_gp = GP3D(
        collection=mock_sncollection, 
        kernel=mock_kernel, 
        filtlist=['B'], 
        phasemin=-20,
        phasemax=50,
        set_to_normalize=mock_sncollection,
    )
    mock_gp.prepare_data = Mock(return_value=None)
    
    return mock_gp