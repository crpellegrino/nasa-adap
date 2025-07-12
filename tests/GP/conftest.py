import pytest
from caat import GP
import numpy as np

@pytest.fixture
def mock_gp(mock_sncollection, mock_kernel) -> GP:
    mock_gp = GP(
        sne_collection=mock_sncollection,
        kernel=mock_kernel,
        filtlist=["B"],
        phasemin=-20,
        phasemax=50,
        log_transform=22
    )
    return mock_gp

@pytest.fixture
def mock_process_data_result() -> tuple[np.ndarray]:
    return (
        np.random.rand(100),
        np.random.rand(100),
        np.random.rand(100),
        np.random.rand(100),
    )
