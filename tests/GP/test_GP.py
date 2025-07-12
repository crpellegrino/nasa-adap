import pytest
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

from unittest.mock import patch, Mock


class TestGP:

    @pytest.fixture(autouse=True)
    def setup(self, mock_gp):
        self.phasemin = -20
        self.phasemax = 50
        self.gp = mock_gp
        self.phase_grid = np.arange(0.5, 5.0, 0.1)
        self.wl_grid = np.arange(2000.0, 8000.0, 100.0)

    def test_prepare_data(self, mock_datacube):
        with patch(
            "os.path.join", Mock(return_value=None)
        ), patch(
            "os.path.exists", Mock(return_value=True)
        ), patch("pandas.read_csv", return_value=mock_datacube):
            self.gp.prepare_data()
            assert isinstance(self.gp.collection.sne[0].cube, pd.DataFrame)
        
    def test_prepare_data_with_log_transform(self, mock_datacube, mock_log_transformed_gp):
        self.gp = mock_log_transformed_gp
        with patch(
            "os.path.join", Mock(return_value=None)
        ), patch(
            "os.path.exists", Mock(return_value=True)
        ), patch("pandas.read_csv", return_value=mock_datacube):
            self.gp.prepare_data()
            assert isinstance(self.gp.collection.sne[0].cube, pd.DataFrame)

    def test_process_dataset(self):
        """Test process_dataset method"""
        phases, wls, mags, errs = self.gp.process_dataset("B")
        assert all([isinstance(arr, np.ndarray) for arr in [phases, wls, mags, errs]])

    def test_run_gp(self, mock_process_data_result):
        with patch(
            "caat.GP.GP.process_dataset", Mock(return_value=mock_process_data_result)
        ), patch(
            "sklearn.gaussian_process.GaussianProcessRegressor.fit", Mock(return_value=None)
        ):
            gp, phases, mags, errs = self.gp.run("B", 0.9)
            assert isinstance(gp, GaussianProcessRegressor)
            assert phases.shape == mags.shape == errs.shape

    @patch("sklearn.gaussian_process.GaussianProcessRegressor.predict")
    def test_predict_gp(self, mock_gp_predict, mock_process_data_result):
        mock_gp_predict.return_value=(
            np.random.rand(10),
            np.random.rand(10),
        )
        with patch(
            "caat.GP.GP.process_dataset", Mock(return_value=mock_process_data_result)
        ), patch(
            "sklearn.gaussian_process.GaussianProcessRegressor.fit", Mock(return_value=None)
        ):
            self.gp.predict("B", 0.9)
            mock_gp_predict.assert_called_once()

    @patch("sklearn.gaussian_process.GaussianProcessRegressor.predict")
    def test_predict_gp_should_plot(self, mock_gp_predict, mock_process_data_result):
        mock_gp_predict.return_value=(
            np.random.rand(10),
            np.random.rand(10),
        )
        mock_plot_gp = Mock(return_value=None)
        with patch(
            "caat.GP.GP.process_dataset", Mock(return_value=mock_process_data_result)
        ), patch(
            "sklearn.gaussian_process.GaussianProcessRegressor.fit", Mock(return_value=None)
        ), patch(
            "caat.Plot.Plot.plot_gp_predict_gp", mock_plot_gp
        ):
            self.gp.predict("B", 0.9, plot=True)
            mock_plot_gp.assert_called_once()
