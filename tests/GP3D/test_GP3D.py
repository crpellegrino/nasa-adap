import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch


class TestGP3D:
    
    @pytest.fixture(autouse=True)
    def setup(self, mock_gp3d):
        self.phasemin = -20
        self.phasemax = 50
        self.gp = mock_gp3d
        self.phase_grid = np.arange(0.5, 5.0, 0.1)
        self.wl_grid = np.arange(2000.0, 8000.0, 100.0)
    
    def test_build_samples(self):
        """Test build_samples method"""
        ### Test all boolean combinations of log_transform and use_flux
        with patch("caat.GP.GP.process_dataset", Mock(
            return_value=(
                np.asarray([0.5, 3.5]), 
                np.asarray([-1.0, 3.0]),
                np.asarray([0.1, 0.1]), 
                np.asarray([3.5, 3.5])
            )
        )):
            for log_transform in [False, 22]:
                for use_flux in [False, True]:
                    phases, wls, mags, err_grid = (
                        self.gp.build_samples('B', log_transform=log_transform, use_fluxes=use_flux)
                    )

                    assert all([isinstance(var, np.ndarray) for var in [phases, mags, wls, err_grid]])
                    assert len(phases) == len(mags) == len(wls) == len(err_grid)

    def test_process_dataset(self):
        """Test process_dataset method"""
        template_df = self.gp.process_dataset()
        assert isinstance(template_df, pd.DataFrame)

    def test_median_grid(self, mock_datacube):
        """Test construct_median_grid method"""
        phase_grid, wl_grid, mag_grid, err_grid = (
            self.gp.construct_median_grid(
                self.phasemin,
                self.phasemax,
                ['B'],
                mock_datacube,
                log_transform=22,
                plot=False,
                use_fluxes=True,
            )
        )

        assert mag_grid.shape == (len(phase_grid), len(wl_grid))
        assert err_grid.shape == (len(phase_grid), len(wl_grid))

    def test_polynomial_grid(self, mock_datacube):
        """Test construct_polynomial_grid method"""
        phase_grid, wl_grid, mag_grid, err_grid = (
            self.gp.construct_polynomial_grid(
                self.phasemin,
                self.phasemax,
                ['B'],
                mock_datacube,
                log_transform=22,
                plot=False,
                use_fluxes=True,
            )
        )

        assert mag_grid.shape == (len(phase_grid), len(wl_grid))
        assert err_grid.shape == (len(phase_grid), len(wl_grid))

    def test_subtract_from_grid(self, mock_sn):
        """Test subtract_from_grid method"""
        residuals = self.gp.subtract_data_from_grid(
            mock_sn,
            ['B'],
            self.phase_grid,
            self.wl_grid,
            np.random.random((len(self.phase_grid), len(self.wl_grid))),
            np.ones((len(self.phase_grid), len(self.wl_grid))) * 0.01,
            log_transform=22,
            use_fluxes=True
        )

        assert isinstance(residuals, pd.DataFrame)

    def test_run_gp_full_sample_without_specifying_subtract(self):
        """Should raise an Exception when running GP without specifying a subtract method"""
        with pytest.raises(Exception, match=r'Must toggle either .*'):
            self.gp.run_gp_on_full_sample(
                plot=False,
            )

    def test_run_gp_individually_without_specifying_subtract(self):
        """Should raise an Exception when running GP without specifying a subtract method"""
        with pytest.raises(Exception, match=r'Must toggle either .*'):
            self.gp.run_gp_individually(
                plot=False,
            )

    def test_build_test_wavelength_phase_grid_from_photometry(self, mock_datacube):
        """Test build_test_wavelength_phase_grid_from_photometry method"""
        (
            x, 
            y, 
            wl_inds_fitted, 
            phase_inds_fitted, 
            min_phase
        ) = self.gp.build_test_wavelength_phase_grid_from_photometry(
            mock_datacube["Wavelength"].values,
            mock_datacube["Phase"].values,
            self.wl_grid,
            self.phase_grid,
        )

        assert len(x) == len(y)
        assert isinstance(wl_inds_fitted, list)
        assert isinstance(phase_inds_fitted, list)
        assert isinstance(min_phase, float | None)

    def test_optimize_hyperparameters(self):
        """Test optimize_hyperparameters method"""
        class MockGaussianProcessRegressor:
            class kernel_:
                theta = [1.0, 1.0]
            
            def fit(*args, **kwargs):
                pass

        with patch(
            "caat.GP3D.GP3D.construct_polynomial_grid", Mock(
                return_value=(
                    self.phase_grid, 
                    self.wl_grid,
                    np.random.random((len(self.phase_grid), len(self.wl_grid))),
                    np.ones((len(self.phase_grid), len(self.wl_grid))) * 0.01,
                )
            )
        ), patch(
            "caat.GP3D.GP3D.subtract_data_from_grid", Mock(
                return_value=pd.DataFrame(
                    [
                        {
                            "Filter": "B",
                            "Phase": -15.0,
                            "Wavelength": 5000.0,
                            "MagResidual": 0.1,
                            "MagErr": 0.1,
                            "Mag": -0.5,
                            "Nondetection": False,
                        },
                        {
                            "Filter": "B",
                            "Phase": -15.0,
                            "Wavelength": 5000.0,
                            "MagResidual": 0.1,
                            "MagErr": 0.1,
                            "Mag": -0.5,
                            "Nondetection": False,
                        }
                    ]
                )
            )
        ), patch(
            "sklearn.gaussian_process.GaussianProcessRegressor", MockGaussianProcessRegressor
        ):
            self.gp.kernel.set_params = Mock()
            kernel_params = self.gp.optimize_hyperparams(subtract_polynomial=True)
            assert isinstance(kernel_params, list)
            self.gp.kernel.set_params.assert_called_once()

    # def test_run_gp3d(self):

    #     gp = GP3D(sncollection, kernel)
    #     gaussian_processes, phase_grid, kernel_params, wl_grid = (
    #         gp.run_gp(
    #             ['B', 'g', 'V'],
    #             -20,
    #             50,
    #             log_transform=30,
    #             fit_residuals=True,
    #             set_to_normalize=sncollection.sne,
    #             subtract_polynomial=True,
    #             use_fluxes=use_flux
    #         )
    #     )

    #     assert len(gaussian_processes) > 0 and len(phase_grid) > 0 and len(kernel_params) > 0 and len(wl_grid) > 0
    #     assert gaussian_processes[0].shape <= (len(phase_grid), len(wl_grid))