import pytest
from caat import Kernel

import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class TestKernels:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rbf_kernel = RBF()
        self.white_kernel = WhiteKernel()

    def test_init_should_set_kernel_param(self):
        """Should set kernel param on init"""
        kernel = Kernel(self.rbf_kernel)
        assert kernel.kernel == self.rbf_kernel

    def test_call_should_call_components(self):
        """___call__ should call component methods"""
        X = np.asarray([1, 2, 3])
        kernel = Kernel(self.rbf_kernel)
        assert kernel(X) == self.rbf_kernel(X)

    def test_call_should_call_all_components(self):
        """___call__ should call all component kernel methods"""
        X = np.asarray([1, 2, 3])
        kernel = Kernel(self.rbf_kernel * self.white_kernel)
        print(kernel(X), (self.rbf_kernel * self.white_kernel)(X))
        assert np.all(kernel(X) == (self.rbf_kernel * self.white_kernel)(X))

    def test_is_stationary_should_call_components(self):
        """is_stationary should call component methods"""
        kernel = Kernel(self.rbf_kernel * self.white_kernel)
        assert kernel.is_stationary() == (self.rbf_kernel * self.white_kernel).is_stationary()

    def test_diag_should_call_all_components(self):
        """diag method should call component methods"""
        X = np.asarray([1, 2, 3])
        kernel = Kernel(self.rbf_kernel * self.white_kernel)
        assert np.all(kernel.diag(X) == (self.rbf_kernel * self.white_kernel).diag(X))

    def test_components_should_return_array(self):
        """Calling components should return array"""
        kernel = Kernel(self.rbf_kernel * self.white_kernel)
        assert isinstance(kernel.components, np.ndarray)

    def test_components_should_return_all_component_kernels(self):
        """Components should contain all component kernels"""
        kernel = Kernel(self.rbf_kernel * self.white_kernel)
        assert all([k in kernel.components for k in [self.rbf_kernel, self.white_kernel]])

    def test_recursively_set_params_should_work_for_single_kernel(self):
        """Recursively setting params should work for single kernels"""
        kernel = Kernel(self.rbf_kernel)
        kernel.recursively_set_params([1.0, 2.0], "fixed", kernel=kernel.kernel)
        assert np.all(np.isclose(kernel.get_params()["kernel"].length_scale, [1.0, 2.0]))

    def test_recursively_set_params_should_work_for_fixed_bounds(self):
        """Recursively setting params should work for fixed param bounds"""
        kernel = Kernel(self.rbf_kernel)
        kernel.recursively_set_params([1.0, 2.0], "fixed", kernel=kernel.kernel)
        assert kernel.get_params()["kernel"].length_scale_bounds == "fixed"

    def test_recursively_set_params_should_work_for_variable_bounds(self):
        """Recursively setting params should work for variable param bounds"""
        kernel = Kernel(self.rbf_kernel)
        kernel.recursively_set_params([1.0, 2.0], [0.1, 10], kernel=kernel.kernel)
        assert np.all(np.isclose(kernel.get_params()["kernel"].length_scale_bounds, [0.1, 10]))

    def test_recursively_set_params_should_work_recursively(self):
        """Recursively setting params should work recursively on all component kernels"""
        kernel = Kernel(RBF([0.99, 1.99], (1e-2, 1e2)) * WhiteKernel(1, (1e-2, 1e2)))
        kernel.recursively_set_params([1.0, 2.0, 0.5], "fixed", kernel=kernel.kernel)
        k1_params = kernel.components[0].get_params()["length_scale"]
        k2_params = [kernel.components[1].get_params()["noise_level"]]
        flattened_params = np.concatenate((k1_params, k2_params))
        assert np.all(np.isclose(flattened_params, [1.0, 2.0, 0.5]))

    def test_recursively_set_params_should_use_self_by_default(self):
        """Recursively setting params should use the kernel attribute by default"""
        kernel = Kernel(self.rbf_kernel)
        kernel.recursively_set_params([1.0, 2.0], "fixed")
        assert np.all(
            np.isclose(
                kernel.get_params()["kernel"].length_scale,
                self.rbf_kernel.get_params()["length_scale"]
            )
        )