import warnings
import numpy as np

from sklearn.gaussian_process.kernels import Kernel as SklearnKernel, KernelOperator

warnings.filterwarnings("ignore")


class Kernel(SklearnKernel):

    def __init__(self, kernel: SklearnKernel):
        self.kernel = kernel

    def __call__(self, X, Y=None, eval_gradient=False):
        return self.kernel.__call__(X, Y=Y, eval_gradient=eval_gradient)
    
    def is_stationary(self):
        return self.kernel.is_stationary()
    
    def diag(self, X):
        return self.kernel.diag(X)

    def _recursively_get_component(self, kernel=None):
        """Recursively get all components of self.kernel"""
        if kernel is None:
            kernel = self.kernel

        if not isinstance(kernel, KernelOperator):
            return np.asarray([kernel])
        
        return np.hstack([self._recursively_get_component(kernel.k1), self._recursively_get_component(kernel.k2)])

    @property
    def components(self):
        return self._recursively_get_component()
        
    def recursively_set_params(
            self,
            values: list,
            bounds: list | str,
            kernel = None,
        ):
        """
        Recursively set parameters for all component kernels.
        Gets the valid param keywords for the kernel and uses
        them to set the values of the correct parameters.
        If Kernel is a product or sum of kernels, assumes the 
        input values and bounds are in the same order as the kernels
        """
        if kernel is None:
            kernel = self.kernel
        
        if not isinstance(kernel, KernelOperator):
            valid_param_keys = list(kernel.get_params().keys())
            valid_params = {valid_param_keys[0]: values}
            valid_params[valid_param_keys[1]] = bounds
            kernel.set_params(**valid_params)
        
        else:
            k1_dim = kernel.k1.n_dims
            self.recursively_set_params(
                values[:k1_dim][0] if k1_dim == 1 else values[:k1_dim],
                bounds if isinstance(bounds, str) else bounds[:k1_dim],
                kernel=kernel.k1
            )
            self.recursively_set_params(
                values[k1_dim:][0] if kernel.k2.n_dims == 1 else values[k1_dim:],
                bounds if isinstance(bounds, str) else bounds[k1_dim:],
                kernel=kernel.k2
            )
