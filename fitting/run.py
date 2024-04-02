from models import *

kernel = RBFKernel([np.log(10.0), 500.0], (0.1, 2.0e3)).kernel + WhiteNoiseKernel(1., (1e-10, 10.)).kernel

gp = GP3D('SNII', 'SNIIP', kernel)
gp.predict_gp(['UVW2', 'UVM2', 'UVW1', 'U', 'B', 'V'], -20, 50, 0.9, plot=True, log_transform=30)

print('Kernel Used: ', gp.gaussian_process.kernel)