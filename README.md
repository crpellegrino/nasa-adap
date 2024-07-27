### summary of classes and functions within models.py ###

class __SN__\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **\_\_init\_\_**(self, classification, subtype, name)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **\_\_repr\_\_**(self)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **load_swift_data**(self)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **load_json_data**(self)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **plot_data**(self, only_this_filt='', shift=False)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **fit_for_max**(self, filt, shift_array=[-3,-2,-1,0,1,2,3], plot=False)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **shift_to_max**(self, filt, shift_Array=[-3,-2,-1,0,1,2,3], plot=False)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **log_transform_time**(self, phases, phase_start=30)
	
class __Type__\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **\_\_init\_\_**(self, classification)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **get_subtypes**(self)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **build_object_list**(self)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **plot_all_lcs**(self, filt)
  
class __RBFKernel__\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **\_\_init\_\_**(self, length_scale, length_scale_bounds)
  
class __WhiteNoiseKernel__\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **\_\_init\_\_**(self, noise_level, noise_level_bounds)
  
class __MaternKernel__\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **\_\_init\_\_**(self, length_scale, length_scale_bounds, nu)
  
class __Fitter__\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **\_\_init\_\_**(self, classification)
  
class __GP(Fitter)__\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **\_\_init\_\_**(self, classification, subtype, kernel)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **process_dataset_for_gp**(self, filt, phasemin, phasemax, log_transform=False)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **run_gp**(self, filt, phasemin, phasemax, test_size)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **predict_gp**(self, filt, phasemin, phasemax, test_size, plot=False)
  
class __GP3D(GP)__\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **build_samples_3d**(self, filt, phasemin, phasemax, log_transform=False)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **process_dataset_for_gp_3d**(self, filtlist, phasemin, phasemax, log_transform=False)\
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **run_gp**(self, filtlist, phasemin, phasemax, test_size, log_transform=False)\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **predict_gp**(self, filtlist, phasemin, phasemax, test_size, plot=False, log_transform=False)
