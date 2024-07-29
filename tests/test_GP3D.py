from caat import SNCollection, RBFKernel, GP3D
import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel


def test_build_gp3d_samples():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP3D(sncollection, kernel)

    ### Test all boolean combinations of log_transform and use_flux
    for log_transform in [False, 30]:
        for use_flux in [False, True]:
            phases, wls, mags, err_grid = (
                gp.build_samples_3d(
                    'V',
                    -20,
                    50,
                    log_transform=log_transform,
                    sn_set=sncollection.sne,
                    use_fluxes=use_flux
                )
            )

            assert len(phases) > 0 and len(mags) > 0 and len(wls) > 0 and len(err_grid) > 0
            assert len(phases) == len(mags) == len(wls) == len(err_grid)

def test_process_dataset_for_gp3d():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP3D(sncollection, kernel)

    ### Test all boolean combinations of log_transform and use_flux
    for log_transform in [False, 30]:
        for use_flux in [False, True]:
            (
                all_phases,
                all_wls,
                all_mags,
                all_errs,
                all_template_phases,
                all_template_wls,
                all_template_mags,
                all_template_errs,
            ) = gp.process_dataset_for_gp_3d(
                ['B', 'g', 'V'],
                -20,
                50,
                log_transform=log_transform,
                fit_residuals=True,
                set_to_normalize=sncollection.sne,
                use_fluxes=use_flux,
            )
            assert len(all_phases) > 0 and len(all_wls) > 0 and len(all_mags) > 0 and len(all_errs) > 0
            assert len(all_phases) == len(all_wls) == len(all_mags) == len(all_errs)

            assert len(all_template_phases) > 0 and len(all_template_wls) > 0 and len(all_template_mags) > 0 and len(all_template_errs) > 0
            assert len(all_template_phases) == len(all_template_wls) == len(all_template_mags) == len(all_template_errs)

def test_median_grid():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP3D(sncollection, kernel)

    (
        _,
        _,
        _,
        _,
        all_template_phases,
        all_template_wls,
        all_template_mags,
        all_template_errs,
    ) = gp.process_dataset_for_gp_3d(
        ['B', 'g', 'V'],
        -20,
        50,
        log_transform=30,
        fit_residuals=True,
        set_to_normalize=sncollection.sne,
        use_fluxes=True,
    )
    phase_grid, wl_grid, mag_grid, err_grid = (
        gp.construct_median_grid(
            -20,
            50,
            ['B', 'g', 'V'],
            all_template_phases,
            all_template_wls,
            all_template_mags,
            all_template_errs,
            log_transform=30,
            plot=False,
            use_fluxes=True,
        )
    )

    assert mag_grid.shape == (len(phase_grid), len(wl_grid))
    assert err_grid.shape == (len(phase_grid), len(wl_grid))

def test_polynomial_grid():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP3D(sncollection, kernel)

    (
        _,
        _,
        _,
        _,
        all_template_phases,
        all_template_wls,
        all_template_mags,
        all_template_errs,
    ) = gp.process_dataset_for_gp_3d(
        ['B', 'g', 'V'],
        -20,
        50,
        log_transform=30,
        fit_residuals=True,
        set_to_normalize=sncollection.sne,
        use_fluxes=True,
    )
    phase_grid, wl_grid, mag_grid, err_grid = (
        gp.construct_polynomial_grid(
            -20,
            50,
            ['B', 'g', 'V'],
            all_template_phases,
            all_template_wls,
            all_template_mags,
            all_template_errs,
            log_transform=30,
            plot=False,
            use_fluxes=True,
        )
    )

    assert mag_grid.shape == (len(phase_grid), len(wl_grid))
    assert err_grid.shape == (len(phase_grid), len(wl_grid))

def test_subtract_from_grid():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP3D(sncollection, kernel)

    (
        _,
        _,
        _,
        _,
        all_template_phases,
        all_template_wls,
        all_template_mags,
        all_template_errs,
    ) = gp.process_dataset_for_gp_3d(
        ['B', 'g', 'V'],
        -20,
        50,
        log_transform=30,
        fit_residuals=True,
        set_to_normalize=sncollection.sne,
        use_fluxes=True,
    )
    phase_grid, wl_grid, mag_grid, err_grid = (
        gp.construct_polynomial_grid(
            -20,
            50,
            ['B', 'g', 'V'],
            all_template_phases,
            all_template_wls,
            all_template_mags,
            all_template_errs,
            log_transform=30,
            plot=False,
            use_fluxes=True,
        )
    )
    residuals = gp.subtract_data_from_grid(
        sncollection.sne[0],
        -20,
        50,
        ['B', 'g', 'V'],
        phase_grid,
        wl_grid,
        np.random.random((len(phase_grid), len(wl_grid))),
        np.ones((len(phase_grid), len(wl_grid))) * 0.01,
        log_transform=30,
        use_fluxes=True
    )

    assert len(residuals) > 0

def test_run_gp_without_specifying_subtract():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP3D(sncollection, kernel)

    with pytest.raises(Exception, match=r'Must toggle either .*'):
        gp.run_gp(
            ['B', 'g', 'V'],
            -20,
            50,
            log_transform=30,
            fit_residuals=True,
            set_to_normalize=sncollection.sne,
        )

def test_run_gp3d():

    for use_flux in [True, False]:
        sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
        kernel = RBFKernel([np.log(10.0), np.log10(5000)], (0.01, 4.0)).kernel * ConstantKernel(2., (1e-2, 1e2))
        gp = GP3D(sncollection, kernel)
        gaussian_processes, phase_grid, kernel_params, wl_grid = (
            gp.run_gp(
                ['B', 'g', 'V'],
                -20,
                50,
                log_transform=30,
                fit_residuals=True,
                set_to_normalize=sncollection.sne,
                subtract_polynomial=True,
                use_fluxes=use_flux
            )
        )

        assert len(gaussian_processes) > 0 and len(phase_grid) > 0 and len(kernel_params) > 0 and len(wl_grid) > 0
        assert gaussian_processes[0].shape <= (len(phase_grid), len(wl_grid))