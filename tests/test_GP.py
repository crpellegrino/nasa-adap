from caat import SNCollection, GP, RBFKernel


def test_gp_init():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP(sncollection, kernel)

    assert len(gp.collection.sne) == len(sncollection.sne)
    assert kernel.get_params == gp.kernel.get_params

def test_process_dataset_for_gp():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP(sncollection, kernel)

    phases, mags, errs, wls  = (
        gp.process_dataset_for_gp(
            'V',
            -20,
            50
        )
    )
    assert len(phases) > 0 and len(mags) > 0 and len(errs) > 0 and len(wls) > 0
    assert len(phases) == len(mags) == len(errs) == len(wls)

def test_gp_with_fluxes():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP(sncollection, kernel)

    _, mags, _, _  = (
        gp.process_dataset_for_gp(
            'V',
            -20,
            50,
            use_fluxes=True
        )
    )
    assert all([m < 10 for m in mags])

def test_gp_with_log_transform():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP(sncollection, kernel)

    log_mjds, _, _, _  = (
        gp.process_dataset_for_gp(
            'V',
            -20,
            50,
            log_transform=30
        )
    )

    mjds, _, _, _  = (
        gp.process_dataset_for_gp(
            'V',
            -20,
            50,
        )
    )
    assert len(mjds) == len(log_mjds)
    assert all([mjd > 0 for mjd in log_mjds])

def test_run_gp():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP(sncollection, kernel)

    ### Test all boolean combinations of log_transform and use_flux
    for log_transform in [False, 30]:
        for use_flux in [False, True]:

            gaussian_process, _, _, _ = gp.run_gp(
                'V', 
                -20, 
                50, 
                0.9, 
                log_transform=log_transform, 
                sn_set=sncollection.sne, 
                use_fluxes=use_flux
            )

            assert len(gaussian_process.kernel_.theta) == len(kernel.theta)

def test_predict_gp():
    sncollection = SNCollection(sntype='SESNe', snsubtype='SNIIb')
    kernel = RBFKernel(5.0, [1.0, 100.0]).kernel
    gp = GP(sncollection, kernel)

    ### Test all boolean combinations of log_transform and use_flux
    for log_transform in [False, 30]:
        for use_flux in [False, True]:
            gaussian_process, phases, _, _ = gp.run_gp(
                'V', 
                -20, 
                50, 
                0.9, 
                log_transform=log_transform, 
                sn_set=sncollection.sne, 
                use_fluxes=use_flux
            )

            mean_prediction = gaussian_process.predict(sorted(phases))
            assert len(phases) == len(mean_prediction)


