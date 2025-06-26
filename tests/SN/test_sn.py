from caat import SN, CAAT
import numpy as np


def test_bad_sn_name():
    try:
        sn = SN(name='bogus_name')
        assert False
    except Exception as e:
        assert True

def test_good_sn_name():
    caat = CAAT().caat
    sn = SN(name=caat.Name.values[0])
    assert sn.name != ""

def test_create_sn_from_data_dict():
    sn = SN(data={'B': [{'mjd': 60000.0, 'mag': 15.0, 'err': 0.1}]})
    assert len(sn.data) > 0


def test_read_sn_info_from_caat():
    caat = CAAT().caat
    row = caat.sample()
    sn = SN(name=row["Name"].values[0])

    assert (sn.info.get("peak_mjd", np.nan) == row["Tmax"].values[0]) or (np.isnan(sn.info.get("peak_mjd", np.nan)) and np.isnan(row["Tmax"].values[0]))
    assert (sn.info.get("peak_mag", np.nan) == row["Magmax"].values[0]) or (np.isnan(sn.info.get("peak_mag", np.nan)) and np.isnan(row["Magmax"].values[0]))
    assert (sn.info.get("peak_filt", np.nan) == row["Filtmax"].values[0]) or (np.isnan(sn.info.get("peak_filt", np.nan)) and np.isnan(row["Filtmax"].values[0]))

def test_load_swift_file():
    sn = SN(name='SN2022acko')
    sn.data = {}
    sn.load_swift_data()
    assert len(sn.data) > 0

def test_load_json_data():
    sn = SN(name='SN2022acko')
    sn.data = {}
    sn.load_json_data()
    assert len(sn.data) > 0

# def test_convert_to_fluxes():
#     sn = SN(name='SN2022acko')
#     sn.load_json_data()
#     sn.load_swift_data()
#     sn.convert_to_fluxes()
#     assert all([d.get('flux', False) for f in sn.data.keys() for d in sn.data[f]])

def test_extinction_correction():
    sn = SN(name='SN2022acko')
    sn.load_json_data()
    sn.load_swift_data()
    sn.correct_for_galactic_extinction()
    assert all([d.get('ext_corrected', False) for f in sn.data.keys() for d in sn.data[f]])

def test_shift_to_max():
    sn = SN(name='SN2022acko')
    sn.load_json_data()
    sn.load_swift_data()
    mjds, _, _, _ = sn.shift_to_max(list(sn.data.keys())[0])
    assert len(mjds) > 0 and all([mjd < 50000.0 for mjd in mjds])