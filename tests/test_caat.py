from caat import CAAT
import pandas as pd
import os
from caat.utils import ROOT_DIR


def test_caat_exists():
    try:
        caat = CAAT() 
    except Exception as e:
        assert False, e

def test_caat_type():
    caat = CAAT()
    assert isinstance(caat.caat, pd.DataFrame)

def test_create_caat():
    CAAT.create_db_file(CAAT, base_db_name="caat_tmp.csv")
    assert os.path.isfile(os.path.join(ROOT_DIR, "data/", "caat_tmp.csv"))
    os.popen(f'rm {os.path.join(ROOT_DIR, "data/", "caat_tmp.csv")}')