import os
from urllib.request import urlopen
from io import BytesIO
from astropy.io.votable import parse
import numpy as np
import matplotlib.pyplot as plt


ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


FILT_TEL_CONVERSION = {'UVW2': 'Swift',
                        'UVM2': 'Swift',
                        'UVW1': 'Swift',
                        'U': 'Swift',
                        'c': 'Atlas',
                        'o': 'Atlas',
                        'B': 'Swift',
                        'V': 'Swift',
                        'g': 'ZTF',
                        'r': 'ZTF',
                        'i': 'ZTF',
                        'R': 'CTIO',
                        'I': 'CTIO',
                        'J': 'CTIO',
                        'H': 'CTIO',
                        'K': 'CTIO',
                        'Y': 'DECam',
                        'u': 'DECam',
                        'G': 'GAIA',
                        'y': 'PAN-STARRS',
                        'z': 'PAN-STARRS',
                        'w': 'PAN-STARRS'
                    }

colors = {
    "U": "purple",
    "B": "blue",
    "V": "lime",
    "g": "cyan",
    "r": "orange",
    "i": "red",
    "UVW2": "#FE0683",
    "UVM2": "#BF01BC",
    "UVW1": "#8B06FF",
    "c": "turquoise",
    "o": "salmon",
}


def query_svo_service(instrument, filter):

    base_url = 'http://svo2.cab.inta-csic.es/theory/fps/fps.php?'
    if instrument.lower() == 'swift':
        url = base_url + f'ID={instrument}/UVOT.{filter}'
    elif instrument.lower() == 'atlas':
        filter_dict = {'o': 'orange', 'c': 'cyan'}
        url = base_url + f'ID=Misc/ATLAS.{filter_dict[filter]}'
    elif instrument.lower() == 'ztf':
        url = base_url + f'ID=Palomar/ZTF.{filter}'
    elif instrument.lower() == 'ctio' and filter not in ['J', 'H', 'K']:
        url = base_url + f'ID=CTIO/ANDICAM.{filter}_KPNO'
    elif instrument.lower() == 'ctio': 
        url = base_url + f'ID=CTIO/ANDICAM.{filter}'
    elif instrument.lower() == 'decam':
        url = base_url + f'ID=CTIO/DECam.{filter}'
    elif instrument.lower() == 'gaia':
        url = base_url + f'ID=GAIA/GAIA0.G'
    elif instrument.lower() == 'pan-starrs':
        url = base_url + f'ID=PAN-STARRS/PS1.{filter}'
    else:
        url = base_url + f'ID={instrument}/{instrument}.{filter}'
    s = BytesIO(urlopen(url).read())
    table = parse(s).get_first_table().to_table(use_names_over_ids=True)
    return table['Wavelength'], table['Transmission']


def bin_spec(wl, flux, wl2, plot=False):
    """
    Bin a spectrum to a certain resolution
    Parameters
    --------------
    wl: wavelength array
    flux: flux array
    wl2: wavelength array to bin to
    plt: plot the binned and unbinned arrays to compare
    """

    binned_wl = []
    binned_flux= []
    for i in range(len(wl2)):
        ind = np.argmin(abs(wl - wl2[i]))
        if wl[ind] not in binned_wl:
            binned_wl.append(wl[ind])
            binned_flux.append(flux[ind])

    if plot:
        plt.plot(binned_wl, binned_flux, color='blue', label='Binned', alpha=0.5)
        plt.plot(wl, flux, color='orange', label='Original', alpha=0.5)
        plt.legend()
        plt.show()
    return np.asarray(binned_wl), np.asarray(binned_flux)