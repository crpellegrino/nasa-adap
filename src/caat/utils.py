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