import pytest

from unittest.mock import patch

from caat import DataCube, SN


class TestDataCube:

    @pytest.fixture(autouse=True)
    def setup(self):
        sn = SN(name='SN2020acat')
        filts = ['UVW2', 'UVM2', 'UVW1', 'U', 'B', 'g', 'c', 'V', 'r', 'o', 'i']

        self.cube = DataCube(sn=sn, filt_list=filts)


    @patch('matplotlib.pyplot.show')
    def test_plot_cube(self, mock_show):

        self.cube.plot_cube()
        mock_show.assert_called_once()
