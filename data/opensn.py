import requests
import matplotlib.pyplot as plt
import json
import os


class OpenSN:

    def __init__(self):

        self.base_url = 'https://api.astrocats.space/'
        self.phot_url = '/photometry/magnitude+e_magnitude+band+time+telescope'


    def query_opensn(self, snname):
        """
        Queries the OpenSN Catalog and downloads the data
        Takes as input the name of a SN and returns
        a dictionary of the returned data, if it exists
        """
        response = json.loads(requests.get(self.base_url + snname + self.phot_url).text)

        if snname not in response.keys(): # Not found
            return {}

        opensn_data = {}
        for phot in response[snname]['photometry']:
            
            if any([source in phot[4] for source in ['Swift', 'ATLAS']]): # Check if it's from a source we're reducing ourselves
                continue
            
            if any([filt in phot[2].lower() for filt in ['w2', 'm2', 'w1']]): # Some entries don't have the source listed, so check if the filter is a UVOT filter
                continue
            
            if not phot[3]: # No time info, so ignore
                continue
            
            if phot[2] and phot[0] and phot[1]: # Mag and error, so real detection
                opensn_data.setdefault(phot[2], []).append({'mag': float(phot[0]), 'err': float(phot[1]), 'mjd': float(phot[3])})
            
            elif phot[2] and phot[0]:
                opensn_data.setdefault(phot[2], []).append({'mag': float(phot[0]), 'err': 9999, 'mjd': float(phot[3])})

        return opensn_data


    def save_opensn_data(self, data, filepath, filename, plot=False):

        """
        Save the returned data from the OpenSN Catalog
        Takes as input the data returned as well as the path
        and name of the file to save
        """

        colors = {'B': 'blue', 'V': 'lime', 'g': 'cyan', 'r': 'orange', 'i': 'red'}
        if plot:
            for filt, phot in data.items():
                for d in phot:
                    if d['err'] < 9999:
                        plt.errorbar(d['mjd'], d['mag'], yerr=d['err'], fmt='o', color=colors.get(filt, 'k'), mec='k')
                    else:
                        plt.scatter(d['mjd'], d['mag'], marker='v', color=colors.get(filt, 'k'), edgecolor='k')

            plt.gca().invert_yaxis()
            plt.show()

        with open(os.path.join(filepath, filename), 'w+') as f:
            json.dump(data, f, indent=4)


    def run_opensn(self, sn, typ, subtyp, base_path):

        """
        Runs the OpenSN routines
        Input:
        ===========================
        sn: a dictionary of information for a specific SN
        typ: a string (removed of spaces) of the SN type
        subtyp: a string (removed of spaces) of the SN subtype
        base_path: the path to the top level of the data directory
        """

        opensn_data = self.query_opensn(sn['name'])
        if len(opensn_data) > 0:
            print(sn['name'], opensn_data.keys())
            if not os.path.exists(os.path.join(base_path, typ, subtyp, sn['name'])):
                os.mkdir(os.path.join(base_path, typ, subtyp, sn['name']))

            self.save_opensn_data(opensn_data, os.path.join(base_path, typ, subtyp, sn['name']), sn['name']+'_opensn.json')

# For testing:
# with open('/home/cmp5cr/nasa_adap/swift_uvot_reductions/sn_list_with_template_info.json', 'r') as f:
#     sn_list = json.load(f)

# base_path = '/home/cmp5cr/nasa_adap/data/'
# typp = 'SESNe'
# typ = typp.replace(' ', '')
# subtypp = 'SN Ic'
# subtyp = subtypp.replace(' ', '')