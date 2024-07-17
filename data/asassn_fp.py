from pyasassn.client import SkyPatrolClient
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np
from astropy.time import Time
from statistics import median


class ASASSNFP:

    def query_asassn_fp(self, ra, dec, radius=5.0/3600.0):
        """
        Queries the ASASSN FP server
        Takes as input coordinates RA and Dec as well as 
        a radius for the forced photometry cone search
        Returns a Pandas df of the returned data, or None if none were found
        """

        client = SkyPatrolClient()

        try:
            lcs = client.cone_search(ra, dec, radius=radius, download=True)
        except: # Throws an error when it can't download the LC
            print('No ASASSN LC to download for this object')
            return None

        df = lcs.data

        ### Save the pandas dataframe as a csv file, but for now lets just return it
        return df


    def get_mag_from_asassn_fp(self, full_df, jdstart, jdend):
        """
        Convert the raw ASASSN forced photometry data to cleaned,
        calibrated magnitudes.
        Takes as input a Pandas dataframe of the forced photometry
        as well as start and end JDs and returns a dictionary of 
        the cleaned, finalized magnitudes.
        """

        colors = {'V': 'green', 'g': 'cyan'}
        asassn_data = {}
        for filt in set(full_df['phot_filter'].values):

            df = full_df[(full_df['phot_filter'] == filt) & (full_df['quality'] == 'G')]
            df = df.reset_index(drop=True)
            
            ### First, process by subtracting off baseline
            cameras = set(df['camera'].values)
            for camera in cameras:
                current_df = df[(df['camera'] == camera) & ((df['jd'] < jdstart - 20) | (df['jd'] > jdend + 20))]
                if len(current_df.index) < 30:
                    print('Not enough epochs to estimate baseline for ', filt)
                    continue

                median_flux = median(current_df['flux'])
                print(filt, camera, median_flux)
                df.loc[(df['camera'] == camera), 'flux'] -= median_flux

            jdmin = int(min(df['jd'].values)) - 1
            jdmax = int(max(df['jd'].values)) + 1
            jds = np.linspace(jdmin, jdmax, int(jdmax-jdmin)+1)

            ### Bin dataframe by mjds
            df['bin'] = pd.cut(df['jd'], jds, labels=jds[:-1])

            for jd in jds:
                # Get rows that fell into this bin

                good = np.where((df['bin'] == jd))[0]
                if len(good) == 0:
                    continue

                avg_jd = np.average(df['jd'][good])
                avg_fluxes = np.average(df['flux'][good], weights=1/df['flux_err'][good].values**2)
                avg_flux_errs = 1 / np.sqrt(sum([1/err**2 for err in df['flux_err'][good].values]))

                if avg_fluxes / avg_flux_errs > 5:
                    avg_mag = -2.5*np.log10(avg_fluxes * 1e3) + 23.9
                    avg_mag_err = 1.0857 * avg_flux_errs / avg_fluxes

                    plt.errorbar(avg_jd, avg_mag, yerr=avg_mag_err, fmt='o', color=colors[filt], mec='black')

                else:
                    avg_mag = -2.5*np.log10(5 * avg_flux_errs * 1e3) + 23.9
                    avg_mag_err = 9999

                    plt.scatter(avg_jd, avg_mag, marker='v', color=colors[filt], edgecolor='black', alpha=0.5)

                asassn_data.setdefault(filt, []).append({'mag': avg_mag, 'err': avg_mag_err, 'mjd': avg_jd - 2400000.5})

        plt.gca().invert_yaxis()
        plt.show()

        return asassn_data


    def run_asassn_fp(self, typp, subtypp, sn_list, base_path):

        """
        Runs the full ASASSN forced photometry routine
        Takes as input a SN type and subtype as well as
        the list of SNe in our sample and the path to the top level 
        directory containing the data
        """
        base_path = base_path + '{}/{}/'.format(typp.replace(' ', ''), subtypp.replace(' ', ''))
        for sn in sn_list[typp][subtypp]:
        
            print(sn['name'])
            df = self.query_asassn_fp(sn['ra'], sn['dec'])
            
            if df is not None:
                if not os.path.exists(os.path.join(base_path, sn['name'])):
                    os.mkdir(os.path.join(base_path, sn['name']))
        
                df.to_csv(os.path.join(base_path, sn['name'], sn['name']+'_asassn_raw.dat'), sep=' ')

                jdstart = Time(sn['discovered'], format='iso', scale='utc').jd - 20
                jdend = Time(sn['discovered'], format='iso', scale='utc').jd + 365
                asassn_data = self.get_mag_from_asassn_fp(df, jdstart, jdend)

# For testing:

# with open('/home/cmp5cr/nasa_adap/swift_uvot_reductions/sn_list_with_template_info.json', 'r') as f:
#     sn_list = json.load(f)