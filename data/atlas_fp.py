import os
import requests
import sys
import time
import io
import pandas as pd
import numpy as np
import json
from astropy.time import Time
import matplotlib.pyplot as plt


def query_atlas_fp(ra, dec, filepath, filename, mjd_min, mjd_max=60157.0, verbose=False):
    BASEURL = "https://fallingstar-data.com/forcedphot"

    token = os.environ['ATLAS_TOKEN']
    headers = {'Authorization': f'Token {token}', 'Accept': 'application/json'}

    task_url = None
    while not task_url:
        with requests.Session() as s:
            resp = s.post(f"{BASEURL}/queue/", headers=headers, data={
                'ra': ra, 'dec': dec, 'mjd_min': mjd_min, 'mjd_max': mjd_max})

            if resp.status_code == 201:  # successfully queued
                task_url = resp.json()['url']
                if verbose:
                    print(f'The task URL is {task_url}')
            elif resp.status_code == 429:  # throttled
                message = resp.json()["detail"]
                if verbose:
                    print(f'{resp.status_code} {message}')
                t_sec = re.findall(r'available in (\d+) seconds', message)
                t_min = re.findall(r'available in (\d+) minutes', message)
                if t_sec:
                    waittime = int(t_sec[0])
                elif t_min:
                    waittime = int(t_min[0]) * 60
                else:
                    waittime = 10
                if verbose:
                    print(f'Waiting {waittime} seconds')
                time.sleep(waittime)
            else:
                print(f'ERROR {resp.status_code}')
                print(resp.json())
                sys.exit()

    result_url = None
    while not result_url:
        with requests.Session() as s:
            resp = s.get(task_url, headers=headers)

            if resp.status_code == 200:  # HTTP OK
                if resp.json()['finishtimestamp']:
                    result_url = resp.json()['result_url']
                    if verbose:
                        print(f"Task is complete with results available at {result_url}")
                    break
                time.sleep(10)
            else:
                print(f'ERROR {resp.status_code}')
                print(resp.json())
                sys.exit()

    with requests.Session() as s:
        textdata = s.get(result_url, headers=headers).text

    with open(os.path.join(filepath, filename), 'w+') as outfile:
        outfile.write(textdata.replace('###', ''))


def get_mag_from_atlas_fp(filename):

    full_df = pd.read_csv(filename, delim_whitespace=True)
    colors = {'c': 'cyan', 'o': 'orange'}
    atlas_data = {}
    for filt in ['c', 'o']:

        df = full_df[(full_df['F'] == filt) & (full_df['chi/N'] < 2.0)]
        df = df.reset_index(drop=True)

        if len(df) == 0:
            continue

        mjdmin = int(df['MJD'].iloc[0]) - 1
        mjdmax = int(df['MJD'].iloc[-1]) - 1
        mjds = np.linspace(mjdmin, mjdmax, int(mjdmax-mjdmin)+1)

        ### Bin dataframe by mjds
        df['bin'] = pd.cut(df['MJD'], mjds, labels=mjds[:-1])
        
        for mjd in mjds:
            # Get rows that fell into this bin

            #good = np.where((df['uJy'] / df['duJy'] > 3))[0]
            #bad = np.where((df['uJy'] / df['duJy'] <= 3))[0]

            good = np.where((df['bin'] == mjd))[0]
            if len(good) == 0:
                continue

            avg_mjd = np.average(df['MJD'][good])
            avg_fluxes = np.average(df['uJy'][good], weights=1/df['duJy'][good].values**2)
            avg_flux_errs = 1 / np.sqrt(sum([1/err**2 for err in df['duJy'][good].values]))

            if avg_fluxes / avg_flux_errs > 3:
                avg_mag = -2.5*np.log10(avg_fluxes) + 23.9
                avg_mag_err = 1.0857 * avg_flux_errs / avg_fluxes

                plt.errorbar(avg_mjd, avg_mag, yerr=avg_mag_err, fmt='o', color=colors[filt], mec='black')

            else:
                avg_mag = -2.5*np.log10(5 * avg_flux_errs) + 23.9
                avg_mag_err = 9999

                plt.scatter(avg_mjd, avg_mag, marker='v', color=colors[filt], edgecolor='black', alpha=0.5)

            atlas_data.setdefault(filt, []).append({'mag': avg_mag, 'err': avg_mag_err, 'mjd': avg_mjd})

            #df['mag'] = -2.5*np.log10(df['uJy'][good]) + 23.9
            #df['sigma_mag'] = 1.0857 * df['duJy'] / df['uJy']
            #df['mag'][bad] = 23.9 - 2.5*np.log10(5 * df['duJy'][bad])


            #plt.errorbar(df['MJD'][good], df['mag'][good], yerr=df['sigma_mag'][good], fmt='o', color=colors[filt], mec='black')
            #plt.scatter(df['MJD'][bad], df['mag'][bad], marker='v', color=colors[filt], edgecolor='black', alpha=0.5)

            #good_mjds = df['MJD'][good].to_numpy()
            #good_mags = df['mag'][good].to_numpy()
            #good_errs = df['sigma_mag'][good].to_numpy()

            #for i in range(len(good_mjds)):
            #    atlas_data.setdefault(filt, []).append({'mag': good_mags[i], 'err': good_errs[i], 'mjd': good_mjds[i]})


            #bad_mjds = df['MJD'][bad].to_numpy()
            #bad_mags = df['mag'][bad].to_numpy()

            #for i in range(len(bad_mjds)):
            #    atlas_data.setdefault(filt, []).append({'mag': bad_mags[i], 'err': 9999, 'mjd': bad_mjds[i]})


    plt.gca().invert_yaxis()
    plt.show()

    return atlas_data

with open('../swift_uvot_reductions/sn_list_with_template_info.json', 'r') as f:
    sn_list = json.load(f)

base_path = '/home/cmp5cr/nasa_adap/data/'

### Query the data
#for typp in ['SLSN-II']:
#    typ = typp.replace(' ', '')
#    if not os.path.exists(os.path.join(base_path, typ)):
#        os.mkdir(os.path.join(base_path, typ))
#    for subtypp in sn_list[typp].keys():
#        subtyp = subtypp.replace(' ', '')
#        if not os.path.exists(os.path.join(base_path, typ, subtyp)):
#            os.mkdir(os.path.join(base_path, typ, subtyp))
#
#        for sn in sn_list[typp][subtypp]:
#            atlasname = ''
#            for alias in [a for a in sn['aliases'][0].split(', ')]:
#                if 'ATLAS' in alias:
#                    atlasname = alias
#                    break
#            if atlasname:
#                if not os.path.exists(os.path.join(base_path, typ, subtyp, sn['name'])):
#                    os.mkdir(os.path.join(base_path, typ, subtyp, sn['name']))
#                if not os.path.exists(os.path.join(base_path, typ, subtyp, sn['name'], sn['name']+'_atlas_fp_raw.dat')):
#                    mjdmin = Time(sn['discovered'], format='iso', scale='utc').mjd - 10
#                    print('Querying ATLAS for ', sn['name'])
#                    query_atlas_fp(sn['ra'], sn['dec'], os.path.join(base_path, typ, subtyp, sn['name']), sn['name']+'_atlas_fp_raw.dat', mjdmin)

### Reduce the data and save it
for typp in ['SLSN-II']:
    typ = typp.replace(' ', '')
    for subtypp in sn_list[typp].keys():
        subtyp = subtypp.replace(' ', '')
        for sn in sn_list[typp][subtypp]:
            if os.path.exists(os.path.join(base_path, typ, subtyp, sn['name'])) and any(['atlas_fp_raw.dat' in f for f in os.listdir(os.path.join(base_path, typ, subtyp, sn['name']))]):
                for f in os.listdir(os.path.join(base_path, typ, subtyp, sn['name'])):
                    if 'atlas_fp_raw.dat' in f:
                        atlas_data = get_mag_from_atlas_fp(os.path.join(base_path, typ, subtyp, sn['name'], f))
                        if not os.path.isfile(os.path.join(base_path, typ, subtyp, sn['name'], sn['name']+'_atlas_fp.json')):
                            with open(os.path.join(base_path, typ, subtyp, sn['name'], sn['name']+'_atlas_fp.json'), 'w+') as newfile:
                                json.dump(atlas_data, newfile, indent=4)

