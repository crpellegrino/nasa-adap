import requests
from bs4 import BeautifulSoup
import os
import json
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
from statistics import median
import numpy as np


def check_ztf_fps(batch=True):
    
    ### From here: https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_zfps_userguide.pdf
    
    output_urls = []
    settings = {'email': 'cmp5cr@virginia.edu','userpass': 'gvrc380',
    'option': 'All recent jobs', 'action': 'Query Database'}
    
    if batch:
        r = requests.get('https://ztfweb.ipac.caltech.edu/cgi-bin/' +\
            'getBatchForcedPhotometryRequests.cgi',
            auth=('ztffps', 'dontgocrazy!') ,params=settings)
    else:
        r = requests.get('https://ztfweb.ipac.caltech.edu/cgi-bin/' +\
            'getForcedPhotometryRequests.cgi',
            auth=('ztffps', 'dontgocrazy!') ,params=settings)
    
    if r.status_code == 200:
        print("Script executed normally and queried the ZTF Batch " +\
        "Forced Photometry database.\n")
        
        soup = BeautifulSoup(r.text, 'html.parser')
        for tr in soup.find('table')('tr'):
            row = [td.get_text() for td in tr('td')]
            if len(row) == 0:
                continue
            ra = float(row[1])
            dec = float(row[2])
            lc = row[-1].replace(' ', '')
            if len(lc) == 0:
                continue
     
            wget_prefix = 'wget --http-user=ztffps --http-passwd=dontgocrazy! '#-O '
            wget_url = 'https://ztfweb.ipac.caltech.edu'
            wget_suffix = '"'
            
            #wget_command = wget_prefix + " " + lc.split('/')[-1] + " \"" + wget_url + lc + wget_suffix
            wget_command = wget_prefix + "\"" + wget_url + lc + wget_suffix
            output_urls.append({'ra': ra, 'dec': dec, 'lc': wget_command, 'filename': lc.split('/')[-1]})
    else:
        print("Status_code=",r.status_code,"; Jobs either queued or abnormal execution.")
        
    return output_urls


def download_ztf_fp(sn_list, fp_info, base_path):

    """
    Takes as input a list of SN information and the 
    forced photometry info returned by check_ztf_fps
    """
    for typp in sn_list.keys():
        typ = typp.replace(' ', '')
        if not os.path.exists(os.path.join(base_path, typ)):
            os.mkdir(os.path.join(base_path, typ))
        for subtypp in sn_list[typp].keys():
            subtyp = subtypp.replace(' ', '')
            if not os.path.exists(os.path.join(base_path, typ, subtyp)):
                os.mkdir(os.path.join(base_path, typ, subtyp))

            for sn in sn_list[typp][subtypp]:
                if sn['ra'] == fp_info['ra'] and sn['dec'] == fp_info['dec']:
                    ### Check if the directory for the SN exists
                    if not os.path.exists(os.path.join(base_path, typ, subtyp, sn['name'])):
                        os.mkdir(os.path.join(base_path, typ, subtyp, sn['name']))

                    ### Check if the ZTF file exists, and if not, download it
                    if not os.path.exists(os.path.join(base_path, typ, subtyp, sn['name'], fp_info['filename'])):
                        ztf_fp_file = os.popen(fp_info['lc']).read()
                        os.popen('mv {} {}'.format(fp_info['filename'], os.path.join(base_path, typ, subtyp, sn['name'], fp_info['filename'])))


def process_ztf_fp(file, jdstart, jdend):

    full_df = pd.read_csv(file, delim_whitespace=True, comment='#', header=0, names=['sindex', 'field', 'ccdid', 'qid', 'filter',
                            'pid', 'infobitssci', 'sciinpseeing',
                            'scibckgnd', 'scisigpix', 'zpmaginpsci',
                            'zpmaginpsciunc', 'zpmaginpscirms',
                            'clrcoeff', 'clrcoeffunc', 'ncalmatches',
                            'exptime', 'adpctdif1', 'adpctdif2',
                            'diffmaglim', 'zpdiff', 'programid', 'jd',
                            'rfid', 'forcediffimflux',
                            'forcediffimfluxunc', 'forcediffimsnr',
                            'forcediffimchisq', 'forcediffimfluxap',
                            'forcediffimfluxuncap', 'forcediffimsnrap',
                            'aperturecorr', 'dnearestrefsrc',
                            'nearestrefmag', 'nearestrefmagunc',
                            'nearestrefchi', 'nearestrefsharp',
                            'refjdstart', 'refjdend', 'procstatus'])

    # Get rid of bad data
    df = full_df[(full_df['infobitssci'] < 33554432) & (full_df['scisigpix'] <= 25) & (full_df['sciinpseeing'] <= 4)]
    df = full_df.reset_index(drop=True)

    # Make baseline correction
    colors = {'ZTF_g': 'green', 'ZTF_r': 'red', 'ZTF_i': 'brown'}

    for filt in ['ZTF_g', 'ZTF_r', 'ZTF_i']:
        ### Get unique values of field, ccdid, qid
        fields = set(df[df['filter'] == filt]['field'].values)
        ccdids = set(df[df['filter'] == filt]['ccdid'].values)
        qids = set(df[df['filter'] == filt]['qid'].values)        
        
        for field in fields:
            for ccdid in ccdids:
                for qid in qids:
                    current_df = df[(df['filter'] == filt) & (df['field'] == field) & (df['ccdid'] == ccdid) & (df['qid'] == qid) & ((df['jd'] < jdstart - 20) | (df['jd'] > jdend + 20))]
                    if len(current_df.index) < 30:
                        print('Not enough epochs to estimate baseline for ', filt)
                        continue

                    plt.errorbar(current_df['jd'], current_df['forcediffimflux'], yerr=abs(current_df['forcediffimfluxunc']), fmt='o', color=colors[filt], mec='k')
                    median_flux = median(current_df['forcediffimflux'])
                    print(filt, median_flux)
                    df.loc[(df['filter'] == filt) & (df['field'] == field) & (df['ccdid'] == ccdid) & (df['qid'] == qid), 'forcediffimflux'] -= median_flux
        
        
    plt.show()

    return df


def get_mag_from_ztf_fp(full_df):
    
    colors = {'ZTF_g': 'green', 'ZTF_r': 'red', 'ZTF_i': 'brown'}
    ztf_data = {}
    for filt in ['ZTF_g', 'ZTF_r', 'ZTF_i']:
        
        df = full_df[full_df['filter'] == filt]

        df['nearestrefflux'] = 10**(0.4 * (df['zpdiff'] - df['nearestrefmag']))
        df['nearestreffluxunc'] = df['nearestrefmagunc'] * df['nearestrefflux'] / 1.0857

        snt = 3
        snu = 5

        df.drop_duplicates(['pid', 'forcediffimflux', 'forcediffimfluxunc'],
                              inplace=True)


        df = df.reset_index(drop=True)
        good = np.where((df['forcediffimflux'] / df['forcediffimfluxunc'] > snt))[0]
        bad = np.where((df['forcediffimflux'] / df['forcediffimfluxunc'] <= snt))[0]
        df['mag'] = df['zpdiff'][good] - 2.5*np.log10(df['forcediffimflux'][good])
        df['sigma_mag'] = 1.0857 * df['forcediffimfluxunc'] / df['forcediffimflux']
        df['mag'][bad] = df['zpdiff'][bad] - 2.5*np.log10(snu * df['forcediffimfluxunc'][bad])
    

        plt.errorbar(df['jd'][good], df['mag'][good], yerr=df['sigma_mag'][good], fmt='o', color=colors[filt], mec='black')
        plt.scatter(df['jd'][bad], df['mag'][bad], marker='v', color=colors[filt], edgecolor='black')
        
        good_jds = df['jd'][good].to_numpy()
        good_mags = df['mag'][good].to_numpy()
        good_errs = df['sigma_mag'][good].to_numpy()
        
        for i in range(len(good_jds)):
            ztf_data.setdefault(filt.replace('ZTF_', ''), []).append({'mag': good_mags[i], 'err': good_errs[i], 'mjd': good_jds[i]-2400000.5})
        
        
        bad_jds = df['jd'][bad].to_numpy()
        bad_mags = df['mag'][bad].to_numpy()
        
        for i in range(len(bad_jds)):
            ztf_data.setdefault(filt.replace('ZTF_', ''), []).append({'mag': bad_mags[i], 'err': 9999, 'mjd': bad_jds[i]-2400000.5})
        
        
    plt.gca().invert_yaxis()
    plt.show()
    
    return ztf_data


with open('../swift_uvot_reductions/sn_list_with_template_info.json', 'r') as f:
    sn_list = json.load(f)


typ = 'SLSN-II'
subtyp = 'SLSN-II'
duration = 500 #365 days, SN duration to use for baseline estimate
base_path = '/home/cmp5cr/nasa_adap/data/'+typ.replace(' ', '')+'/'+subtyp.replace(' ', '')+'/'

### Reduce the data and save it
for sn in sn_list[typ][subtyp]:
    if os.path.exists(os.path.join(base_path, sn['name'])) and any(['batch' in f for f in os.listdir(os.path.join(base_path, sn['name']))]):
        for f in os.listdir(os.path.join(base_path, sn['name'])):
            if 'batch' in f:
                ### Get jdstart and jdend from sn
                jdstart = Time(sn['discovered'], format='iso', scale='utc').jd - 20
                jdend = Time(sn['discovered'], format='iso', scale='utc').jd + duration
                df = process_ztf_fp(os.path.join(base_path, sn['name'], f), jdstart, jdend)
                ztf_data = get_mag_from_ztf_fp(df)
                if not os.path.isfile(os.path.join(base_path, sn['name'], sn['name']+'_ztf_fp.json')):
                    with open(os.path.join(base_path, sn['name'], sn['name']+'_ztf_fp.json'), 'w+') as newfile:
                        json.dump(ztf_data, newfile, indent=4)
 
### Download any files we're missing from the FPS
#ztf_fp_urls = check_ztf_fps()
#for fp_info in ztf_fp_urls:
#    download_ztf_fp(sn_list, fp_info, '/home/cmp5cr/nasa_adap/data')
