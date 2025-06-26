import json
import os

from astropy.coordinates import SkyCoord
import astropy.units as u
from caat import CAAT
from caat.utils import ROOT_DIR
import pandas as pd

class CSPProcessor:
    """
    Class to process the CSP SESNe dataset from Bianco + 2014
    Contains routines to parse the table of objects and data,
    add new objects to the repository's CAAT file, 
    create new directories for missing objects, and save
    the datasets in a way to be parsed by the SN class.
    """
    def __init__(
        self,
        save: bool = False,
        force: bool = False
    ):
        self.force = force
        self.save = save
        
        self.caat = CAAT().caat
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        self.data_file = 'table6.txt'
        self.object_file = 'table1.txt'

    def _parse_objects(self):
        """
        Parse the object file, retrieve a list of objects,
       and iteratively add them to the CAAT or overwrite them, 
       depending on the init parameters
        """
        df = pd.read_csv(
            os.path.join(
                self.base_path,
                self.object_file
            ),
            skiprows=5,
            delimiter="\t",
            names=[
                "SNName", 
                "CfASpectra", 
                "CfA NIR", 
                "R.A.", 
                "Decl.", 
                "SN Type", 
                "Discovery Date", 
                "Discovery Reference", 
                "Spectroscopic ID"
            ],
            index_col=False
        )

        for _, row in df.iterrows():
            name = row["SNName"].split('^')[0].replace(' ', '')
            if name[:2] == 'SN':
                ra_hms = row["R.A."][:10]
                dec_dms = row["Decl."][:10]
                snsubtype = "SN" + row["SN Type"].split(' ')[0].split('/')[0].strip('-BL').split('-pec')[0].replace('-', '')

                coord = SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.deg))
                #print(name, coord.ra.degree, coord.dec.degree, snsubtype)
                if snsubtype != 'SNIbn':
                    sntype = "SESNe"
                else:
                    sntype = "FBOT"
                new_row = {
                    "Name": name, 
                    "Type": sntype, 
                    "Subtype": snsubtype, 
                    "RA": round(coord.ra.degree, 5),
                    "Dec": round(coord.dec.degree, 5)
                }                
                if len(self.caat[self.caat["Name"] == new_row["Name"]].values) == 0:
                    self.caat = pd.concat([self.caat, pd.DataFrame([new_row])], ignore_index=True)
                elif self.force:
                    self.caat.combine_first(pd.DataFrame([new_row]))
                else:
                    print(f"Metadata for {name} already exists. Use force to overwrite its row in the CAAT file") 

        if self.save:
            CAAT().save_db_file(os.path.join(ROOT_DIR, "data/", "caat.csv"), self.caat, force=self.save)
        else:
            print(self.caat)       

    def _parse_data(self):
        """
        Parse the data file, create new SN data directories as necessary,
        and save or overwrite any existing data files, depending on the init parameters
        """
        df = pd.read_csv(
            os.path.join(
                self.base_path,
                self.data_file
            ),
            skiprows=17,
            delim_whitespace=True,
            names=[
                "SNName",
                "Filter",
                "MJD", 
                "Mag", 
                "Err", 
                "Source", 
            ],
            index_col=False
        )
        sne = list(set(df["SNName"].values))
        for sn in sne:
            data = {}
            for filt in list(set(df[df["SNName"] == sn]["Filter"].values)):
                data[filt.replace('\'', '')] = []
                df_for_filt = df[(df["SNName"] == sn) & (df["Filter"] == filt)]
                for i, row in df_for_filt.iterrows():
                    phot = {}
                    phot["mjd"] = row["MJD"]
                    phot["mag"] = row["Mag"]
                    phot["err"] = row["Err"]
                    data[filt.replace('\'', '')].append(phot)

            # Find or make a directory for the data and save it as a json
            info_row = self.caat[self.caat["Name"] == "SN"+sn]
            if len(info_row) > 0:
                sntype = info_row["Type"].values[0]
                snsubtype = info_row["Subtype"].values[0]
                sn_path = os.path.join(ROOT_DIR, "data/", sntype, snsubtype, "SN"+sn)
                if not os.path.exists(sn_path) and self.save:
                    os.mkdir(sn_path)
                if self.save:
                    filepath = os.path.join(sn_path, "SN"+sn+"_cfa_data_release.json")
                    if self.force or not os.path.exists(filepath):
                        with open(filepath, 'w+') as f:
                            json.dump(data, f, indent=4)
                else:
                    print("====MOCK RUN====")
                    print(f"Making directory {sn_path}")
                    print(f"Dumping data to {os.path.join(sn_path, 'SN'+sn+'_cfa_data_release.json')}")

    def process(self):
        """
        Wrapper function to parse the list of objects in the CSP SESNe
        sample, create directories, and add or overwrite data files,
        depending on the init parameters
        """
        self._parse_objects()
        self._parse_data()


processor = CSPProcessor(save=True)
processor.process()