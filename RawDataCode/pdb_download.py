# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:37:07 2023

@author: Xinrui
"""

# ==============================================================
#
#        To download pdb format files from rcsb.org
# 
#===============================================================


import os
import urllib
import argparse
import pandas as pd
import glob



args = argparse.ArgumentParser(description='Argparse for PDB format file downloading')
args.add_argument('-input_folder', type=str, default='./DataProcessed/',help='local directory of dataset contains PDB_ID needs to be downloaded')
args.add_argument('-output_folder', type=str, default='./PDB_rawfiles_BDB_MOAD/', help='local directory folder for downloaded pdb files')
args.add_argument('-type', type=str, default='rcsb', help='download from either rcsb or AlphaFold')


params, _ = args.parse_known_args()

    
    


def pdb_download(id_list, output_path):  # download pdb file from rcsb.org
    for ID in id_list: 
        url = 'http://files.rcsb.org/download/' + ID + '.pdb'
        outfile = output_path + os.sep + ID.upper() + '.pdb'
        try: 
            urllib.request.urlretrieve(url, outfile)
        except:
            pass
        

        
def AF_download(id_list, output_path):  # download pdb file from AlphaFold
    for ID in id_list:
        url = 'https://alphafold.ebi.ac.uk/api/prediction/' + ID + '?key=AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94'
        outfile = output_path + os.sep + ID.upper() + '.json'
        try:
            urllib.request.urlretrieve(url, outfile)
        except:
            pass
            


# download unique PDB_ID for KIKD+IC50 of BDB and MOAD combined
if __name__ == "__main__":    
    input_folder = params.input_folder
    output_folder = params.output_folder
    d_type = params.type

    # download pdb format files
    if not os.path.isdir(output_folder):
      os.mkdir(output_folder)

    data = pd.read_csv(input_folder + "All_data_inductive.csv")
    id_list = data.loc[data['source']!= 'PDBbind', 'PDB_ID']
    id_list = [ele.strip() for ele in id_list]
    id_have = glob.glob(output_folder + os.sep + "*.pdb")
    id_have = [ele.split("\\")[1][0:4] for ele in id_have]
    id_download = [ele for ele in id_list if ele not in id_have]
    
    if d_type == 'rcsb':
        pdb_download(id_download, output_folder)
    elif d_type == 'AlphaFold':
        AF_download(id_download, output_folder)
    else:
        print("Please select download type of either 'rcsb' or 'AlphaFold'!")
    
    
    