# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 19:29:11 2023

@author: Xinrui
"""

# ===============================================================================================
#
#  BindingMOAD dataset is downloaded from http://www.bindingmoad.org/Home/download
#    (Binding data  -- only those w/ binding data)
#
# ===============================================================================================



import os
import argparse
import pandas as pd
import numpy as np


args = argparse.ArgumentParser(description='Argparse for Binding MOAD data preprocessing')
args.add_argument('-input_folder', type=str, default='D:/computational_biology/database/BindingMOAD/', help='local directory of BindingMOAD data downloaded from website')
args.add_argument('-output_folder', type=str, default='./DataProcessed/', help='local directory to save cleaned version of binding data')

params, _ = args.parse_known_args()

input_folder = params.input_folder
output_folder = params.output_folder
os.makedirs(output_folder, exist_ok=True)




if __name__ == '__main__':
    with open(input_folder + os.sep + "every_bind.csv") as f:
        data = []
        for _ in range(11):
            next(f)
        for line in f:
            data.append(line.strip().split(","))  

    affinity_data = pd.DataFrame(data)

    # remove redundant rows and columns
    affinity_data = affinity_data.drop(affinity_data.columns[[0, 1, 10]],axis = 1)
    affinity_data.columns = ['PDB_ID','Range','validity','measure','symbol','value_raw','unit','Ligand_SMILES']

    remove_index = [i for i in range(len(affinity_data)) if sum(affinity_data.iloc[i,:]=='')==len(affinity_data.columns) ]
    affinity_data = affinity_data.drop(index = remove_index)
    affinity_data['PDB_ID'].replace('', np.nan, inplace=True)
    affinity_data = affinity_data.fillna(method='ffill')

    # remove rows if measurement or unit column is empty
    affinity_data['measure'].replace('', np.nan, inplace=True)
    affinity_data['unit'].replace('', np.nan, inplace=True)
    affinity_data = affinity_data[~affinity_data['measure'].isnull()]
    affinity_data = affinity_data[~affinity_data['unit'].isnull()]

    # drop rows that Ligand_SMILES empty
    affinity_data['Ligand_SMILES'].replace('', np.nan, inplace=True)
    affinity_data = affinity_data[~affinity_data['Ligand_SMILES'].isnull()].reset_index(drop=True)

    # unify all units to nM (same as BindingDB)
    affinity_data['value_raw'] = pd.to_numeric(affinity_data['value_raw'])
    affinity_data['affinity_nM'] = ''

    nM_i = affinity_data[ affinity_data['unit']=='nM' ].index
    affinity_data['affinity_nM'][nM_i] = affinity_data['value_raw'][nM_i]
    M_i = affinity_data[ affinity_data['unit']=='M' ].index
    affinity_data['affinity_nM'][M_i] = affinity_data['value_raw'][M_i]*(10**9)
    mM_i = affinity_data[ affinity_data['unit']=='mM' ].index
    affinity_data['affinity_nM'][mM_i] = affinity_data['value_raw'][mM_i]*(10**6)
    uM_i = affinity_data[ affinity_data['unit']=='uM' ].index
    affinity_data['affinity_nM'][uM_i] = affinity_data['value_raw'][uM_i]*(10**3)
    pM_i = affinity_data[ affinity_data['unit']=='pM' ].index
    affinity_data['affinity_nM'][pM_i] = affinity_data['value_raw'][pM_i]*(10**-3)
    fM_i = affinity_data[ affinity_data['unit']=='fM' ].index
    affinity_data['affinity_nM'][fM_i] = affinity_data['value_raw'][fM_i]*(10**-6)
    M_inverse_i = affinity_data[ affinity_data['unit']=='M^-1' ].index
    affinity_data['affinity_nM'][M_inverse_i] = affinity_data['value_raw'][M_inverse_i]*(10**-9)

    # split Range into 3 new columns
    affinity_data[['het_name', 'chain', 'auth_seq_id']] = affinity_data.Range.str.split(":", expand = True)

    # remove duplicates by PDB_ID, hetname and SMILES
    affinity_data = affinity_data.drop_duplicates(subset = ['Ligand_SMILES', 'PDB_ID', 'het_name']).reset_index(drop = True)



    ############# convert affinity to binary: if affinity <= 1000nM, bind; else not
    affinity_data['label'] = ''

    for i in range(len(affinity_data)):
        if ('=' in affinity_data['symbol'][i]) or ('~' in affinity_data['symbol'][i]):
            if affinity_data['affinity_nM'][i] <= 1000:
                affinity_data['label'][i] = 1
            else:
                affinity_data['label'][i] = 0
        elif ('>' in affinity_data['symbol'][i]) and (affinity_data['affinity_nM'][i] > 1000):
            affinity_data['label'][i] = 0
        elif ('<' in affinity_data['symbol'][i]) and (affinity_data['affinity_nM'][i] <= 1000):
            affinity_data['label'][i] = 1
        else:
            affinity_data['label'][i] = None
    affinity_data = affinity_data[~affinity_data['label'].isnull()]

    # remove unnecessary columns
    affinity_data = affinity_data.loc[:,['PDB_ID','Ligand_SMILES','measure','affinity_nM','label','het_name']]
    
    # add \t to PDB_ID and het_name so that it won't be changed when write as csv file
    affinity_data['PDB_ID'] = affinity_data['PDB_ID'].astype(str) + '\t'
    affinity_data['het_name'] = affinity_data['het_name'].astype(str) + '\t'
    
    # save processed data to local directory "./DataProcessed/"
    affinity_data.to_csv(output_folder + os.sep + "BindingMOAD.csv", index=None, sep=",")
