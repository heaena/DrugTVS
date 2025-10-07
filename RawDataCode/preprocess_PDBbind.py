# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 22:13:15 2023

@author: Xinrui
"""

# ===============================================================================================
# 
# PDBbind dataset is downloaded from http://www.pdbbind.org.cn/download.php 
#
# 1) All PDBbind relevant files should be stored in your local directory, 
#      and named as "PDBbind_v2020" or "PDBbind_vXXXX" for your use of version.  
# 2) The subfolders of "refined" and "other" were concatenated, so that the directory level
#      was "./PDBbind_v2020/PDB ID/". 
# 3) The subfolder "index" remained under "./PDBbind_v2020/", which contains
#      ligand-protein complexes binding affinity, pdb chain-uniprot mapping, etc;

# ===============================================================================================


import os
import argparse
import pandas as pd
import re

from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromMol2File

args = argparse.ArgumentParser(description='Argparse for PDBbind data preprocessing')
args.add_argument('-input_folder', type=str, default='D:/computational_biology/database/PDBbind_v2020/', help='local directory of PDBbind data downloaded from website')
args.add_argument('-output_folder', type=str, default='./DataProcessed/', help='local directory to save cleaned version of binding data')

params, _ = args.parse_known_args()


input_folder = params.input_folder
output_folder = params.output_folder
os.makedirs(output_folder, exist_ok=True)


def load_data(dataset: str):
    
    with open(input_folder + os.sep + "index" + os.sep + dataset) as f:
        data = []
        for _ in range(6):
            next(f)
        for line in f:
            data.append(line.split(" "))
    return data


def get_smiles(PDB_ID: str):  # ligand with sdf format    
    mol_path = os.path.normpath(input_folder + os.sep + PDB_ID + os.sep + PDB_ID + "_ligand.sdf")
    mol = Chem.SDMolSupplier(mol_path)
    ligand_smiles = Chem.MolToSmiles(mol[0])
    return ligand_smiles


def get_smiles2(PDB_ID: str):  # ligand with mol2 format   
    mol_path = os.path.normpath(input_folder + os.sep + PDB_ID + os.sep + PDB_ID + "_ligand.mol2")
    mol = MolFromMol2File(mol_path)
    ligand_smiles = Chem.MolToSmiles(mol)
    return ligand_smiles


    


if __name__ == '__main__':   
    affinity_data = load_data(dataset = "INDEX_general_PL_data.2020")   
    affinity_data = [[x for x in ele if x != ''] for ele in affinity_data]
    affinity_data = pd.DataFrame(affinity_data).iloc[:,0:5]
    
    # extract affinity value and unit from 5th column
    affinity_data['value_raw'] = [re.findall(r"[-+]?(?:\d*\.*\d+)", x)[-1] for x in affinity_data[4]]
    affinity_data['value_raw'] = pd.to_numeric(affinity_data['value_raw'])
    affinity_data['measure'] = [ele.split("=")[0] for ele in affinity_data[4]]
    affinity_data['unit'] = affinity_data[4].str[-2:]
    
    # unify all units to nM (same as BindingDB)
    affinity_data['affinity_nM'] = float()
    
    nM_i = affinity_data[ affinity_data['unit']=='nM' ].index
    affinity_data['affinity_nM'][nM_i] = affinity_data['value_raw'][nM_i]
    mM_i = affinity_data[ affinity_data['unit']=='mM' ].index
    affinity_data['affinity_nM'][mM_i] = affinity_data['value_raw'][mM_i]*(10**6)
    uM_i = affinity_data[ affinity_data['unit']=='uM' ].index
    affinity_data['affinity_nM'][uM_i] = affinity_data['value_raw'][uM_i]*(10**3)
    pM_i = affinity_data[ affinity_data['unit']=='pM' ].index
    affinity_data['affinity_nM'][pM_i] = affinity_data['value_raw'][pM_i]*(10**-3)
    fM_i = affinity_data[ affinity_data['unit']=='fM' ].index
    affinity_data['affinity_nM'][fM_i] = affinity_data['value_raw'][fM_i]*(10**-6)
    
    
    
    
    ########## convert affinity to binary: if affinity <= 1000nM, bind; else not
    affinity_data['label'] = ''
    
    for i in range(len(affinity_data)):
        if ('=' in affinity_data.iloc[i, 4]) or ('~' in affinity_data.iloc[i, 4]):
            if affinity_data['affinity_nM'][i] <= 1000:
                affinity_data['label'][i] = 1
            else:
                affinity_data['label'][i] = 0
        elif ('>' in affinity_data.iloc[i, 4]) and (affinity_data['affinity_nM'][i] > 1000):
            affinity_data['label'][i] = 0
        elif ('<' in affinity_data.iloc[i, 4]) and (affinity_data['affinity_nM'][i] <= 1000):
            affinity_data['label'][i] = 1
        else:
            affinity_data['label'][i] = None
    affinity_data = affinity_data[~affinity_data['label'].isnull()]
    
    # delete unnecessary columns and rename columns
    affinity_data = affinity_data.loc[:,[0,'measure','affinity_nM','label']]
    affinity_data = affinity_data.rename(columns = {0:'PDB_ID'})
    
    # add Ligand_SMILES to affinity_data, from subfolder of each pdbID folder downloaded from PBDbind.org   
    smiles, pdbs = [], []
    for pdb in affinity_data['PDB_ID']:
        try:
            s = get_smiles(PDB_ID = pdb)
            smiles.append(s)
            pdbs.append(pdb)
        except:
            pass
       
    smiles = pd.DataFrame(pdbs, smiles).reset_index()
    smiles.columns = ['Ligand_SMILES','PDB_ID']
    
    affinity_data = affinity_data.merge(smiles, how="right", on="PDB_ID")
    
    # add \t to PDB ID so that it won't be changed when write as csv file
    affinity_data['PDB_ID'] = affinity_data['PDB_ID'].astype(str) + '\t'
    affinity_data['PDB_ID'] = [ele.upper() for ele in affinity_data['PDB_ID']]
    
    # reorder columns
    affinity_data = affinity_data.loc[:,['PDB_ID','Ligand_SMILES','measure','affinity_nM','label']]
    
    # save processed version to local directory "./DataProcessed/"
    affinity_data.to_csv(output_folder + os.sep + "PDBbind.csv", index=None, sep=',')


