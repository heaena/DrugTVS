# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:36:20 2024

@author: Xinrui
"""

import requests
import pandas as pd
import glob
import shutil
import os


def download_sdf_files(ids, output_directory):
    base_url = "https://zinc20.docking.org/substances/"

    for compound_id in ids:
        url = base_url + compound_id + ".sdf"
        response = requests.get(url)

        if response.status_code == 200:
            output_file = output_directory + compound_id + ".sdf"
            with open(output_file, "wb") as file:
                file.write(response.content)                
        else:
            None

# select top 1000 and bottom 1000 of molecules in the prediction output to download sdf
pred = pd.read_csv("D:/projects/CPI_disease/breast_cancer/screening_ZINC20_predictions_all_pairs.csv")
pred = pred.loc[pred.groupby(pred['Screening SMILES'].astype(str))['Similarity Scores'].idxmin()]


input_mol_files = glob.glob("D:/Projects/CPI_disease/zinc20/" + '*.txt')
dfs_mol = []
for file in input_mol_files:
    df = pd.read_csv(file, header=None, delimiter='\t')
    dfs_mol.append(df)
df_mol = pd.concat(dfs_mol).reset_index(drop=True)
df_mol.columns = ['zinc_id','Screening SMILES']
df_mol = df_mol.drop_duplicates(['Screening SMILES'])

# merge prediction with zinc id
pred = pd.merge(pred, df_mol, how='left', on='Screening SMILES')

top1000 = pred.sort_values(by=['Similarity Scores']).reset_index(drop=True)[0:1000]
bottom1000 = pred.sort_values(by=['Similarity Scores']).reset_index(drop=True)[(len(pred)-1000):]


compound_ids = pd.concat([top1000['zinc_id'], bottom1000['zinc_id']]).tolist()
output_dir = "D:/projects/CPI_disease/breast_cancer/sdf2000/"

download_sdf_files(compound_ids, output_dir)


###############
# split sdf file by pdb ID
def split_sdf(PDB_ID):
    zinc_ids = pred2000[pred2000['PDB_ID']==PDB_ID]['zinc_id'].tolist()
    move_path = "D:/projects/CPI_disease/breast_cancer/instaDock_result/" + PDB_ID + os.sep
    os.makedirs(move_path, exist_ok=True)
    
    for zinc_id in zinc_ids:
        shutil.copy(output_dir + zinc_id + ".sdf",
                    move_path + zinc_id + ".sdf")
    
pred2000 = pd.concat([top1000, bottom1000])
PDB_IDs = set(pred2000['PDB_ID'])
for pdb_id in PDB_IDs:
    split_sdf(pdb_id)


################
# 
pred2000['mark'] = [1]*1000 + [0]*1000
pred_dock = pd.read_excel("D:/projects/CPI_disease/breast_cancer/instaDock_result/docking_pred_binding.xlsx")
pred_dock = pd.merge(pred_dock, pred2000, how='left', on='zinc_id')
pred_dock.to_csv("D:/projects/CPI_disease/breast_cancer/instaDock_result/docking_pred_binding.csv", index=None)

