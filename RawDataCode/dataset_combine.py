# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:19:41 2023

@author: Xinrui
"""



# ===============================================================================================
# Data concatenation: PDBbind + BindingMOAD + BindingDB 
#
# A) Two forms of dataset combined with the three measurements (Ki+Kd+IC50):
#      1) All_data_transductive (with duplicates in protein and ligand respectively);
#      2) All_data_inductive (with unique data in protein and ligand)
# B) Two forms of dataset combined with the two measurements (Ki+Kd):
#      1) KIKD_transductive (with duplicates in protein and ligand respectively);
#      2) KIKD_inductive (with unique data in protein and ligand)

# ===============================================================================================



import os
import pandas as pd
import argparse
from itertools import compress


args = argparse.ArgumentParser(description='Argparse for data concatenation')
args.add_argument('-input_folder', type=str, default='./DataProcessed/', help='local directory of the processed data')
args.add_argument('-output_folder', type=str, default='./DataProcessed/', help='local directory to save the concatenated data')

params, _ = args.parse_known_args()

input_folder = params.input_folder
output_folder = params.output_folder



if __name__ == '__main__':  
    PDBbind = pd.read_csv(input_folder + os.sep + "PDBbind.csv")
    MOAD = pd.read_csv(input_folder + os.sep + "BindingMOAD.csv")
    BDB = pd.read_csv(input_folder + os.sep + "BDB.csv")


    ### A)
    All_data = pd.concat([PDBbind, MOAD, BDB]).reset_index(drop=True)
    All_data['source'] = ['PDBbind']*len(PDBbind) + ['MOAD']*len(MOAD) + ['BDB']*len(BDB)
    All_data_transductive = All_data.drop_duplicates(['PDB_ID','Ligand_SMILES']).reset_index(drop=True)

    All_data_inductive = All_data_transductive.drop_duplicates(['PDB_ID'])
    All_data_inductive = All_data_inductive.drop_duplicates(['Ligand_SMILES']).reset_index(drop=True)




    ### B)
    KIKD_index = ['IC50' not in ele.upper() for ele in All_data_transductive['measure']]
    KIKD_index = list(compress(range(len(KIKD_index)), KIKD_index))
    KIKD_transductive = All_data_transductive.loc[KIKD_index]

    KIKD_inductive = KIKD_transductive.drop_duplicates(['PDB_ID'])
    KIKD_inductive = KIKD_inductive.drop_duplicates(['Ligand_SMILES']).reset_index(drop=True)



    ######## save processed 4 datasets to local directory "./DataProcessed/"
    All_data_transductive.to_csv(output_folder + os.sep + "All_data_transductive.csv", index=None, sep=",")
    All_data_inductive.to_csv(output_folder + os.sep + "All_data_inductive.csv", index=None, sep=",")

    KIKD_transductive.to_csv(output_folder + os.sep + "KIKD_transductive.csv", index=None, sep=",")
    KIKD_inductive.to_csv(output_folder + os.sep + "KIKD_inductive.csv", index=None, sep=",")
