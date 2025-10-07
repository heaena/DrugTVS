# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 00:38:02 2023

@author: Xinrui
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 00:07:24 2023

@author: Xinrui
"""


# =========================================================================================
#
#  BindingDB_all.tsv data was originally downloaded from https://www.bindingdb.org/
# 
# =========================================================================================




import os 
import argparse
import pandas as pd
import re

args = argparse.ArgumentParser(description='Argparse for BindingDB data preprocessing')
args.add_argument('-input_folder', type=str, default='D:/computational_biology/database/BindingDB/', help='local directory of BindingDB data downloaded from website')
args.add_argument('-output_folder', type=str, default='./DataProcessed/', help='local directory to save cleaned version of binding data')

params, _ = args.parse_known_args()

input_folder = params.input_folder
output_folder = params.output_folder
os.makedirs(output_folder, exist_ok=True)



if __name__ == '__main__': 
    with open(input_folder + os.sep + "BindingDB_All.tsv", encoding="gb18030", errors='ignore') as f:
        data = []
        for line in f:
            data.append(line.split("\t"))  

    columns = data[0][0:50]
    del data[0]
    df = pd.DataFrame(data)
    # drop redundant columns
    df = df.drop(df.iloc[:, 50:638],axis = 1)
    df.columns = columns

    # # load PDB_ID+chain and uniprot ID mapping file
    # pdb_chain_uniprot = pd.read_csv(input_folder + os.sep + "uniprot_segments_observed.tsv", sep='\t', skiprows=1)
    # pdb_chain_uniprot['PDB'] = pdb_chain_uniprot['PDB'].astype(str) + '\t'
    # pdb_chain_uniprot = pdb_chain_uniprot.loc[:,['PDB','CHAIN','SP_PRIMARY']]
    # pdb_chain_uniprot = pdb_chain_uniprot.drop_duplicates()

    # pdb_chain_uniprot = pdb_chain_uniprot.fillna('').groupby(['PDB','SP_PRIMARY'], as_index=False).agg({'CHAIN' : ', '.join})



    affinity_data = df[['Ligand SMILES','IC50 (nM)','Ki (nM)','Kd (nM)',\
                        'Ligand HET ID in PDB','PDB ID(s) for Ligand-Target Complex',\
                        'PDB ID(s) of Target Chain','BindingDB Target Chain Sequence']]


    # remove rows of which either Ki Kd, or IC50 is blank, or target sequence is blank
    affinity_data = affinity_data[(affinity_data['Ki (nM)'] != '') | (affinity_data['Kd (nM)'] != '') | (affinity_data['IC50 (nM)'] != '')].reset_index(drop=True)
    affinity_data = affinity_data[affinity_data['BindingDB Target Chain Sequence'] != ''].reset_index(drop=True)


    ####### drop rows of which either PDB1 or PDB2 is empty
    affinity_data = affinity_data[affinity_data['PDB ID(s) for Ligand-Target Complex'] != ''].reset_index(drop=True)
    affinity_data = affinity_data[affinity_data['PDB ID(s) of Target Chain'] != ''].reset_index(drop=True)

    # two lists of PDB_ID
    PDB1 = list(affinity_data['PDB ID(s) for Ligand-Target Complex'])  # PDB_ID associated with ligand
    PDB1 = [ele.split(',') for ele in PDB1]
    PDB2 = affinity_data['PDB ID(s) of Target Chain']  # PDB_ID associated with protein
    PDB2 = [ele.split(',') for ele in PDB2]

    # keep rows of PDB1 and PDB2 contain same PDB_ID
    keep = [[x for x in PDB1[i] if x in PDB2[i]] for i in range(len(affinity_data))]
    keep_index = [i for i, x in enumerate(keep) if x!=[]]
    affinity_data = affinity_data.iloc[keep_index, :]

    # add PDB_ID of both associated with ligand and protein to dataset
    keep = list(map(','.join, keep))
    affinity_data['PDB_ID'] = [keep[i] for i in keep_index]
    
    # make PDB_IDs into multiple rows
    affinity_data['PDB_ID'] = affinity_data['PDB_ID'].str.split(',')
    affinity_data = affinity_data.explode('PDB_ID')
    
    ####### concatenate Ki,Kd,IC50 into one column
    Ki = affinity_data.drop(['IC50 (nM)','Kd (nM)'], axis=1)
    Kd = affinity_data.drop(['IC50 (nM)','Ki (nM)'], axis=1)
    IC50 = affinity_data.drop(['Ki (nM)','Kd (nM)'], axis=1)
    
    Ki = Ki.rename(columns = {'Ki (nM)':'affinity'})
    Kd = Kd.rename(columns = {'Kd (nM)':'affinity'})
    IC50 = IC50.rename(columns = {'IC50 (nM)':'affinity'})
    
    affinity_data = pd.concat([Ki,Kd,IC50]).reset_index(drop=True)
    affinity_data['measure'] = ['Ki']*len(Ki) + ['Kd']*len(Kd) + ['IC50']*len(IC50)
    
    affinity_nM = []
    for x in affinity_data['affinity']:
        try:               
            value = re.findall(r"[-+]?(?:\d*\.*\d+)", x)[-1]  # find digits
        except:
            value = None
        affinity_nM.append(value)
                     
    affinity_data['affinity_nM'] = pd.to_numeric(affinity_nM)
    affinity_data = affinity_data[~affinity_data['affinity_nM'].isnull()].reset_index(drop=True)

    affinity_data = affinity_data.drop_duplicates(['PDB_ID','Ligand SMILES','measure'])
    
    
    ########## convert affinity to binary: if affinity <= 1000nM, bind; else not
    affinity_data['label'] = ''
 
    for i in range(len(affinity_data)):
        if ('>' in affinity_data['affinity'][i]) and (affinity_data['affinity_nM'][i] > 1000):
            affinity_data['label'][i] = 0
        elif ('<' in affinity_data['affinity'][i]) and (affinity_data['affinity_nM'][i] <= 1000):
            affinity_data['label'][i] = 1
        elif affinity_data['affinity_nM'][i] > 1000:
            affinity_data['label'][i] = 0
        elif affinity_data['affinity_nM'][i] <= 1000:
            affinity_data['label'][i] = 1
        else:
            affinity_data['label'][i] = None
        
    affinity_data = affinity_data[~affinity_data['label'].isnull()]
    
    # remove unnecessary columns
    affinity_data = affinity_data.rename(columns = {'Ligand HET ID in PDB':'het_name',
                                                    'Ligand SMILES':'Ligand_SMILES'})
    affinity_data = affinity_data.loc[:,['PDB_ID','Ligand_SMILES','measure','affinity_nM','label','het_name']]
    
    # add \t to PDB_ID and het_name so that it won't be changed when write as csv file
    affinity_data['PDB_ID'] = affinity_data['PDB_ID'].astype(str) + '\t'
    affinity_data['het_name'] = affinity_data['het_name'].astype(str) + '\t'
    
    ### save processed version to local directory "./DataProcessed/"
    affinity_data.to_csv(output_folder + os.sep + 'BDB.csv', index=None, sep=',')








