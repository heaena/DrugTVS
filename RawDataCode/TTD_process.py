# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:18:24 2022

@author: Xinrui
"""


import os
import itertools
import pandas as pd


### Table1. Drug_disease
path = "D:/computational_biology/database/TTD/"

with open(path + "P1-05-Drug_disease.txt") as f:
    lines = f.read()
    
drugs = lines.split("\n\n")[2: ]
drugs = drugs[0].split("\n\t\n")   #split by blank line with \t


drugid = [x.split("TTDDRUID\t")[1].split("\n")[0] for x in drugs]
drugname = [x.split("DRUGNAME\t")[1].split("\n")[0] for x in drugs]
indicate = [';'.join(str(e) for e in x.split("\nINDICATI\t")[1:]) for x in drugs]

drug_disease = pd.DataFrame(list(zip(drugid, drugname, indicate)), columns=("Drug_ID","Drug_Name","Indicate"))


### Table2. Drug_compound
with open(os.path.join(path, "P1-02-TTD_drug_download.txt")) as f:
    lines = f.read()

drugs = lines.split("\t\t\n\t\t\n")[1: ]
drugs = drugs[0].split("\n\t\t\n")

drugid = [x.split("\tDRUG__ID")[0] for x in drugs]

smiles = list()
for x in drugs:
    try:
        smiles.append(x.split("DRUGSMIL\t")[1].split("\n")[0])
    except:
        smiles.append('NULL')
        
compound = pd.DataFrame(list(zip(drugid, smiles)), columns=("Drug_ID","smiles"))

## merge two tables, remove smiles= 'NULL' or [Sb]
drug_compound_disease = drug_disease.merge(compound, how="left", on="Drug_ID")
drug_compound_disease = drug_compound_disease[~drug_compound_disease['smiles'].isin(['NULL','[Sb]'])]



### Table3. Target
with open(os.path.join(path, "P1-01-TTD_target_download.txt")) as f:
    lines = f.read()
    
targets = lines.split("\n\n")[2]
targets = targets.split("\t\t\n\t\t\t\t\n")

## find targets match Drug_ID
match_targets = dict()
for x in drug_compound_disease['Drug_ID']:
    match_targets.update({x: [target for target in targets if x in target]})

targetid, uniprotid, targetname, genename, sequence, pdbid = dict(), dict(), dict(), dict(), dict(), dict()
for k,v in match_targets.items():
        targetid.update({k: [obj.split("\tTARGETID")[0] if "\tTARGETID" in obj else [] for obj in v or [] ]})
        uniprotid.update({k: [obj.split("UNIPROID\t")[1].split("\t\t\n")[0] if "UNIPROID\t" in obj else [] for obj in v or [] ]})
        targetname.update({k: [obj.split("TARGNAME\t")[1].split("\t\t\n")[0] if "TARGNAME\t" in obj else [] for obj in v or [] ]})  
        genename.update({k: [obj.split("GENENAME\t")[1].split("\t\t\n")[0] if "GENENAME\t" in obj else [] for obj in v or [] ]})
        sequence.update({k: [obj.split("SEQUENCE\t")[1].split("\t\t\n")[0] if "SEQUENCE\t" in obj else [] for obj in v or [] ]})  
        pdbid.update({k: [obj.split("PDBSTRUC\t")[1].split("\t\t\n")[0] if "PDBSTRUC\t" in obj else [] for obj in v or [] ]})

    
targetid = (pd.DataFrame.from_dict(targetid, orient='index').T.melt(var_name='Drug_ID', value_name='Target_ID').dropna(subset=['Target_ID']))
uniprotid = (pd.DataFrame.from_dict(uniprotid, orient='index').T.melt(var_name='Drug_ID', value_name='Uniprot_ID').dropna(subset=['Uniprot_ID']))
targetname = (pd.DataFrame.from_dict(targetname, orient='index').T.melt(var_name='Drug_ID', value_name='Target_Name').dropna(subset=['Target_Name']))
genename = (pd.DataFrame.from_dict(genename, orient='index').T.melt(var_name='Drug_ID', value_name='Gene_Name').dropna(subset=['Gene_Name']))
sequence = (pd.DataFrame.from_dict(sequence, orient='index').T.melt(var_name='Drug_ID', value_name='Sequence').dropna(subset=['Sequence']))
pdbid = (pd.DataFrame.from_dict(pdbid, orient='index').T.melt(var_name='Drug_ID', value_name='PDB_ID').dropna(subset=['PDB_ID']))

targets_info = targetid.merge(uniprotid, left_index=True, right_index=True)\
    .merge(targetname, left_index=True, right_index=True)\
        .merge(genename, left_index=True, right_index=True)\
            .merge(sequence, left_index=True, right_index=True)\
                .merge(pdbid, left_index=True, right_index=True)

targets_info = targets_info.rename(columns={"Drug_ID_x": "Drug_ID", "Drug_ID_y": "Drug_ID"})
targets_info = targets_info.loc[:,~targets_info.columns.duplicated()].copy()

#### merge with drug_compound_disease
TTD_all = drug_compound_disease.merge(targets_info, how="left", on="Drug_ID")
TTD_all.rename(columns={'smiles':'Ligand SMILES'}, inplace=True)    

TTD_all.to_csv("D:/projects/CPI_disease/TTD_all.csv", index=None)





