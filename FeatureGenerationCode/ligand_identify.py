# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:09:21 2023

@author: Xinrui
"""


import re



########## 1. download and preprocess cofactor 3-letter-code
# r = requests.get("https://www.ebi.ac.uk/pdbe/api/pdb/compound/cofactors")
# cofactor = r.json()

# cofactor_het = []
# for v1 in cofactor.values():
#     dict_sub = v1[0]
#     v2 = dict_sub['cofactors']
    
#     cofactor_het.extend(v2)


# with open('./Data/cofactor_het.txt', 'w') as f:
#     for item in cofactor_het:
#       f.write("%s\n" % item)
    


# load cofactor het_ids
# with open('D:/projects/CPI_disease/cofactor_het.txt', 'r') as f:
#     cofactor_het = f.read().split("\n")


def lig_id(pdb_file, lig_in_dataset, chain_in_dataset=None):
    # 1. load raw pdb file downloaded from rcsb.org
    with open(pdb_file, "r") as f:
        lines = f.readlines()
    
    # 2. extract lines contain "HET" and "HETNAM" info
    het = []
    for line in lines:
        if line.startswith("HET "):
            het.append(line)
        elif line.startswith("HETNAM "):
            het.append(line)
    
    # 3. extract 3-letter-code of het name from "HET" line
    hetname = []
    for line in het:
        if line.startswith("HET "):
            hetname.append(line[7:10]) # refer to guide of PDB file format 
    
    # 4. check if the het name is ion, and remove ion from hetname
    ion_name = []
    for line in het:
        if re.search(r'\b' + "ION" + r'\b', line):
            ion_name.append(line[11:14]) # refer to guide of PDB file format 
    
    hetname = [ele for ele in hetname if ele not in ion_name]
    
    # 5. check if the het name is cofactor, and remove cofactor from hetname
    # if len(set(hetname)) >1:
    #     cofactor_name = [ele for ele in hetname if ele in cofactor_het]
    # else:
    #     cofactor_name = []
    
    # hetname = [ele for ele in hetname if ele not in cofactor_name]
    # hetname = list(dict.fromkeys(hetname))
    
    # 6. check if hetname extracted from pdb file matched with ligand name shown in the dataset
    hetname_try = [ele for ele in hetname if ele in lig_in_dataset]
    if hetname_try != []:
        hetname = hetname_try[0]    
    elif len(hetname) == 0:
        hetname = lig_in_dataset
    else:
        hetname = hetname[0]
    
    # 7. extract corresponding chain
    if chain_in_dataset is None:    
        for i in range(len(het)):
            if het[i].startswith("HET ") and hetname in het[i]:
                chain = het[i][12]
    else:
        chain = chain_in_dataset
    
    try:
        chain
    except:
        chain = None
    
    if chain is None:
        hetname_chain = []
    else:
        hetname_chain = (hetname, chain)
    
    return hetname_chain

