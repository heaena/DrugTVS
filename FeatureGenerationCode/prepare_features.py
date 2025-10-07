# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:13:45 2023

@author: Xinrui
"""


import os
import pandas as pd
import numpy as np
import torch
import re
import math
import deepchem as dc
from deepchem.models.torch_models.dmpnn import _MapperDMPNN

from FeatureGenerationCode.prepare_pocket import Pocket_feature



# def get_info_from_dataset(input_file, delimiter=','):
#     dataset = pd.read_csv(input_file, header=None, sep=delimiter)
#     remove_idx = [i for i in range(len(dataset)) if len(dataset[0][i]) < 6 or len(dataset[1][i]) >= 500]
#     dataset = dataset.drop(remove_idx).reset_index(drop=True)

#     # sequence = [item.strip() for item in dataset[1]]
#     # smiles_list = [item.strip() for item in dataset[0]]
#     # label = [item for item in dataset[2]]
    
#     return dataset



# def get_info_from_dataset(input_file, delimiter=','):
#     dataset = pd.read_csv(input_file, header=None, sep=delimiter)
#     inductive_dataset = dataset.drop_duplicates(0)
#     inductive_dataset = inductive_dataset.drop_duplicates(1).reset_index(drop=True)
#     remove_idx = [i for i in range(len(inductive_dataset)) if len(inductive_dataset[0][i]) < 6 or len(inductive_dataset[1][i]) >= 500]
#     inductive_dataset = inductive_dataset.drop(remove_idx).reset_index(drop=True)

#     sequence = [item.strip() for item in inductive_dataset[1]]
#     smiles_list = [item.strip() for item in inductive_dataset[0]]
#     label = [item for item in inductive_dataset[2]]
    
#     return sequence, smiles_list, label




def torsion_dist_embeddin(feat_path):
    torsion = torch.load(os.path.join(feat_path, "torsion.pt"))
    dist = torch.load(os.path.join(feat_path, "dist.pt"))
    
    dist_id = [np.digitize(x, bins=(5, 10, 15, 20, 25, 30, 35, 40)) for x in dist]

    torsion_id = []
    for k in range(len(torsion)):
        
        phi, psi = torsion[k][:,0], torsion[k][:,1]
        phi = [math.degrees(x) for x in phi]
        psi = [math.degrees(x) for x in psi]
        region_list = Pocket_feature().torsion_region_bound()
    
        region_idx = []
        for i in range(len(phi)):
            phi_i, psi_i = phi[i], psi[i]
            region_class = [Pocket_feature().torsion_region(region, phi_i, psi_i) for region in region_list]
            try:
                idx = region_class.index(True) + 1
            except:
                idx = 9
            region_idx += [idx]
                
        region_idx = np.array(region_idx)
        torsion_id += [region_idx]
    
    
    return torsion_id, dist_id



# def generate_features(input_file, comp_model_path, radius=1):
#     word2vec_model = word2vec.Word2Vec.load(comp_model_path)
    
#     df = get_info_from_dataset(input_file)
#     df.columns = ['Smiles','Sequence','label']
#     PandasTools.AddMoleculeColumnToFrame(df, smilesCol='Smiles')
#     df = df[df['ROMol'].notnull()]
#     df['Smiles'] = df['ROMol'].map(Chem.MolToSmiles)
#     df['mol-sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], radius)), axis=1)
#     comp_vector = sentences2vec(df['mol-sentence'], word2vec_model).tolist()
#     comp_vector = torch.Tensor(comp_vector)
#     c_mask = torch.ones(len(comp_vector), 300)
    
#     sequence, label_list = df['Sequence'], df['label']
    
#     data_list = [[sequence[k], comp_vector[k], c_mask[k], label_list[k]] for k in range(len(df)) ]
#     return data_list
    
def Compound_feature(mapper):            
    atom_feature, f_ini_atoms_bond, atom_to_incoming_bond, mapping, global_feature = mapper.values 
    atom_feature = torch.from_numpy(atom_feature).float()
    f_ini_atoms_bond = torch.from_numpy(f_ini_atoms_bond).float()
    atom_to_incoming_bond = torch.from_numpy(atom_to_incoming_bond)
    mapping = torch.from_numpy(mapping)
    global_feature = torch.from_numpy(global_feature).float()
    mol_len = len(atom_feature) # molecule length for single mol
    
    compound_feature = [atom_feature, f_ini_atoms_bond, atom_to_incoming_bond, mapping,\
                        mol_len, global_feature]
       
    return compound_feature



def generate_features(input_file, torsion_file):
    torsion_id = torch.load(torsion_file)
    dataset = pd.read_csv(input_file, header=None)
    smiles_list = dataset[0]
    
    feat = dc.feat.DMPNNFeaturizer(features_generators = ['morgan'], is_adding_hs = True)
    comp_index, compound_feature_list = [],[] 
    
    for k in range(len(smiles_list)):  
        try:
            # compound features
            graph = feat.featurize(smiles_list[k])
            mapper = _MapperDMPNN(graph[0])
            compound_feature = Compound_feature(mapper)    
            comp_index += [k] 
            compound_feature_list += [compound_feature]              
        except:
            pass
        
    sequence_list = [dataset[1][i] for i in comp_index]   
    label_list = [dataset[2][i] for i in comp_index] 
    torsion_list = [torsion_id[i] for i in comp_index]
        
    data_list = [[sequence_list[i], compound_feature_list[i][0], compound_feature_list[i][1], 
                  compound_feature_list[i][2], compound_feature_list[i][3], 
                  compound_feature_list[i][4], compound_feature_list[i][5], torsion_list[i],
                  label_list[i]] for i in range(len(sequence_list)) ]
    return data_list
    


### for input dataset: column name should at least include: "PDB_ID", "Ligand_SMILES", "het_name", "label","source"
### the input dataset adding a new column "Pocket_Sequence" was returned along with data_feature_list
def generate_features_w_torsion(dataset, pdb_folder, lig_type = "Ligand_SMILES"):    
    PDB_ID = dataset['PDB_ID']
    ligands = dataset['het_name']
    sources = dataset['source']
    if 'auth_chain' in dataset.columns:
        chains = dataset['auth_chain']
    else:
        chains = None
    
    pocket_seq_list, torsion_pocket_list = [], []
    for k in range(len(PDB_ID)):        
        pdb_id = re.sub(r'\s+', '', PDB_ID[k])
        lig_in_dataset = str(ligands[k])
        source = sources[k]
        if chains != None:
            chain_in_dataset = chains[k]
        else:
            chain_in_dataset = None
        
        # if PDB file is a pocket file processed from PDBbind, use pocket_torsion_w_pocketfile,
        # if PDB file is a raw file downloaded from rcsb, use pocket_torsion_w_hetname
        if source == "PDBbind":
            pdb_file = pdb_folder + 'PDBbind_v2020/' + pdb_id.lower()  + os.sep + pdb_id.lower() + '_protein' + '.pdb'
            pocket_file = pdb_folder + 'PDBbind_v2020/' + pdb_id.lower() + os.sep + pdb_id.lower()  + '_pocket' + '.pdb'
            df_torsion_pocket, PocketSeq = Pocket_feature().pocket_torsion_w_pocketfile(pdb_file, pocket_file)
        else:
            pdb_file = pdb_folder + pdb_id + '.pdb'
            df_torsion_pocket, PocketSeq = Pocket_feature().pocket_torsion_w_hetname(pdb_file, lig_in_dataset, chain_in_dataset)

        if isinstance(df_torsion_pocket, pd.DataFrame):
            torsion_pocket = [df_torsion_pocket[['phi','psi']].to_numpy()]
        else:
            torsion_pocket = ["null"]
            
        torsion_pocket_list += torsion_pocket
        pocket_seq_list += [PocketSeq]

    # remove records that don't generate pocket sequence and torsion properly
    remove_idx = [i for i, e in enumerate(pocket_seq_list) if e == '']
    dataset = dataset.drop(index=remove_idx).reset_index(drop=True)
    for ele in sorted(remove_idx, reverse = True): 
        del pocket_seq_list[ele]
    for ele in sorted(remove_idx, reverse = True): 
        del torsion_pocket_list[ele]
    
    dataset['Pocket_Sequence'] = pocket_seq_list
    # create torsion_id
    torsion_id = Pocket_feature().torsion_embeddin(torsion_pocket_list)
    
    if lig_type == "Screening_SMILES":
        smiles_list = dataset['Screening_SMILES']
    else:
        smiles_list = dataset['Ligand_SMILES']
    
    ### create compound feature
    feat = dc.feat.DMPNNFeaturizer(features_generators = ['morgan'], is_adding_hs = True)
    comp_index, compound_feature_list = [],[] 
    
    for k in range(len(smiles_list)):  
        try:
            # compound features
            graph = feat.featurize(smiles_list[k])
            mapper = _MapperDMPNN(graph[0])
            compound_feature = Compound_feature(mapper)    
            comp_index += [k] 
            compound_feature_list += [compound_feature]              
        except:
            pass
        
    sequence_list = [dataset['Pocket_Sequence'][i] for i in comp_index]   
    label_list = [dataset['label'][i] for i in comp_index] 
    torsion_list = [torsion_id[i] for i in comp_index]
    dataset = dataset.iloc[comp_index].reset_index(drop=True)
        
    data_list = [[sequence_list[i], compound_feature_list[i][0], compound_feature_list[i][1], 
                  compound_feature_list[i][2], compound_feature_list[i][3], 
                  compound_feature_list[i][4], compound_feature_list[i][5], torsion_list[i],
                  label_list[i]] for i in range(len(sequence_list)) ]

    return data_list, dataset




### for input dataset: column[0] = SMILES; column[1] = Protein Sequence; column[2] = label. Column names don't matter.
def generate_features_wo_torsion(input_file):
    dataset = pd.read_csv(input_file, header=None)
    smiles_list = dataset[0]
    
    feat = dc.feat.DMPNNFeaturizer(features_generators = ['morgan'], is_adding_hs = True)    
    comp_index, compound_feature_list = [],[] 
    
    for k in range(len(smiles_list)):  
        try:
            # compound features
            graph = feat.featurize(smiles_list[k])
            mapper = _MapperDMPNN(graph[0])
            compound_feature = Compound_feature(mapper)    
            comp_index += [k] 
            compound_feature_list += [compound_feature]              
        except:
            pass
        
    sequence_list = [dataset[1][i] for i in comp_index]   
    label_list = [dataset[2][i] for i in comp_index] 
    
    torsion_id = [[] for _ in range(len(comp_index))]
    data_list = [[sequence_list[i], compound_feature_list[i][0], compound_feature_list[i][1], 
                  compound_feature_list[i][2], compound_feature_list[i][3], 
                  compound_feature_list[i][4], compound_feature_list[i][5], torsion_id[i],
                  label_list[i]] for i in range(len(sequence_list)) ]
    return data_list    


    

def prot_embedding(tokenizer, model, sequence, seq_lengths, device):   
    
    sequence = [item.upper() for item in sequence]
    sequence = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequence]
    ids = tokenizer(sequence, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    
    seq_embeddings = embedding_repr.last_hidden_state
    seq_embeddings = seq_embeddings[:, :-1, :]
    p_mask = torch.zeros(len(seq_embeddings), len(seq_embeddings[0]))
    for k in range(len(p_mask)):
        x = seq_lengths[k]
        for i in range(x):
            p_mask[k][i] = 1
    
    return seq_embeddings, p_mask.to(device)




