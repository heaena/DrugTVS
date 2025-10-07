# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:55:49 2023

@author: Xinrui
"""

import pandas as pd
import numpy as np
import torch
import deepchem as dc
from deepchem.models.torch_models.dmpnn import _MapperDMPNN
from transformers import T5Tokenizer, T5EncoderModel

import params as params
from FeatureGenerationCode.prepare_features import generate_features_w_torsion, prot_embedding, Compound_feature
from ModelingCode.model import MBATT
from ModelingCode.training import screen_prob, screen_similarity_score


params = params.get_params()
device = torch.device('cuda')
task = params.y_type

# import data for screening with certain protein targets specified in advance
input_mol_file = "D:/projects/CPI_disease/TTD_all.csv"
df_mol = pd.read_csv(input_mol_file)
input_targets_file = "D:/projects/CPI_disease/IBD/certain_targets.xlsx"
df_tar = pd.read_excel(input_targets_file, sheet_name = "selected_tar")

# generate protein features for certain protein targets specified in advance
pdb_folder = "D:/projects/CPI_disease/PDB_certain_targets/"
targets_feature_list, df_tar = generate_features_w_torsion(df_tar, pdb_folder)

tokenizer = T5Tokenizer.from_pretrained("D:/projects/prot_t5_xl_uniref50/")
prot_model = T5EncoderModel.from_pretrained("D:/projects/prot_t5_xl_uniref50/").to(device)
targets_seq = list(df_tar['Pocket_Sequence'])
targets_seq_length = [len(item) for item in targets_seq]
p, p_mask = prot_embedding(tokenizer, prot_model, targets_seq, targets_seq_length, device)
torsion = [ele[7] for ele in targets_feature_list]


# generate compound features
smiles_list = df_mol['Ligand SMILES'].drop_duplicates()

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

# filter out unsuccessful mol
selected_smiles_list = [smiles_list[i] for i in comp_index]



# load checkpoint for saved model
checkpoint = torch.load('checkpoint_epoch2.pth')
state_dict = checkpoint['state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# instantiate model with two forms
n_torsion = 9
max_mol_len = 320
batch_size_screen = params.batch_size_screen
n_targets_pair = len(df_tar)


################ calculate predicted general binding probability for each mol with each target (n_mol * n_targets_pair output data points)
pred_model = MBATT(n_torsion, max_mol_len, params)
pred_model.load_state_dict(state_dict)
pred_model.eval()  # Set the model to evaluation mode
pred_model = pred_model.to(device)

predictions = screen_prob(pred_model, task, compound_feature_list, batch_size_screen, n_targets_pair, p, p_mask, torsion, device)



################ calculate similarity score for each mol with each target pair (n_mol * n_targets_pair output data points)
similarity_model = MBATT(n_torsion, max_mol_len, params)
similarity_model.load_state_dict(state_dict)
similarity_model.output = torch.nn.Identity() # remove the <output> part for similarity model

# Define a new forward method that returns the concatenated tensor
def new_forward(self, batch_data, p, p_mask, device):
    c, c_mask = self.comp_model(batch_data, device)
    c = c.unsqueeze(2)

    torsion = self.tor(batch_data, device)

    cf_pf0 = self.bi_att0(c, c_mask, p, p_mask)
    cf_pf1 = self.bi_att1(c, c_mask, torsion, p_mask)

    pred = torch.cat([cf_pf0, cf_pf1], 1)

    return pred

# Replace the forward method of the new model with the modified one
similarity_model.forward = new_forward.__get__(similarity_model)
similarity_model = similarity_model.to(device)

cp_screen, cp_template = screen_similarity_score(similarity_model, task, 
                                                 compound_feature_list, 
                                                 targets_feature_list, 
                                                 batch_size_screen, 
                                                 n_targets_pair, 
                                                 p, p_mask, torsion, device)


# convert list of tensor to list of array
cp_screen_arr = [cp_screen[i].numpy() for i in range(len(cp_screen))]
cp_template_arr = [cp_template[i].numpy() for i in range(len(cp_template))]

# calculate Euclidean distance between the screening mol-target pair and specified drug-target pair
scores = []
for i in range(len(cp_screen)):
    score = np.linalg.norm(cp_screen_arr[i] - cp_template_arr[i])
    scores += [score]


## concatenate input smiles for screening mol with ouput predictions and smilarity scores        
smiles_list_new = [item for item in selected_smiles_list for _ in range(n_targets_pair)]
smiles_list_new =  pd.DataFrame({'Screening SMILES': smiles_list_new})
target_pair_new = pd.concat([df_tar] * len(selected_smiles_list), ignore_index=True)
target_pair_new = target_pair_new[['Ligand SMILES','PDB_ID','het_name','Uniprot_ID']]
predictions = pd.DataFrame({'predicted binding prob': predictions})
scores = pd.DataFrame({'Similarity Scores': scores})
predictions_all_pairs = pd.concat([target_pair_new, smiles_list_new, predictions, scores], axis=1)

predictions_all_pairs['PDB_ID'] = predictions_all_pairs['PDB_ID'].astype(str) + '\t'
predictions_all_pairs.to_csv("D:/projects/CPI_disease/IBD/screening_TTD_all_pairs.csv")




predictions_all_pairs['predicted binding prob'] = pd.to_numeric(predictions_all_pairs['predicted binding prob'])
predictions_all_pairs['Similarity Scores'] = pd.to_numeric(predictions_all_pairs['Similarity Scores'])

## for each screening mol, select largest predictions among all target pairs
predictions_largest_pred = predictions_all_pairs.loc[predictions_all_pairs.groupby('Screening SMILES')['predicted binding prob'].idxmax()]
## for each screening mol, select smallest similarity score among all target pairs
predictions_smallest_score = predictions_all_pairs.loc[predictions_all_pairs.groupby('Screening SMILES')['Similarity Scores'].idxmin()]

predictions_largest_pred.to_csv("D:/projects/CPI_disease/IBD/screening_TTD_largest_pred.csv")
predictions_smallest_score.to_csv("D:/projects/CPI_disease/IBD/screening_TTD_smallest_score.csv")

  





