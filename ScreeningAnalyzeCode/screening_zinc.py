# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:10:32 2023

@author: Xinrui
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:55:49 2023

@author: Xinrui
"""

import os
import glob
import pandas as pd
import torch
import warnings
import deepchem as dc
from deepchem.models.torch_models.dmpnn import _MapperDMPNN
from transformers import T5Tokenizer, T5EncoderModel

import params as params
from FeatureGenerationCode.prepare_features import generate_features_w_torsion, prot_embedding, Compound_feature
from ModelingCode.model import MBATT
from ModelingCode.training import screen_prob, screen_similarity_score, tested_prob


params = params.get_params()
device = torch.device('cuda')


Screening_folder = "ScreeningResult"
if not os.path.isdir(Screening_folder):
  os.mkdir(Screening_folder)
  
# import data for screening with certain protein targets specified in advance
# concatenate multiple zinc files into one
input_mol_files = glob.glob("./Data/zinc20/" + '*.txt')
dfs_mol = []
for file in input_mol_files:
    df = pd.read_csv(file, header=None, delimiter='\t')
    dfs_mol.append(df)

df_mol = pd.concat(dfs_mol).reset_index(drop=True)
df_mol.columns = ['zinc_id','Screening_SMILES']
df_mol = df_mol.drop_duplicates(subset='Screening_SMILES')

input_templates_file = "./Data/breast_cancer_DT_templates.csv"
df_temp = pd.read_csv(input_templates_file)

# generate protein and compound features for certain protein-compound templates specified in advance
pdb_folder = "./PDB_files/PDB_templates/"
targets_feature_list, df_tar = generate_features_w_torsion(df_temp, pdb_folder) # if no 3D info from protein, then use generate_feature_wo_torsion()

tokenizer = T5Tokenizer.from_pretrained("./prot_t5_xl_uniref50/")
prot_model = T5EncoderModel.from_pretrained("./prot_t5_xl_uniref50/").to(device)
targets_seq = list(df_tar['Pocket_Sequence'])
targets_seq_length = [len(item) for item in targets_seq]
p, p_mask = prot_embedding(tokenizer, prot_model, targets_seq, targets_seq_length, device)
torsion = [ele[7] for ele in targets_feature_list]


# generate features for screening compounds
smiles_list = df_mol['Screening_SMILES']

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
checkpoint = torch.load('./ModelingResult/checkpoint_epoch2.pth')
state_dict = checkpoint['state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# instantiate model with two forms
n_torsion = 9
max_mol_len = 320
task = params.y_type
batch_size = params.batch_size
batch_size_screen = params.batch_size_screen
n_targets_pair = len(df_tar)



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

scores = screen_similarity_score(similarity_model,  
                                 compound_feature_list, 
                                 targets_feature_list, 
                                 batch_size_screen, 
                                 n_targets_pair, 
                                 p, p_mask, torsion, device)


prob_model = MBATT(n_torsion, max_mol_len, params)
prob_model.load_state_dict(state_dict)
prob_model.eval()  # Set the model to evaluation mode
prob_model = prob_model.to(device)

prob = screen_prob(prob_model, task, 
                   compound_feature_list, 
                   batch_size_screen, 
                   n_targets_pair, 
                   p, p_mask, torsion, device)


## concatenate input smiles for screening mol with ouput smilarity scores        
smiles_list_new = [item for item in selected_smiles_list for _ in range(n_targets_pair)]
smiles_list_new =  pd.DataFrame({'Screening_SMILES': smiles_list_new})
target_pair_new = pd.concat([df_tar] * len(selected_smiles_list), ignore_index=True)
target_pair_new = target_pair_new[['Ligand_SMILES','PDB_ID','het_name','Uniprot_ID']]
pred_output = pd.DataFrame({'Similarity_Scores': scores, 'Probabilities': prob})

pred_prob_all_pairs = pd.concat([target_pair_new, smiles_list_new, pred_output], axis=1)
pred_prob_all_pairs['Similarity_Scores'] = pd.to_numeric(pred_prob_all_pairs['Similarity_Scores'])
pred_prob_all_pairs['Probabilities'] = pd.to_numeric(pred_prob_all_pairs['Probabilities'])


## for each screening mol, select smallest similarity score among all target pairs
pred_scores = pred_prob_all_pairs.loc[pred_prob_all_pairs.groupby(pred_prob_all_pairs['Screening_SMILES'].astype(str))['Similarity_Scores'].idxmin()]
# merge prediction with zinc id
pred_scores = pd.merge(pred_scores, df_mol, how='left', on='Screening_SMILES').reset_index(drop=True)
pred_scores.to_csv("./ScreeningResult/screening_ZINC20_scores.csv", index=None)

# pred_scores = pred_scores[pred_scores['Similarity Scores']<=0.05]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    screen_feature_list, pred_scores = generate_features_w_torsion(pred_scores, pdb_folder, lig_type = "Screening_SMILES")



################ calculate predicted general binding probability for each mol with each target (n_mol * n_targets_pair output data points)
pred_model = MBATT(n_torsion, max_mol_len, params)
pred_model.load_state_dict(state_dict)
pred_model.eval()  # Set the model to evaluation mode
pred_model = pred_model.to(device)

pred_prob = tested_prob(pred_model, task, screen_feature_list, batch_size, tokenizer, prot_model, device)


## concatenate pred_scores with ouput pred_prob
pred_prob = pd.DataFrame({'predicted binding prob': pred_prob})

pred_scores = pd.concat([pred_scores, pred_prob], axis=1)

pred_scores['PDB_ID'] = pred_scores['PDB_ID'].astype(str) + '\t'
pred_scores.to_csv("./ScreeningResult/screening_ZINC20_prob.csv", index=None)








