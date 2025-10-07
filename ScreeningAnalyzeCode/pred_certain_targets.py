# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:55:49 2023

@author: Xinrui
"""

import pandas as pd
import os
import torch
from transformers import T5Tokenizer, T5EncoderModel

import params as params
from FeatureGenerationCode.prepare_features import generate_features_w_torsion
from utils import pdb_download
from ModelingCode.model import MBATT
from ModelingCode.training import tested_prob


input_file = "D:/projects/CPI_disease/IBD/certain_targets.xlsx"
dataset = pd.read_excel(input_file, sheet_name='input')
pdb_folder = "D:/projects/CPI_disease/PDB_certain_targets/"
os.makedirs(pdb_folder, exist_ok = True)

PDB_ID = dataset['PDB_ID'].drop_duplicates()
pdb_download(PDB_ID, pdb_folder)

params = params.get_params()

device = torch.device('cuda')


# generate data feature
tokenizer = T5Tokenizer.from_pretrained("D:/projects/prot_t5_xl_uniref50/")
prot_model = T5EncoderModel.from_pretrained("D:/projects/prot_t5_xl_uniref50/").to(device)

data_test, dataset = generate_features_w_torsion(dataset, pdb_folder)

n_torsion = 9
#max_mol_len = max([data_test[i][5] for i in range(len(data_test))])
max_mol_len = 320

model = MBATT(n_torsion, max_mol_len, params)
model = model.to(device)
checkpoint = torch.load('checkpoint_epoch2.pth')
state_dict = checkpoint['state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

task = params.y_type
batch_size = len(dataset)

predictions = tested_prob(model, task, data_test, batch_size, tokenizer, prot_model, device)
dataset['predicted_prob'] = predictions
dataset.to_csv("D:/projects/CPI_disease/IBD/IBD_output.csv")
