# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:00:01 2023

@author: Xinrui
"""

import torch
import os
import csv
import pandas as pd

import params as params
from ModelingCode.training import train_eval
from ModelingCode.model import MBATT, BATT
from FeatureGenerationCode.prepare_features import generate_features_w_torsion, generate_features_wo_torsion
from utils import split_data, check_gpu

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    Test_folder = "ModelingResult"
    if not os.path.isdir(Test_folder):
      os.mkdir(Test_folder)
    
    params = params.get_params()
    print(params)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu = [0]
    if params.mode == 'gpu':
        device = check_gpu(gpu)
    else: 
        device = check_gpu([])   

    input_file = params.input_file
    split_ratio = params.split_ratio
    with_torsion = params.with_torsion
    y_type = params.task

    dataset = pd.read_csv(input_file)

    # choose train model with 3D protein info or only with sequence info
    if with_torsion == 1:
        # generate compound and protein features
        pdb_folder = "./PDB_files/"
        data_feature_list, dataset = generate_features_w_torsion(dataset, pdb_folder)
        # randomly split data
        data_train, data_val, data_test = split_data(data_feature_list, split_ratio)

        # prepare parameters for running model
        n_torsion = 9    
        Test_folder = params.Test_folder
        if not os.path.isdir(Test_folder):
            os.mkdir(Test_folder)
        
        max_mol_len = max([data_feature_list[i][5] for i in range(len(data_feature_list))])
        print("maximum number of molecule length is: " + str(max_mol_len))
                
        ## run training
        binding_model = MBATT(n_torsion, max_mol_len, params)
        binding_model = binding_model.to(device)
        binding_model = torch.nn.DataParallel(binding_model)
    
    elif  with_torsion == 0:
        # generate compound and protein features
        data_feature_list, dataset = generate_features_wo_torsion(dataset)
        # randomly split data
        data_train, data_val, data_test = split_data(data_feature_list, split_ratio="8:1:1")

        # prepare parameters for running model
        n_torsion = 9    
        Test_folder = params.Test_folder
        if not os.path.isdir(Test_folder):
            os.mkdir(Test_folder)
        
        max_mol_len = max([data_feature_list[i][5] for i in range(len(data_feature_list))])
        print("maximum number of molecule length is: " + str(max_mol_len))
                
        ## run training
        binding_model = BATT(max_mol_len, params)
        binding_model = binding_model.to(device)
        binding_model = torch.nn.DataParallel(binding_model)

    else:
        print("Please choose with_torsion as either 1 (include 3D protein info) or 0 (no 3D protein info)")
    
    # produce evaluation metrics
    res_all = train_eval(binding_model, y_type, data_train, data_val, data_test, params, device)
    
    if params.task == "affinity":
        metrics = ['train_rmse','train_pearson','train_spearman',
                   'validate_rmse','validate_pearson','validate_spearman',
                   'test_rmse','test_pearson','test_spearman', 'epoch']
    else:
        metrics = ['train_auc','train_acc','train_aupr',
                   'validate_auc','validate_acc','validate_aupr',
                   'test_auc','test_acc','test_aupr', 'epoch']
    with open(Test_folder + '/training_metrics.csv', 'w', newline='') as f:
        write = csv.writer(f)
         
        write.writerow(metrics)
        write.writerows(res_all)




