# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 12:09:46 2023

@author: Xinrui
"""


# from datetime import datetime
import argparse





args = argparse.ArgumentParser(description='Argparse for compound-protein interactions prediction')

args.add_argument('-input_file', type=str, default='./DataProcessed/All_data_inductive.csv', help="input data folder path. Data must at least contains columns of smiles, pdbID, labels in order")
args.add_argument('-y_type', type=str, default='interaction', help="type of prediction target variable. Can be either 'affinity' or 'interaction'.")
args.add_argument('-pdb_source', type=str, default='rcsb', help="source of pdb format file. Can be either 'AlphaFold' or 'rcsb'.")
args.add_argument('-split_ratio', type=str, default='8:1:1', help="ratio of spliting data into train, validate, test. Can be written as the format of 'x:x:x' as string.")
args.add_argument('-mode', type=str, default='gpu', help="choose either 'gpu' or 'cpu'.")
args.add_argument('-verbose', type=int, default=1, help='0: do not output log in stdout, 1: output log')
args.add_argument('-with_torsion', type=int, default=1, help='whether to include protein 3D info in training and screening, set to 0 if no 3D info provided')

args.add_argument('-batch_size', type=int, default=16, help="batch size of training data")
args.add_argument('-batch_size_screen', type=int, default=5, help="batch size for screening with certain targets specified")
args.add_argument('-num_epochs', type=int, default=20, help="number of epochs")
args.add_argument('-lr', type=float, default=0.0005, help="initial learning rate")
args.add_argument('-step_size', type=int, default=10, help='step size of lr_scheduler')
args.add_argument('-gamma', type=float, default=0.5, help='lr weight decay rate')


args.add_argument('-multi_head', type=int, default=4, help="number of multi-head bi-directional attention mechanism")
args.add_argument('-activation', type=str, default='leakyrelu', help="type of activation function for Pointwise feed-forward layers. Can be a choice of 'relu','leakyrelu','prelu','tanh','selu','elu','linear'.")
args.add_argument('-layer_ffn', type=int, default=3, help="number of layers for Pointwise feed-forward layers")
args.add_argument('-dropout', type=float, default=0.1, help="dropout rate for Pointwise feed-forward layers")
args.add_argument('-latent_dim', type=float, default=40, help="latent dimension for bi-directional attention network")


args.add_argument('-window', type=int, default=5, help='window size of cnn model')
args.add_argument('-layer_cnn', type=int, default=3, help='number of layer in cnn model')

args.add_argument('-alpha', type=float, default=0.1, help='LeakyReLU alpha')


def get_params():
    params, _ = args.parse_known_args()
    return params



