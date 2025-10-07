import os
import urllib
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from math import sqrt
from scipy import stats
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
import pynvml


def get_info_from_dataset(input_file, delimiter=','):
    dataset = pd.read_csv(input_file, header=None, sep=delimiter)
    pdb_ids = [item.strip() for item in dataset[1]]
    smiles_list = [item.strip() for item in dataset[0]]
    label = [item for item in dataset[2]]
    
    return pdb_ids, smiles_list, label


def split_data(data_list: list, split_ratio: str="8:1:1"):
      
    r_train, r_val = int(split_ratio.split(":")[0]), int(split_ratio.split(":")[1])
    
    N = len(data_list)  # number of total samples in dataset
    n_train, n_val = int(N * r_train/10), int(N * r_val/10)
    
    idx = np.arange(N)
    np.random.shuffle(idx)
    data_train = [ data_list[i] for i in idx[ :n_train] ]
    data_val = [ data_list[i] for i in idx[n_train: (n_train+n_val)] ]
    data_test = [ data_list[i] for i in idx[(n_train+n_val): ] ]
    return data_train, data_val, data_test


def batch_pad(arr):
    N = max([a.shape[0] for a in arr]) # max length of original features of each type (0-5) from data_train
    if arr[0].ndim == 1:
        new_arr = np.zeros((len(arr), N))
        new_arr_mask = np.zeros((len(arr), N))
        for i, a in enumerate(arr):
            n = a.shape[0]
            new_arr[i, :n] = a + 1
            new_arr_mask[i, :n] = 1
        return new_arr, new_arr_mask

    elif arr[0].ndim == 2:  # for adjacencies_pad
        new_arr = np.zeros((len(arr), N, N))
        new_arr_mask = np.zeros((len(arr), N, N))
        for i, a in enumerate(arr):
            n = a.shape[0]
            new_arr[i, :n, :n] = a
            new_arr_mask[i, :n, :n] = 1
        return new_arr, new_arr_mask


def fps2number(arr):
    new_arr = np.zeros((arr.shape[0], 1024))
    for i, a in enumerate(arr):
        new_arr[i, :] = np.array(list(a), dtype=int)
    return new_arr


def batch_to_tensor(batch_data, device):
    atoms_pad, atoms_mask = batch_pad(batch_data[0])
    adjacencies_pad, _ = batch_pad(batch_data[1])
    fps = [[int(x) for x in item] for item in batch_data[2]]
    amino_pad, amino_mask = batch_pad(batch_data[3])
    dist_pad, _ = batch_pad(batch_data[4])
    torsion_pad, _ = batch_pad(batch_data[5])
    

    atoms_pad = Variable(torch.LongTensor(atoms_pad)).to(device)
    atoms_mask = Variable(torch.FloatTensor(atoms_mask)).to(device)
    adjacencies_pad = Variable(torch.LongTensor(adjacencies_pad)).to(device)
    fps = Variable(torch.FloatTensor(fps)).to(device)
    amino_pad = Variable(torch.LongTensor(amino_pad)).to(device)
    amino_mask = Variable(torch.FloatTensor(amino_mask)).to(device)
    dist_pad = Variable(torch.LongTensor(dist_pad)).to(device)
    torsion_pad = Variable(torch.LongTensor(torsion_pad)).to(device)

    label = torch.FloatTensor(batch_data[6]).to(device)

    return atoms_pad, atoms_mask, adjacencies_pad, fps, amino_pad, amino_mask, dist_pad, torsion_pad, label


def load_data(datadir, target_type):
    if target_type:
        dir_input = datadir + '/' + target_type + '/'
    else:
        dir_input = datadir + '/'
    compounds = np.load(dir_input + 'compounds.npy', allow_pickle=True)
    adjacencies = np.load(dir_input + 'adjacencies.npy', allow_pickle=True)
    fingerprint = np.load(dir_input + 'fingerprint.npy', allow_pickle=True)
    proteins = np.load(dir_input + 'proteins.npy', allow_pickle=True)
    interactions = np.load(dir_input + 'interactions.npy', allow_pickle=True)
    data_pack = [compounds, adjacencies, fingerprint, proteins, interactions]
    return data_pack



def regression_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    rmse = sqrt(((label - pred)**2).mean(axis=0))
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(rmse, 6), round(pearson, 6), round(spearman, 6)


def classification_scores(label, pred_score, pred_label):
    label = label.reshape(-1)
    pred_score = pred_score.reshape(-1)
    pred_label = pred_label.reshape(-1)
    auc = roc_auc_score(label, pred_score)
    acc = accuracy_score(label, pred_label)
    precision, recall, _ = precision_recall_curve(label, pred_label)
    aupr = metrics.auc(recall, precision)
    return round(auc, 6), round(acc, 6), round(aupr, 6)



def pdb_download(id_list, output_path):  # download pdb file from rcsb.org
    for ID in id_list: 
        file_path = output_path + ID + ".pdb"
        if not os.path.exists(file_path):
            url = 'http://files.rcsb.org/download/' + ID + '.pdb'
            outfile = output_path + os.sep + ID.upper() + '.pdb'
            try: 
                urllib.request.urlretrieve(url, outfile)
            except:
                pass



def check_gpu(gpus):
    if len(gpus) > 0 and torch.cuda.is_available():
        pynvml.nvmlInit()
        for i in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # if single gpu on machine, use 0; if multi-gpu on machine, use i
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memused = meminfo.used / 1024 / 1024
            print('GPU{} used: {}M'.format(i, memused))
        pynvml.nvmlShutdown()
        return torch.device('cuda')
    else:
        print('Using CPU!')
        return torch.device('cpu')