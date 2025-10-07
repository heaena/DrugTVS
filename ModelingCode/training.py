# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:29:43 2023

@author: Xinrui
"""



import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import sys
from transformers import T5Tokenizer, T5EncoderModel


from utils import regression_scores, classification_scores
from FeatureGenerationCode.prepare_features import prot_embedding



def train_eval(model, task, data_train, data_dev, data_test, params, device):
    if task == 'affinity':
        criterion = F.mse_loss
        best_res = 2 ** 10
    elif task == 'interaction':
        criterion = F.cross_entropy
        best_res = 0
    else:
        print("Please choose a correct mode!!!")
        return 
    
    tokenizer = T5Tokenizer.from_pretrained("./prot_t5_xl_uniref50/")
    prot_model = T5EncoderModel.from_pretrained("./prot_t5_xl_uniref50/").to(device)
    # prot_model.full() if device=='cpu' else prot_model.half()
    
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    idx = np.arange(len(data_train))
    batch_size = params.batch_size
    res_all = []
    for epoch in range(params.num_epochs):
        print('epoch: {}'.format(epoch))
        np.random.shuffle(idx)
        model.train()
        pred_labels = []
        predictions = []
        labels = []
        for i in range(math.ceil(len(data_train) / batch_size)):
            batch_data = [data_train[x] for x in idx[i * batch_size: (i + 1) * batch_size] ]
            
            batch_seq = [item[0] for item in batch_data]
            batch_seq_length = [len(item) for item in batch_seq]
            p, p_mask = prot_embedding(tokenizer, prot_model, batch_seq, batch_seq_length, device)
            
            label = [batch_data[x][8] for x in range(len(batch_data))]
            label = torch.tensor(label).to(device)

            pred = model(batch_data, p, p_mask, device)
            if task == 'affinity':
                loss = criterion(pred.squeeze(1), label)
                predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
                labels += label.cpu().numpy().reshape(-1).tolist()
            elif task == 'interaction':
                loss = criterion(pred.float(), label.view(label.shape[0]).long())
                ys = F.softmax(pred, 1).to('cpu').data.numpy()
                pred_labels += list(map(lambda x: np.argmax(x), ys))
                predictions += list(map(lambda x: x[1], ys))
                labels += label.cpu().numpy().reshape(-1).tolist()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if params.verbose:
                sys.stdout.write('\repoch:{}, batch:{}/{}, loss:{}'.format(epoch, i, math.ceil(len(data_train)/batch_size)-1, float(loss.data)))
                sys.stdout.flush()

        if task == 'affinity':
            print(' ')
            predictions = np.array(predictions)
            labels = np.array(labels)
            rmse_train, pearson_train, spearman_train = regression_scores(labels, predictions)
            print('Train rmse:{}, pearson:{}, spearman:{}'.format(rmse_train, pearson_train, spearman_train))

            rmse_dev, pearson_dev, spearman_dev = test(model, task, data_dev, batch_size, tokenizer, prot_model, device)
            print('Dev rmse:{}, pearson:{}, spearman:{}'.format(rmse_dev, pearson_dev, spearman_dev))

            rmse_test, pearson_test, spearman_test = test(model, task, data_test, batch_size, tokenizer, prot_model, device)
            print( 'Test rmse:{}, pearson:{}, spearman:{}'.format(rmse_test, pearson_test, spearman_test))

            res_train = [rmse_test, pearson_test, spearman_test]
            res_val = [rmse_dev, pearson_dev, spearman_dev]
            res_test = [rmse_test, pearson_test, spearman_test]
            # metrics = ['rmse','pearson','spearman']
            
            res = res_train + res_val + res_test + [epoch]
            res_all += [res]
        
        else:
            print(' ')
            pred_labels = np.array(pred_labels)
            predictions = np.array(predictions)
            labels = np.array(labels)
            auc_train, acc_train, aupr_train = classification_scores(labels, predictions, pred_labels)
            print('Train auc:{}, acc:{}, aupr:{}'.format(auc_train, acc_train, aupr_train))

            auc_dev, acc_dev, aupr_dev = test(model, task, data_dev, batch_size, tokenizer, prot_model, device)
            print('Dev auc:{}, acc:{}, aupr:{}'.format(auc_dev, acc_dev, aupr_dev))

            auc_test, acc_test, aupr_test = test(model, task, data_test, batch_size, tokenizer, prot_model, device)
            print('Test auc:{}, acc:{}, aupr:{}'.format(auc_test, acc_test, aupr_test))

            res_train = [auc_train, acc_train, aupr_train]
            res_val = [auc_dev, acc_dev, aupr_dev]
            res_test = [auc_test, acc_test, aupr_test]
            # metrics = ['auc','acc','aupr']
            
            res = res_train + res_val + res_test + [epoch]
            res_all += [res]
            
        # checkpoint
        if task == 'affinity':            
            if res_test[0] <= best_res:
                best_res = res_test[0]
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'lr_schedule': scheduler.state_dict()}
                torch.save(checkpoint, params.Test_folder + os.sep + 'checkpoint_epoch' + '%d.pth' % (epoch))
            
        else:
            if res_test[0] >= best_res:
                best_res = res_test[0]
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'lr_schedule': scheduler.state_dict()}
                torch.save(checkpoint, params.Test_folder + os.sep + 'checkpoint_epoch' + '%d.pth' % (epoch))
        scheduler.step()
    return res_all


def test(model, task, data_test, batch_size, tokenizer, prot_model, device):
    idx = np.arange(len(data_test))
    np.random.shuffle(idx)
    model.eval()
    predictions = []
    pred_labels = []
    labels = []
    for i in range(math.ceil(len(data_test) / batch_size)):
        batch_data = [data_test[x] for x in idx[i * batch_size: (i + 1) * batch_size] ]
        
        batch_seq = [item[0] for item in batch_data]
        batch_seq_length = [len(item) for item in batch_seq]
        p, p_mask = prot_embedding(tokenizer, prot_model, batch_seq, batch_seq_length, device)
        
        label = [batch_data[x][8] for x in range(len(batch_data))]
        label = torch.tensor(label).to(device)
   
        with torch.no_grad():
            pred = model(batch_data, p, p_mask, device)
        if task == 'affinity':
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += label.cpu().numpy().reshape(-1).tolist()
        else:
            ys = F.softmax(pred, 1).to('cpu').data.numpy()
            pred_labels += list(map(lambda x: np.argmax(x), ys))
            predictions += list(map(lambda x: x[1], ys))
            labels += label.cpu().numpy().reshape(-1).tolist()
    pred_labels = np.array(pred_labels)
    predictions = np.array(predictions)
    labels = np.array(labels)
    if task == 'affinity':
        rmse_value, pearson_value, spearman_value = regression_scores(labels, predictions)
        return rmse_value, pearson_value, spearman_value
    else:
        auc_value, acc_value, aupr_value = classification_scores(labels, predictions, pred_labels)
        return auc_value, acc_value, aupr_value



def tested_prob(model, task, data_test, batch_size, tokenizer, prot_model, device):
    idx = np.arange(len(data_test))
    model.eval()
    predictions = []
    for i in range(math.ceil(len(data_test) / batch_size)):
        batch_data = [data_test[x] for x in idx[i * batch_size: (i + 1) * batch_size] ]
        
        batch_seq = [item[0] for item in batch_data]
        batch_seq_length = [len(item) for item in batch_seq]
        p, p_mask = prot_embedding(tokenizer, prot_model, batch_seq, batch_seq_length, device)
           
        with torch.no_grad():
            pred = model(batch_data, p, p_mask, device)
        if task == 'affinity':
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
        else:
            ys = F.softmax(pred, 1).to('cpu').data.numpy()            
            predictions += list(map(lambda x: x[1], ys))
    
    return predictions



def screen_prob(model, task, screen_data, batch_size_screen, n_targets_pair, p, p_mask, torsion, device):
    idx = np.arange(len(screen_data))
    model.eval()
    predictions = []
    for i in range(math.ceil(len(screen_data) / batch_size_screen)):
        
        # create new batch_data based on batch_size specified
        comp_batch = [screen_data[x] for x in idx[i * batch_size_screen: (i + 1) * batch_size_screen] ]
        comp_batch_new = [item for item in comp_batch for _ in range(n_targets_pair)]
        empty_list = [0] * len(comp_batch_new)
               
        p_new = torch.cat([p] * len(comp_batch), dim=0)
        p_mask_new = torch.cat([p_mask] * len(comp_batch), dim=0)
        torsion_new = torsion * len(comp_batch)
        
        batch_data_new = [[empty_list[i], comp_batch_new[i][0], comp_batch_new[i][1], 
                           comp_batch_new[i][2], comp_batch_new[i][3], 
                           comp_batch_new[i][4], comp_batch_new[i][5], torsion_new[i],
                           empty_list[i]] for i in range(len(comp_batch_new)) ]
        
        with torch.no_grad():
            pred = model(batch_data_new, p_new, p_mask_new, device)
        if task == 'affinity':
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()           
        else:
            ys = F.softmax(pred, 1).to('cpu').data.numpy()
            predictions += list(map(lambda x: x[1], ys))
        
    return predictions
        
        

def screen_similarity_score(model, screen_data, targets_data, batch_size_screen, n_targets_pair, p, p_mask, torsion, device):
    idx = np.arange(len(screen_data))
    model.eval()
    scores = []
    for i in range(math.ceil(len(screen_data) / batch_size_screen)):
        
        # create new batch_data based on batch_size specified
        comp_batch = [screen_data[x] for x in idx[i * batch_size_screen: (i + 1) * batch_size_screen] ]
        comp_batch_new = [item for item in comp_batch for _ in range(n_targets_pair)]
        empty_list = [0] * len(comp_batch_new)
               
        p_new = torch.cat([p] * len(comp_batch), dim=0)
        p_mask_new = torch.cat([p_mask] * len(comp_batch), dim=0)
        torsion_new = torsion * len(comp_batch)
        
        batch_data_new = [[empty_list[i], comp_batch_new[i][0], comp_batch_new[i][1], 
                           comp_batch_new[i][2], comp_batch_new[i][3], 
                           comp_batch_new[i][4], comp_batch_new[i][5], torsion_new[i],
                           empty_list[i]] for i in range(len(comp_batch_new)) ]
        
        targets_data_new = targets_data * len(comp_batch)
        with torch.no_grad():
            cp_screen = model(batch_data_new, p_new, p_mask_new, device)
            cp_template = model(targets_data_new, p_new, p_mask_new, device)
            
            # calculate Euclidean distance between the screening mol-target pair and specified drug-target pair
            cp_screen_arr = cp_screen.cpu().detach().numpy()
            cp_template_arr = cp_template.cpu().detach().numpy()
            score = [np.linalg.norm(cp_screen_arr[k] - cp_template_arr[k]) for k in range(len(cp_screen))]
        scores += score
        
    return scores