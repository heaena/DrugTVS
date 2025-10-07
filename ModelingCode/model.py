# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:36:16 2023

@author: Xinrui
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.torch_models import layers


class comp_encoding(nn.Module):
    def __init__(self, max_mol_len):
        super(comp_encoding, self).__init__()
        self.Comp_EncoderLayer = layers.DMPNNEncoderLayer(
          use_default_fdim = True,
          atom_fdim = 133,
          bond_fdim = 14,
          d_hidden = max_mol_len,
          depth = 3,
          bias = False,
          activation = 'relu',
          dropout_p = 0.0,
          aggregation = 'mean',
          aggregation_norm = 100) 
               
    def get_encoder(self, batch_data, device):       
        encodings = []
        for i in range(len(batch_data)):            
            atom_feature = batch_data[i][1].to(device)
            f_ini_atoms_bond = batch_data[i][2].to(device)
            atom_to_incoming_bond = batch_data[i][3].to(device)
            mapping = batch_data[i][4].to(device)
            mol_len = batch_data[i][5]
            global_f = batch_data[i][6].to(device)
            
            encoding = self.Comp_EncoderLayer(atom_feature, f_ini_atoms_bond, atom_to_incoming_bond, 
                                    mapping, global_f, mol_len)
            encoding = encoding[:mol_len]
            
            encodings += encoding
        return encodings
    
    
    def forward(self, batch_data, device):           
        # for comp, pad all tensors to have same length and create c_mask
        mol_lens = [batch_data[i][5] for i in range(len(batch_data))]
        comp_dim = max(mol_lens)
        
        encodings = self.get_encoder(batch_data, device)  
        comp = [F.pad(x, pad=(0, comp_dim - x.numel()), mode='constant', value=0) for x in encodings]
        comp = torch.stack(comp)    
        c_mask = torch.Tensor([ [1]*int(x)+[0]*(comp_dim -int(x)) for x in mol_lens])
        return comp.to(device), c_mask.to(device)




class BIATT(nn.Module):
    def __init__(self, params): 
        super(BIATT, self).__init__()
        self.multi_head = params.multi_head
        self.alpha = params.alpha
        latent_dim = params.latent_dim
        
        self.U = nn.ParameterList([nn.Parameter(torch.empty(size=(latent_dim, latent_dim))) for _ in range(self.multi_head)])
        for i in range(self.multi_head):
            nn.init.xavier_uniform_(self.U[i], gain=1.414)
        self.c_param = nn.Parameter(torch.empty(size=(1, latent_dim)))
        nn.init.xavier_uniform_(self.c_param, gain=1.414)
        self.p_param = nn.Parameter(torch.empty(size=(1024, latent_dim)))
        nn.init.xavier_uniform_(self.p_param, gain=1.414)
        
        self.transform_c2p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.multi_head)])
        self.transform_p2c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.multi_head)])
        
        self.bihidden_c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.multi_head)])
        self.bihidden_p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.multi_head)])
        self.biatt_c = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.multi_head)])
        self.biatt_p = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.multi_head)])
        
        self.comb_c = nn.Linear(latent_dim * self.multi_head, latent_dim)
        self.comb_p = nn.Linear(latent_dim * self.multi_head, latent_dim)
        

        
    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
        return a_softmax


           
    def forward(self, c, c_mask, p, p_mask):      
        c_vector = torch.matmul(c, self.c_param)
        p_vector = torch.matmul(p, self.p_param)

        b = c.shape[0]
        for idx in range(self.multi_head):
         
            A = torch.tanh(torch.matmul(torch.matmul(c_vector, self.U[idx]), p_vector.transpose(1, 2)))
            A = A * torch.matmul(c_mask.view(b, -1, 1), p_mask.view(b, 1, -1))
            
            c_trans = torch.matmul(A, torch.tanh(self.transform_p2c[idx](p_vector))) # information from residue to atom (9)
            p_trans = torch.matmul(A.transpose(1, 2), torch.tanh(self.transform_c2p[idx](c_vector))) # information from atom to residue (10)

            c_tmp = torch.cat([torch.tanh(self.bihidden_c[idx](c_vector)), c_trans], dim=2) # attention mechanism of atom-to-residue (11)
            p_tmp = torch.cat([torch.tanh(self.bihidden_p[idx](p_vector)), p_trans], dim=2) # attention mechanism of residue-to-atom (12)

            c_att = self.mask_softmax(self.biatt_c[idx](c_tmp).view(b, -1), c_mask.view(b, -1)) # normalized compound attention (11)
            p_att = self.mask_softmax(self.biatt_p[idx](p_tmp).view(b, -1), p_mask.view(b, -1)) # normalized protein attention (12)

            cf = torch.sum(c_vector * c_att.view(b, -1, 1), dim=1) # weighted sum of compound features (13)
            pf = torch.sum(p_vector * p_att.view(b, -1, 1), dim=1) # weighted sum of protein features (14)

            if idx == 0:
                cat_cf = cf
                cat_pf = pf
            else:
                cat_cf = torch.cat([cat_cf.view(b, -1), cf.view(b, -1)], dim=1) # concatenated compound features (15)
                cat_pf = torch.cat([cat_pf.view(b, -1), pf.view(b, -1)], dim=1) # concatenated protein features (16)
            
            
            
        cf_final = self.comb_c(cat_cf) # final compound features, including atoms feature and fingerprints feature (18)
        pf_final = self.comb_p(cat_pf) # final protein features (19)
        cf_pf = torch.matmul(cf_final.view(b, -1, 1), pf_final.view(b, 1, -1))

        return cf_pf
    

class MBATT(nn.Module):
    def __init__(self, n_torsion, max_mol_len, params):
        super(MBATT, self).__init__()
        y_type = params.y_type
        latent_dim = params.latent_dim
        self.alpha = params.alpha

        if y_type == 'affinity': 
            self.output = nn.Linear(latent_dim * latent_dim*2, 1) 
        elif y_type == 'interaction':
            self.output = nn.Linear(latent_dim * latent_dim*2, 2)
        else:
            print("Please choose a correct mode!!!")

        self.comp_model = comp_encoding(max_mol_len)
        self.bi_att0 = BIATT(params)
        self.bi_att1 = BIATT(params)
        self.torsion_embed = nn.Embedding(n_torsion+1, 1024)

    def tor(self, batch_data, device):
        torsion = [batch_data[x][7] for x in range(len(batch_data))]
        torsion = [torch.from_numpy(item) for item in torsion]
        prot_dim = max([len(item) for item in torsion])
        torsion = [F.pad(x, pad=(0, prot_dim - x.numel()), mode='constant', value=0) for x in torsion]
        torsion = torch.stack(torsion).to(device)
        torsion = self.torsion_embed(torsion)
        return torsion


    def forward(self, batch_data, p, p_mask, device): 
        c, c_mask = self.comp_model(batch_data, device)
        c = c.unsqueeze(2)
        b = len(batch_data)
        
        torsion = self.tor(batch_data, device)
        
        cf_pf0 = self.bi_att0(c, c_mask, p, p_mask)
        cf_pf1 = self.bi_att1(c, c_mask, torsion, p_mask)
        
        cf_pf0 = F.leaky_relu(cf_pf0.view(b, -1), self.alpha)
        cf_pf1 = F.leaky_relu(cf_pf1.view(b, -1), self.alpha)
                               
        pred = self.output(torch.cat([cf_pf0, cf_pf1],1))
        
        return pred
    


class BATT(nn.Module):
    def __init__(self, max_mol_len, params):
        super(MBATT, self).__init__()
        y_type = params.y_type
        latent_dim = params.latent_dim
        self.alpha = params.alpha

        if y_type == 'affinity': 
            self.output = nn.Linear(latent_dim * latent_dim*2, 1) 
        elif y_type == 'interaction':
            self.output = nn.Linear(latent_dim * latent_dim*2, 2)
        else:
            print("Please choose a correct mode!!!")

        self.comp_model = comp_encoding(max_mol_len)
        self.bi_att = BIATT(params)


    def forward(self, batch_data, p, p_mask, device): 
        c, c_mask = self.comp_model(batch_data, device)
        c = c.unsqueeze(2)
        b = len(batch_data)
               
        cf_pf0 = self.bi_att0(c, c_mask, p, p_mask)
        
        cf_pf0 = F.leaky_relu(cf_pf0.view(b, -1), self.alpha)
                               
        pred = self.output(torch.cat(cf_pf0, 1))
        
        return pred