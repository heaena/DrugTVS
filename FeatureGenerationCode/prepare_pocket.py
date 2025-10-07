# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:59:40 2023

@author: Xinrui
"""


import os
import Bio.PDB
from Bio.PDB.Polypeptide import one_to_three, three_to_one
import numpy as np
import pandas as pd
import math
import torch
import glob

from FeatureGenerationCode.ligand_identify import lig_id




class Pocket_feature():
    def __init__(self):
        super(Pocket_feature,self).__init__()
        
    
    def get_structure(self, pdb_file):
        pdb_id = pdb_file.split("\\")[-1][0:4]
        
        with open(pdb_file, 'r') as f:
            lines = []
            for line in f:
                if line.startswith("ATOM"):
                    lines.append(line)
        f.close()
            
        seq_num = [l[22:27].strip() for l in lines]
        chain_id = [l[21] for l in lines]
        
        df_seq = pd.DataFrame({'chain_id':chain_id, 'seq_num':seq_num})
        df_seq = df_seq.drop_duplicates().reset_index(drop=True)
        
        structure = Bio.PDB.PDBParser().get_structure(pdb_id, pdb_file)
        
        return df_seq, structure
        
    
    def prot_torsion(self, structure):
        for model in structure:
            df_torsion = []
            for chain in model:
                if chain.get_id() == ' ':
                    continue
                
                
                torsion_peptide, seq_peptide = [],""
                peptides = Bio.PDB.CaPPBuilder().build_peptides(chain)
                for peptide in peptides:
                    torsion = peptide.get_phi_psi_list()
                    seq = str(peptide.get_sequence())
                    
                    
                    torsion_peptide += torsion
                    seq_peptide += seq
                
                chain_id = chain.get_id()
                df_torsion_peptide = pd.DataFrame(torsion_peptide, columns=['phi','psi'])
                df_torsion_peptide['aa'] = list(seq_peptide)
                df_torsion_peptide['chain_id'] = chain_id
                
                df_torsion.append(df_torsion_peptide)
            if df_torsion != []:
                df_torsion = pd.concat(df_torsion).reset_index(drop=True)
                df_torsion['aa'] = [one_to_three(item) for item in df_torsion['aa']]
                    
        return df_torsion

    
    def pocket_torsion_w_pocketfile(self, pdb_file, pocket_file):
        
        with open(pocket_file, 'r') as f:
            lines = []
            for line in f:
                if line.startswith("ATOM"):
                    lines.append(line)
        f.close()
            
        seq_num = [l[22:27].strip() for l in lines]
        chain_id = [l[21] for l in lines]
        res = [l[17:20].strip() for l in lines]
        
        df_seq_pocket = pd.DataFrame({'chain_id':chain_id, 'seq_num':seq_num, 'res':res})
        df_seq_pocket = df_seq_pocket.drop_duplicates().reset_index(drop=True)
        
        df_seq, structure = self.get_structure(pdb_file)
        df_torsion = self.prot_torsion(structure)
        
        if len(df_seq) == len(df_torsion): # check if torsion calculation matched with sequence  
            df_torsion['seq_num'] = df_seq['seq_num']
            
            # extract rows of sequence number within pocket sequence number in each chain
            df_torsion_pocket = pd.merge(df_torsion, df_seq_pocket, how='right', on=['chain_id','seq_num'])
            df_torsion_pocket = df_torsion_pocket.fillna(0)
            
            res_ones = []
            for item in df_seq_pocket['res']:        
                try:
                    res_one = three_to_one(item)
                except:
                    res_one = 'X'  # for non-standard residue names
                res_ones += [res_one]    
            PocketSeq = ''.join(res_ones)
            # PocketSeq = ''.join([three_to_one(item) for item in df_torsion_pocket['res']])        
        else:
            df_torsion_pocket = "null"
            PocketSeq = ''
        
        return df_torsion_pocket, PocketSeq
    
    
    
    def pocket_torsion_w_hetname(self, pdb_file, lig_in_dataset, chain_in_dataset=None):
        
        hetname_chain = lig_id(pdb_file, lig_in_dataset, chain_in_dataset)
        
        if hetname_chain != []:
            with open(pdb_file, "r") as f:
                lig_coord = []
                receptor_coord = []
                for l in f.readlines():
                    if ((l.startswith("HETATM")) and (hetname_chain[0] == l[17:20]) and (hetname_chain[1] == l[21])):
                        lig_coord.append(l)
                    elif l.startswith("ATOM"):
                        receptor_coord.append(l)
                        
            # find lower limit and upper limit for x,y,z based on ligand coordinates
            lig_x = [float(item[31:38]) for item in lig_coord]
            lig_y = [float(item[39:46]) for item in lig_coord]
            lig_z = [float(item[46:54]) for item in lig_coord]
            x = (min(lig_x), max(lig_x))
            y = (min(lig_y), max(lig_y))
            z = (min(lig_z), max(lig_z))
            
            # extract residues around ligand
            receptor_x = [float(item[31:38]) for item in receptor_coord]
            receptor_y = [float(item[39:46]) for item in receptor_coord]
            receptor_z = [float(item[46:54]) for item in receptor_coord]
            
            pocket_coord = []
            for i in range(len(receptor_coord)):
                if receptor_x[i] >= x[0] - 5 and receptor_x[i] <= x[1] + 5:
                    if receptor_y[i] >= y[0] - 5 and receptor_y[i] <= y[1] + 5:
                        if receptor_z[i] >= z[0] - 5 and receptor_z[i] <= z[1] + 5:
                            pocket_coord.append(receptor_coord[i])
            
            seq_num = [l[22:27].strip() for l in pocket_coord]
            chain_id = [l[21] for l in pocket_coord]
            res = [l[17:20].strip() for l in pocket_coord]
            
            df_seq_pocket = pd.DataFrame({'chain_id':chain_id, 'seq_num':seq_num, 'res':res})
            df_seq_pocket = df_seq_pocket.drop_duplicates().reset_index(drop=True)
            
            res_ones = []
            for item in df_seq_pocket['res']:        
                try:
                    res_one = three_to_one(item)
                except:
                    res_one = 'X'  # for non-standard residue names
                res_ones += [res_one]    
            PocketSeq = ''.join(res_ones)
            #PocketSeq = ''.join([three_to_one(item) for item in df_seq_pocket['res']])        
        else:
            df_seq_pocket = pd.DataFrame(columns=['chain_id','seq_num','res'])
            PocketSeq = ''
        
        df_seq, structure = self.get_structure(pdb_file)
        df_torsion = self.prot_torsion(structure)
        
        if len(df_seq) == len(df_torsion):   
            df_torsion['seq_num'] = df_seq['seq_num']
            
            # extract rows of sequence number within pocket sequence number in each chain
            df_torsion_pocket = pd.merge(df_torsion, df_seq_pocket, how='right', on=['chain_id','seq_num'])
            df_torsion_pocket = df_torsion_pocket.fillna(0)
        else:
            df_torsion_pocket = "null"
            PocketSeq = ''
        
        return df_torsion_pocket, PocketSeq
    
    
    
        
    def torsion_region_bound(self):        
        EXT = {(-180,-150):[[150,180]]}
        BET = {(-135,-90):[[105,135]], (-90,-60):[[120,135]]}
        PP2 = {(-75,-45):[[135,165]]}
        BRI = {(-135,-120):[[30,45]], (-135,-120):[[60,75]]}
        HEL = {(-90,-60):[[-30,0]]}
        NHE = {(-120,-60):[[-60,-45]]}
        POS = {(45,60):[[15,60]], (60,75):[[0,45]]}
        GREY = {(-165,-150):[[-180,-165],[120,150]],
                (-150,-135):[[-180,-165],[90,180]],
                (-135,-120):[[-180,-165],[-30,30],[75,105],[135,180]],
                (-120,-105):[[-45,45],[90,105],[135,180]],
                (-105,-90):[[-180,-165],[-45,30],[75,105],[135,180]],
                (-90,-75):[[-180,-165],[-45,-30],[0,15],[60,120],[135,180]],
                (-75,-60):[[-45,-30],[105,120],[165,180]],
                (-60,-45):[[-60,-15],[120,135]],
                (75,90):[[-15,30]],
                (90,105):[[-15,15]]}
        return [EXT,BET,PP2,BRI,HEL,NHE,POS,GREY]
        
                   
    def torsion_region(self, region_dict, phi, psi):
        for k,v in region_dict.items():
            if (phi >= k[0]) & (phi <= k[1]):
                for psi_bound in v:
                    if (psi >= psi_bound[0]) & (psi <= psi_bound[1]): 
                        if_region = True
                        break
                    else:
                        if_region = False
                        
                if if_region == True:
                    break
                else:
                    if_region = False
            else:
                if_region = False
        return if_region
                


    def torsion_embeddin(self, torsion_pocket_list):
               
        torsion_id = []
        for k in range(len(torsion_pocket_list)):
            
            phi, psi = torsion_pocket_list[k][:,0], torsion_pocket_list[k][:,1]
            phi = [math.degrees(x) for x in phi]
            psi = [math.degrees(x) for x in psi]
            region_list = self.torsion_region_bound()
        
            region_idx = []
            for i in range(len(phi)):
                phi_i, psi_i = phi[i], psi[i]
                region_class = [self.torsion_region(region, phi_i, psi_i) for region in region_list]
                try:
                    idx = region_class.index(True) + 1
                except:
                    idx = 9
                region_idx += [idx]
                    
            region_idx = np.array(region_idx)
            torsion_id += [region_idx]
        
        
        return torsion_id
  
    




               
