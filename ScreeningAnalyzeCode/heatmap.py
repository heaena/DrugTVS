# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:38:07 2024

@author: Xinrui
"""

import matplotlib.pyplot as plt
import torch
import os



# load checkpoint for saved model
checkpoint = torch.load('checkpoint_epoch2.pth')
state_dict = checkpoint['state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model_keys = list(state_dict.keys())

# save values accordings to keys as list of numpy arrays
model_values = []
for k in model_keys:
    v = state_dict[k].cpu().numpy()
    model_values.append(v)

# create folder for heatmap
heatmap_folder = './heatmap/'
os.makedirs(heatmap_folder, exist_ok=True)

for i in range(len(model_keys)):
    try:
        #plt.imshow(model_values[i] , cmap='autumn', interpolation='nearest') 
        plt.pcolor(model_values[i] , cmap='autumn', vmin=-0.4, vmax=0.4)
        # Add colorbar 
        plt.colorbar() 
          
        plt.title(model_keys[i]) 
        
        plt.savefig(heatmap_folder + model_keys[i] + '.png', dpi=300)
        
        plt.show() 
    except:
        pass
