# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:22:37 2024

@author: Xinrui
"""



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_curve, auc



df = pd.read_csv("./result/all_out.csv", header=None)
df.columns = ['index','SMILES','Pocket_Seq','interaction','Pred_prob']

y_test = df['interaction']
prob_pos = df['Pred_prob']

# Create a calibration plot
prob_true, prob_pred = calibration_curve(y_test, prob_pos, n_bins=10, strategy='uniform')

plt.plot(prob_pred, prob_true, marker='o', linestyle='--', color='b', label='Calibration Plot')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.text(x=0.7, y=0.1, s='brier score = 0.15')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot')
plt.legend()

plt.savefig('./result/calibration_plot.png', dpi=300)
plt.show()


# Calculate the Brier score
brier_score = brier_score_loss(y_test, prob_pos)



# Compute ROC curve and ROC area for the classification model
fpr, tpr, thresholds = roc_curve(y_test, prob_pos)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.savefig('./result/ROC_curve.png', dpi=300)
plt.show()


# Calculate sensitivity and specificity for different thresholds
sensitivity = tpr
specificity = 1 - fpr

# Find the optimal threshold based on Youden's index
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = specificity[optimal_idx]

print(f'Optimal Cut-off Threshold: {optimal_threshold}')
print(f'Sensitivity at Optimal Cut-off: {optimal_sensitivity}')
print(f'Specificity at Optimal Cut-off: {optimal_specificity}')




# line chart for AUC on train, val, test
auc = pd.read_csv("./result/20231201.csv")
auc_melted = pd.melt(auc, value_vars=['train_auc', 'validate_auc', 'test_auc'], 
                     var_name='Dataset', value_name='AUC')
auc_melted['epoch'] = list(range(1, 21)) * 3


fig, ax = plt.subplots()
for key, grp in auc_melted.groupby('Dataset'):
    ax.plot(grp['epoch'], grp['AUC'], marker='o', label=key)

plt.xticks(range(1, 21))
plt.xlabel('num of epochs')
plt.ylabel('AUROC')
plt.legend(['test', 'train', 'validate'])
# plt.grid(True)

plt.savefig('./result/auc_line.png', dpi=300)
plt.show()