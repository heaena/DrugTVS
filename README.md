# DrugTVS
construction of DrugTVS AI model for drug-protein interaction prediction and drug screening
Created on Thu Aug 14 10:53:11 2025

@author: Xinrui
"""

DrugTVS: Drug-Target Template-based Virtual Screening
This project is aimed at building an integration system for Drug-target interaction (DTI) modeling and small molecule virtual screening. Unlike traditional DTI model which solely considered enhancing model performance based on training and testing result using experimental data as benchmark dataset, our method comprehensively evaluated model's efficiency on training and virtual screening. 

The project includes folders, files that can be roughly summarized as: 
   1.1 RawDataCode: Codes for preprocessing data downloaded from different sources
   1.2 Data: some sample data downloaded
   1.3 DataProcessed 
   1.4 PDB_files: used to save PDB structure files with .pdb format, usually downloaded from https://www.rcsb.org/ 
   
   2.1 FeatureGenerationCode: Codes for generating compound and protein features, using the processed data

   3.1 ModelingCode: Codes for DTI modeling
   3.2 ModelingResult + datetime: modeling evalution results, and best tested model saved as checkpoint_epochX.pth

   4.1 ScreeningAnalyzeCode: Codes for drug screening and analyzing the screening results

   5.1 params: parameters used for training, modeling, screening and all related to this project

   6.1 utils: some utility codes used in this project


Next, some specifications will be noticed below:
   1. Training and testing DTI model:
     1.1 Input datafile is saved under "./DataProcessed/", the default is called "All_data_inductive.csv".

     1.2 It is encouraged to include 3D structural information of protein in the dataset, and it is also the default setting of this project. The input datafile should be in .csv format, including at least columns named as: <"PDB_ID","Ligand_SMILES","label","het_name","source">. The columns mentioned above is mandatory to run a model with protein 3D info, and the name cannot be changed. "het_name" is the ligand's 3-letter code, which can be found in PDB file and It could be left empty; "source" is the downloading source of PDB files, which could be "PDBbind" or others, or empty.
     the parameter "with_torsion" should be set to 1 (default).

     1.3 PDB files should be placed under folder "./PDB_files/". In our case, files downloaded from PDBbind are in "./PDB_files/PDBbind_v2020", the rest are in "./PDB_files/".

     1.4 If you want to run model without 3D structural information of protein, then the input dataset should at least include columns of Compound's SMILES Sequence, Protein's amino acid Sequence, and label. The columns should be ordered like this, but the column names do not matter.
     the parameter "with_torsion" should be set to 0 (default)

     1.5 protein sequence was featurized using a pre-trained model called: ProtTXL, which can be downloaded at:https://github.com/agemagician/ProtTrans. Please refer to this article (https://doi.org/10.1101/2020.07.12.199554) for detailed information. The downloaded ProtTXL folder, in our case, prot_t5_xl_uniref50, should be placed under this project.

    
    2. Virtual Screening:
     2.1 drug-target template dataset should be prepared based on the specific disease or targets you're interested. The datafile should be in .csv format and should at least include columns named as: <"PDB_ID","Ligand_SMILES","label","het_name","source">. 
     
     2.2 PDB files of drug-target template should be placed under folder "./PDB_files/PDB_templates/".

     2.3 the screening library should include: compound_id, compound_SMILES, and should be ordered like this. column names do not matter.

     2.4 the trained DTI model for drug screening editing was saved under folder "./ModelingResult/", and the screening result will be saved under folder "./ScreeningResult/".
