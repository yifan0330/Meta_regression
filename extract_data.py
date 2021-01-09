import numpy as np
import pandas as pd
import random

# load study information 
study_path = "/Users/yifan/Documents/GitHub/Meta_regression/BrainMap/func424/func424_exp.txt"
BrainMap_info = pd.read_csv(study_path, sep="	",engine ='python', header=None) #12839 rows; 18 columns
BrainMap_info.columns = ["study_id", "contrast_id", "ConName", "Pub", "Year", "Author", "PMID", "Type", "Domain", "Exp1", "Exp2", "Exp3", "Exp4", "Pop", "Exp5", "Exp6", "Y/N", "Sample_size"]

# create boolean variable based on two conditions
type_normal = BrainMap_info["Type"] == "Normals"
domain_visual = BrainMap_info["Domain"].str.contains('Visual')
BrainMap_study = BrainMap_info[type_normal & domain_visual] #6574 rows; 18 columns

study_id_list = BrainMap_study["study_id"].tolist() #length: 6574
study_id_unique = sorted(list(set(study_id_list))) # unique and ascening list
n_study = len(study_id_unique) #1615

## As each paper in BrainMap database has multiple contrasts, potentially violating the independence assumption,
## We draw subsamples such that exactly one contrast from each publication is used.
sampled_contrast_list = list()
for study_id in study_id_unique:
    # create sub-dataframe based on study_id
    study_df = BrainMap_study[BrainMap_study['study_id'] == study_id]
    contrast_id = study_df["contrast_id"].tolist()
    sampled_contrast = random.choice(contrast_id) # randomly sample one contrast
    sampled_contrast_list.append(sampled_contrast)

# load foci information
foci_path = "/Users/yifan/Documents/GitHub/Meta_regression/BrainMap/func424/func424_mni.txt"
BrainMap_foci = pd.read_csv(foci_path, sep="	",engine ='python', header=None) #102682 rows; 8 columns
BrainMap_foci.columns = ["study_id", "contrast_id", "x", "y", "z", "col6", "col7", "col8"]

# save foci from the randomly chosen contrast to csv
foci_df= pd.DataFrame(columns=["study_id", "contrast_id", "x", "y", "z"])
for i in range(n_study):
    study_index = study_id_unique[i]
    contrast_index = sampled_contrast_list[i]
    sample_foci = BrainMap_foci.query("study_id == " + str(study_index) + " and contrast_id == " + str(contrast_index))
    sample_foci_subdf = sample_foci[["study_id", "contrast_id", "x", "y", "z"]] # ignore the last three columns
    foci_df = pd.concat([foci_df, sample_foci_subdf])
print(foci_df)
print(foci_df.shape) #15240 foci
foci_df.to_csv("BrainMap_foci.csv",index=False)





