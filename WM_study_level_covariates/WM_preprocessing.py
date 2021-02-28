import nibabel as nib
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz

# load the brain mask
mask_MNI152 = nib.load("MNI152_T1_2mm_brain.nii")
mask = np.array(mask_MNI152.dataobj) # unique values: 0/1
mask[mask > 0] = 1
n_brain_voxel = np.sum(mask) # 292019/228453 within-brain voxels
dim_mask = np.shape(mask) # [91,109,91]

# load stimulus type 
info = open('info.txt', 'r')
info_lines = info.readlines()
verbal_col_num = 6
verbal_list = list()
for x in info_lines:
    verbal_list.append(int(x.split()[verbal_col_num]))

info_df = pd.DataFrame({'contrast': list(range(1,158)), 'verbal': verbal_list})
# load foci data
foci_data = np.loadtxt('WM_original_foci.txt') # shape: (2240,4)
# convert to pandas dataframe
foci_data = pd.DataFrame({'contrast':foci_data[:,0], 'x':foci_data[:,1],
                    'y':foci_data[:,2], 'z':foci_data[:,3]})
foci_data = foci_data.astype("int64")
# merge two dataframes
foci = pd.merge(foci_data, info_df, on='contrast')

# transform the foci from MNI152 coordinates to voxel space
origin = (90,-126,-72)
foci["x"] = (origin[0] - foci["x"])/2 # round to the nearest integer
foci["y"] = (foci["y"] - origin[1])/2
foci["z"] = (foci["z"] - origin[2])/2
foci = foci.round(0) # round to the nearest integer
foci = foci.astype("int64")


# remove the foci if its coordinates are beyond (91,109,91)
drop_condition = foci[(foci["x"]<0)|(foci["x"]>90)|(foci["y"]<0)|(foci["y"]>108)|(foci["z"]<0)|(foci["z"]>90)].index
foci = foci.drop(drop_condition) # shape: (2239, 5)
foci = foci.reset_index(drop=True)
# remove the foci if it falls outside the brain mask
n_foci = foci.shape[0] # 2239

foci_outside = list()
for i in range(n_foci):
    foci_coord = foci.iloc[i,1:4]
    if mask[foci_coord[0], foci_coord[1], foci_coord[2]] == 0:
        foci_outside.append(i)
foci = foci.drop(labels=foci_outside, axis=0) # shape: (2223, 5)/(2018,5)

# remove the blank space around the brain mask
xx, yy, zz = np.arange(dim_mask[0]), np.arange(dim_mask[1]), np.arange(dim_mask[2])
x_remove, y_remove, z_remove = np.array([]), np.array([]), np.array([])
for i in range(dim_mask[0]):
    if np.sum(mask[i,:, :]) == 0:
        x_remove = np.append(x_remove, i)
for j in range(dim_mask[1]):
    if np.sum(mask[:, j, :]) == 0:
        y_remove = np.append(y_remove, j)
for k in range(dim_mask[2]):
    if np.sum(mask[:, :, k]) == 0:
        z_remove = np.append(z_remove, k)
xx = np.setdiff1d(xx, x_remove) #[7,82] 76 / [11,79] 68
yy = np.setdiff1d(yy, y_remove) #[8,101] 94 / [11,98] 88
zz = np.setdiff1d(zz, z_remove) #[3,78] 76 / [0, 75] 76

x_min, x_max = np.min(xx), np.max(xx) # 7, 82
y_min, y_max = np.min(yy), np.max(yy) # 8, 101
z_min, z_max = np.min(zz), np.max(zz) # 3, 78

# remove the foci if its coordinates are beyond (7~82, 8~101, 3~78)
drop_condition2 = foci[(foci["x"]<x_min)|(foci["x"]>x_max)|(foci["y"]<y_min)|(foci["y"]>y_max)|(foci["z"]<z_min)|(foci["z"]>z_max)].index
foci = foci.drop(drop_condition2) #shape: (2223, 5) / (2018, 5)

# split the foci dataset into verbal/non-verbal
foci_verbal = foci.loc[foci['verbal'] == 1] # shape: (1286, 5)
foci_non_verbal = foci.loc[foci['verbal'] == 0] # shape: (937, 5)
# reset the index
foci = foci.reset_index(drop=True)
foci_verbal = foci_verbal.reset_index(drop=True)
foci_non_verbal = foci_non_verbal.reset_index(drop=True)


# count the total number of foci in study i regardless of foci location in verbal/non-verbal group
contrast_verbal = foci_verbal['contrast'].tolist() # list contrast index of verbal stimuli 
Y_i_verbal = list()
for i in np.unique(contrast_verbal):
    Y_i_verbal.append(contrast_verbal.count(i))
Y_i_verbal = np.array(Y_i_verbal) # shape: (102,); sum: 1286

contrast_nonverbal = foci_non_verbal['contrast'].tolist()
Y_i_nonverbal = list()
for i in np.unique(contrast_nonverbal):
    Y_i_nonverbal.append(contrast_nonverbal.count(i))
Y_i_nonverbal = np.array(Y_i_nonverbal) # shape: (55,); sum: 937

np.savetxt('preprocessed_data/Y_i_verbal.txt', Y_i_verbal, fmt='%i')
np.savetxt('preprocessed_data/Y_i_nonverbal.txt', Y_i_nonverbal, fmt='%i')

# create B-spline basis for x/y/z coordinate
x_deg = 3
x_knots = np.arange(min(xx), max(xx), step=14) # 9 23 37 51 65 79
x_design_matrix = patsy.dmatrix("bs(x, knots=x_knots, degree=3,include_intercept=False)", data={"x":xx},return_type='matrix')
x_design_array = np.array(x_design_matrix) 
x_spline = x_design_array[:,1:] # remove the first column (every element is 1); shape: (76, 8) / (69, 8)
# Note the B-spline basis is a partition of unity
x_rowsum = np.sum(x_spline, axis=1)
# Note the B-spline basis is sparse
x_colmean = np.mean(x_spline, axis=0)

y_deg = 3
y_knots = np.arange(min(yy), max(yy), step=14) # 10 24 38 52 66 80 94
y_design_matrix = patsy.dmatrix("bs(y, knots=y_knots, degree=3,include_intercept=False)", data={"y":yy},return_type='matrix')
y_design_array = np.array(y_design_matrix) 
y_spline = y_design_array[:,1:] # shape: (94, 10) / (88, 10)
y_rowsum = np.sum(y_spline, axis=1)
y_colmean = np.mean(y_spline, axis=0)

z_deg = 3
z_knots = np.arange(min(zz), max(zz), step=14) # 1 15 29 43 57 71
z_design_matrix = patsy.dmatrix("bs(z, knots=z_knots, degree=3,include_intercept=False)", data={"z":zz},return_type='matrix')
z_design_array = np.array(z_design_matrix) 
z_spline = z_design_array[:,1:] # shape: (76, 8) / (76, 8)
z_rowsum = np.sum(z_spline, axis=1)
z_colmean = np.mean(z_spline, axis=0)


image_dim = np.array([x_spline.shape[0], y_spline.shape[0], z_spline.shape[0]]) #[76 94 76] / [72 90 76]
x_df, y_df, z_df = x_spline.shape[1], y_spline.shape[1], z_spline.shape[1] 
image_df = np.array([x_df, y_df, z_df]) # 8, 10, 8 / 9, 10, 9

# convert spline bases in 3 dimesion to data matrix by tensor product
X = np.empty(shape=[np.prod(image_dim), np.prod(image_df)]) #shape: (542944, 640) / (461472, 640)
xyz_spline = np.zeros((x_spline.shape[0], y_spline.shape[0], z_spline.shape[0])) #shape: (76, 94, 76) / (69, 88, 76)
colume_number = 0
for bx in range(x_df):
    for by in range(y_df):
        xy_spline = np.outer(x_spline[:,bx], y_spline[:,by]) # outer product: shape (76, 94)
        for bz in range(z_df):
            for z in range(z_spline.shape[0]):
                xyz_spline[:,:,z] = xy_spline * z_spline[z,bz]
            X[:,colume_number] = xyz_spline.reshape((np.prod(image_dim)))
            colume_number += 1
# X = np.kron(np.kron(x_spline, y_spline), z_spline)

# Create data matrix of foci (coefficients of B-spline basis)
foci['index'] = (foci['x'] - x_min) + image_dim[0]*(foci['y'] - y_min) + image_dim[0]*image_dim[1]*(foci['z']-z_min) # shape: (2223, 6)
foci_verbal['index'] = (foci_verbal['x'] - x_min) + image_dim[0]*(foci_verbal['y'] - y_min) + image_dim[0]*image_dim[1]*(foci_verbal['z']-z_min) # shape: (1286, 6)
foci_non_verbal['index'] = (foci_non_verbal['x'] - x_min) + image_dim[0]*(foci_non_verbal['y'] - y_min) + image_dim[0]*image_dim[1]*(foci_non_verbal['z']-z_min) # shape: (937, 6)

# initialize the response vector as 0
y_all= np.zeros((dim_mask[0],dim_mask[1],dim_mask[2])) # shape: (91,109,91)
for i in range(foci.shape[0]):
    row = foci.iloc[i].tolist()
    x,y,z = row[1:4]
    y_all[x,y,z] += 1
y_all = y_all.astype(int)
y_all = y_all[mask == 1] # shape: (292019,)
# sum of y_verbal: 2223; number of nonzero elements: 2125; max of y_verbal: 3
save_npz("preprocessed_data/y.npz", csr_matrix(y))

y_verbal = np.zeros((dim_mask[0],dim_mask[1],dim_mask[2])) # shape: (91,109,91)
for i in range(foci.shape[0]):
    row = foci.iloc[i].tolist()
    verbal = row[4]
    x,y,z = row[1:4]
    if verbal == 1:
        y_verbal[x,y,z] += 1
y_verbal = y_verbal.astype(int)
y_verbal = y_verbal[mask == 1] # shape: (292019,)
# sum of y_verbal: 1286; number of nonzero elements: 1244; max of y_verbal: 3
save_npz("preprocessed_data/y_verbal.npz", csr_matrix(y_verbal))

y_nonverbal = np.zeros((dim_mask[0],dim_mask[1],dim_mask[2])) # shape: (91,109,91)
for i in range(foci.shape[0]):
    row = foci.iloc[i].tolist()
    verbal = row[4]
    x,y,z = row[1:4]
    if verbal == 0:
        y_nonverbal[x,y,z] += 1
y_nonverbal = y_nonverbal.astype(int)
y_nonverbal = y_nonverbal[mask == 1] # shape: (292019,)
# sum of y_non_verbal: 923; number of nonzero elements: 923; max of y_verbal: 2
save_npz("preprocessed_data/y_nonverbal.npz", csr_matrix(y_nonverbal))

## remove the voxels outside brain mask
np.set_printoptions(suppress=True)
outside_brain = np.array([])
for i in range(image_dim[0]):
    for j in range(image_dim[1]):
        for k in range(image_dim[2]):
            coord = np.array([xx[i], yy[j], zz[k]])
            if mask[coord[0], coord[1], coord[2]] == 0:
                index = (coord[0] - x_min) + image_dim[0]*(coord[1] - y_min) + image_dim[0]*image_dim[1]*(coord[2]-z_min)
                outside_brain = np.append(outside_brain, index)
outside_brain = outside_brain.astype(int)
np.savetxt("preprocessed_data/outside_brain.txt", outside_brain, fmt="%i")
X = np.delete(X, obj=outside_brain, axis=0) # shape: (292019, 640) / (194369, 640)

## remove tensor product basis that have no support in the brain
no_suppport_basis = np.array([])
for bx in range(x_df):
    for by in range(y_df):
        for bz in range(z_df):
            basis_index = bz + z_df*by + z_df*y_df*bx
            basis_coef = X[:, basis_index]
            if np.sum(basis_coef) < 192: # 5% of the voxel with largest coefficient summation/maximum 
                no_suppport_basis = np.append(no_suppport_basis, basis_index)
no_suppport_basis = no_suppport_basis.astype(int) # 385
# 197/275/329 tensor product of spline basis have no support in brain mask (computationally)
X = np.delete(X, obj=no_suppport_basis, axis=1) # shape: (292019, 260) / (228453, 425)

# convert to compressed sparse row matrix
save_npz("preprocessed_data/X.npz", csr_matrix(X))

# load the data of participate age and sample size 
info = open('info.txt', 'r')
info_lines = info.readlines()
sample_size_col_num =   1
age_col_num = 2
verbal_col_num = 6
sample_size_list_verbal,sample_size_list_nonverbal = list(), list()
age_list_verbal, age_list_nonverbal = list(), list()
for x in info_lines:
    vebal_variable = int(x.split()[verbal_col_num])
    if vebal_variable == 1:
        sample_size = int(x.split()[sample_size_col_num])
        sample_size_list_verbal.append(sample_size)
        age = float(x.split()[age_col_num])
        age_list_verbal.append(1/np.sqrt(age))
    else:
        sample_size = int(x.split()[sample_size_col_num])
        sample_size_list_nonverbal.append(sample_size)
        age = float(x.split()[age_col_num])
        age_list_nonverbal.append(1/np.sqrt(age))
# convert list to numpy array 
sample_size_list_verbal = np.array(sample_size_list_verbal).reshape((102,1))
sample_size_list_nonverbal = np.array(sample_size_list_nonverbal).reshape((55,1))
age_list_verbal = np.array(age_list_verbal).reshape((102,1))
age_list_nonverbal = np.array(age_list_nonverbal).reshape((55,1))
# save the study-level covariates to Z matrix
Z_verbal = np.concatenate((sample_size_list_verbal, age_list_verbal), axis=1) # shape: (102,2)
Z_nonverbal = np.concatenate((sample_size_list_nonverbal, age_list_nonverbal), axis=1) # shape: (55,2)
np.savetxt('preprocessed_data/Z_verbal.txt', Z_verbal)
np.savetxt('preprocessed_data/Z_nonverbal.txt', Z_nonverbal)




