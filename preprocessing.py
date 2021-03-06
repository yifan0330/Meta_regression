import nibabel as nib
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz

# load the brain mask
mask_dil = nib.load("MNI152_T1_2mm_brain_mask_dil.nii")
mask = np.array(mask_dil.dataobj) # unique values: 0/1
n_brain_voxel = np.sum(mask) # 292019 within-brain voxels
dim_mask = np.shape(mask) # [91,109,91]

# load foci data
foci = pd.read_csv('BrainMap_foci.csv', index_col=False) # shape: (15213, 5)

# transform the foci from MNI152 coordinates to voxel space
origin = (90,-126,-72)
foci["x"] = (origin[0] - foci["x"])/2 # round to the nearest integer
foci["y"] = (foci["y"] - origin[1])/2
foci["z"] = (foci["z"] - origin[2])/2
foci = foci.round(0) # round to the nearest integer
foci = foci.astype("int64")

# remove the foci if its coordinates are beyond (91,109,91)
drop_condition = foci[(foci["x"]<0)|(foci["x"]>90)|(foci["y"]<0)|(foci["y"]>108)|(foci["z"]<0)|(foci["z"]>90)].index
foci = foci.drop(drop_condition) # shape: (15210, 5)
foci = foci.reset_index(drop=True)
# remove the foci if it falls outside the brain mask
n_foci = foci.shape[0] # 15210

foci_outside = list()
for i in range(n_foci):
    foci_coord = foci.iloc[i,2:5]
    if mask[foci_coord[0], foci_coord[1], foci_coord[2]] == 0:
        foci_outside.append(i)
foci = foci.drop(labels=foci_outside, axis=0) # shape: (15116, 5)

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
xx = np.setdiff1d(xx, x_remove) #[7,82] 76
yy = np.setdiff1d(yy, y_remove) #[8,101] 94
zz = np.setdiff1d(zz, z_remove) #[3,78] 76

# remove the foci if its coordinates are beyond (7~82, 8~101, 3~78)
drop_condition2 = foci[(foci["x"]<7)|(foci["x"]>82)|(foci["y"]<8)|(foci["y"]>101)|(foci["z"]<3)|(foci["z"]>78)].index
foci = foci.drop(drop_condition2) #shape: (15116, 5)

# create B-spline basis for x/y/z coordinate
x_deg = 3
x_knots = np.arange(min(xx), max(xx), step=15) # 7, 22, 37, 52, 67
x_design_matrix = patsy.dmatrix("bs(x, knots=x_knots, degree=3,include_intercept=False)", data={"x":xx},return_type='matrix')
x_design_array = np.array(x_design_matrix) 
x_spline = x_design_array[:,1:] # remove the first column (every element is 1); shape: (76, 8) 
# Note the B-spline basis is a partition of unity
x_rowsum = np.sum(x_spline, axis=1)
# Note the B-spline basis is sparse
x_colmean = np.mean(x_spline, axis=0)

y_deg = 3
y_knots = np.arange(min(yy), max(yy), step=15) # 8, 23, 38, 53, 68, 83, 98
y_design_matrix = patsy.dmatrix("bs(y, knots=y_knots, degree=3,include_intercept=False)", data={"y":yy},return_type='matrix')
y_design_array = np.array(y_design_matrix) 
y_spline = y_design_array[:,1:] # shape: (94, 10) 
y_rowsum = np.sum(y_spline, axis=1)
y_colmean = np.mean(y_spline, axis=0)

z_deg = 3
z_knots = np.arange(min(zz), max(zz), step=15) # 3, 18, 33, 48, 63
z_design_matrix = patsy.dmatrix("bs(z, knots=z_knots, degree=3,include_intercept=False)", data={"z":zz},return_type='matrix')
z_design_array = np.array(z_design_matrix) 
z_spline = z_design_array[:,1:] # shape: (76, 8) 
z_rowsum = np.sum(z_spline, axis=1)
z_colmean = np.mean(z_spline, axis=0)

image_dim = np.array([x_spline.shape[0], y_spline.shape[0], z_spline.shape[0]]) #[76 94 76]
x_df, y_df, z_df = x_spline.shape[1], y_spline.shape[1], z_spline.shape[1]
image_df = np.array([x_df, y_df, z_df]) # 8, 10, 8

# convert spline bases in 3 dimesion to data matrix by tensor product
X = np.empty(shape=[np.prod(image_dim), np.prod(image_df)]) #shape: (542944, 640)
xyz_spline = np.zeros((x_spline.shape[0], y_spline.shape[0], z_spline.shape[0])) #shape: (76, 94, 76)
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
x_min, x_max = np.min(xx), np.max(xx) # 7, 82
y_min, y_max = np.min(yy), np.max(yy) # 8, 101
z_min, z_max = np.min(zz), np.max(zz) # 3, 78
foci['index'] = (foci['x'] - x_min) + 76*(foci['y'] - y_min) + 76*94*(foci['z']-z_min)

# initialize the response vector as 0
y = np.zeros((np.prod(image_dim),)) # shape: (542944,)
foci_index = foci['index'].to_numpy()
for i in range(foci.shape[0]):
    index = foci_index[i]
    y[index] += 1
y = y.astype(int)
# sum of y: 23250; number of nonzero elements: 20721; max of y: 9

## remove the voxels outside brain mask
np.set_printoptions(suppress=True)
outside_brain = np.array([])
for i in range(image_dim[0]):
    for j in range(image_dim[1]):
        for k in range(image_dim[2]):
            coord = np.array([xx[i], yy[j], zz[k]])
            if mask[coord[0], coord[1], coord[2]] == 0:
                index = (coord[0] - x_min) + 76*(coord[1] - y_min) + 76*94*(coord[2]-z_min)
                outside_brain = np.append(outside_brain, index)
outside_brain = outside_brain.astype(int)
np.savetxt("outside_brain.txt", outside_brain, fmt="%i")
X = np.delete(X, obj=outside_brain, axis=0) # shape: (292019, 640)
y = np.delete(y, obj=outside_brain) # shape: (292019,)

## remove tensor product basis that have no support in the brain
no_suppport_basis = np.array([])
for bx in range(x_df):
    for by in range(y_df):
        for bz in range(z_df):
            basis_index = bz + z_df*by + z_df*y_df*bx
            basis_coef = X[:, basis_index]
            if np.max(basis_coef) <= 0.01:
                no_suppport_basis = np.append(no_suppport_basis, basis_index)
no_suppport_basis = no_suppport_basis.astype(int)
# 197 tensor product of spline basis have no support in brain mask (computationally)
X = np.delete(X, obj=no_suppport_basis, axis=1) # shape: (292019, 443)

X = csr_matrix(X) # convert to compressed sparse row matrix
y = csr_matrix(y)
save_npz("X.npz", X)
save_npz("y.npz", y)

#np.savetxt("X.txt", X, fmt='%f')
#np.savetxt("y.txt", y,fmt='%i')


