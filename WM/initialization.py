import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

# load the brain mask
mask_dil = nib.load("MNI152_T1_2mm_brain_mask_dil.nii")
mask = np.array(mask_dil.dataobj) # unique values: 0/1
n_brain_voxel = np.sum(mask) # 292019/194369 within-brain voxels
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
foci = pd.merge(foci_data, info_df, on='contrast') # shape: (2240, 5)

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
foci = foci.drop(labels=foci_outside, axis=0) # shape: (2223, 5)

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

x_min, x_max = np.min(xx), np.max(xx) # 7, 82
y_min, y_max = np.min(yy), np.max(yy) # 8, 101
z_min, z_max = np.min(zz), np.max(zz) # 3, 78

# remove the foci if its coordinates are beyond (7~82, 8~101, 3~78)
drop_condition2 = foci[(foci["x"]<x_min)|(foci["x"]>x_max)|(foci["y"]<y_min)|(foci["y"]>y_max)|(foci["z"]<z_min)|(foci["z"]>z_max)].index
foci = foci.drop(drop_condition2) #shape: (2223, 5)

image_dim = np.array([len(xx), len(yy), len(zz)]) # [76 94 76]

# initialize the count of foci as 0
foci_count = np.zeros(shape=(len(xx), len(yy), len(zz))) # (76, 94, 76)
for row_num in range(foci.shape[0]):
    row = foci.iloc[row_num].values.tolist()
    x,y,z = row[1:4]
    x_coord, y_coord, z_coord = x-x_min, y-y_min, z-z_min
    foci_count[x_coord, y_coord, z_coord] += 1 # sum: 2223; max: 3

verbal_foci_count = np.zeros(shape=(len(xx), len(yy), len(zz))) # (76, 94, 76)
for row_num in range(foci.shape[0]): 
    row = foci.iloc[row_num].values.tolist()
    if row[4] == 1:
        x,y,z = row[1:4]
        x_coord, y_coord, z_coord = x-x_min, y-y_min, z-z_min
        verbal_foci_count[x_coord, y_coord, z_coord] += 1 # sum: 1286

nonverbal_foci_count = np.zeros(shape=(len(xx), len(yy), len(zz)))
for row_num in range(foci.shape[0]): 
    row = foci.iloc[row_num].values.tolist()
    if row[4] == 0:
        x,y,z = row[1:4]
        x_coord, y_coord, z_coord = x-x_min, y-y_min, z-z_min
        nonverbal_foci_count[x_coord, y_coord, z_coord] += 1 # sum: 937

# smooth 3D image of counts with Gaussian kernel
sigma = 15/2
gaussian_FWHM = np.sqrt(8*np.log(2)) * sigma
FWHM = np.repeat(gaussian_FWHM, 3)
foci_count_nii = nib.Nifti1Image(foci_count, mask_dil.affine)
smooth_foci_count = nilearn.image.smooth_img(foci_count_nii, FWHM)
smooth_foci_count = np.array(smooth_foci_count.dataobj)
smooth_foci_count_reshape = smooth_foci_count.reshape((np.prod(image_dim), )) # shape: (542944,)

verbal_foci_count_nii = nib.Nifti1Image(verbal_foci_count, mask_dil.affine)
smooth_verbal_foci_count = nilearn.image.smooth_img(verbal_foci_count_nii, FWHM)
smooth_verbal_foci_count = np.array(smooth_verbal_foci_count.dataobj)
smooth_verbal_foci_count_reshape = smooth_verbal_foci_count.reshape((np.prod(image_dim), )) # shape: (542944,)

nonverbal_foci_count_nii = nib.Nifti1Image(nonverbal_foci_count, mask_dil.affine)
smooth_nonverbal_foci_count = nilearn.image.smooth_img(nonverbal_foci_count_nii, FWHM)
smooth_nonverbal_foci_count = np.array(smooth_nonverbal_foci_count.dataobj)
smooth_nonverbal_foci_count_reshape = smooth_nonverbal_foci_count.reshape((np.prod(image_dim), )) # shape: (542944,)

# remove the voxels outside brain mask
outside_brain = np.loadtxt("outside_brain.txt")
outside_brain = outside_brain.astype(int) # convert it to integer array
y = np.delete(smooth_foci_count_reshape, obj=outside_brain) # shape: (292019,)  
# sum: 1974.2219965567635; max: 0.06552637805648526; mean: 0.006760594333097379
save_npz("y_init.npz", csr_matrix(y))

y_verbal = np.delete(smooth_verbal_foci_count_reshape, obj=outside_brain) # shape: (292019,)  
# sum: 1134.2473067772848; max: 0.04310029364418784; mean: 0.003884155848685479
save_npz("verbal_init.npz", csr_matrix(y_verbal))

y_nonverbal = np.delete(smooth_nonverbal_foci_count_reshape, obj=outside_brain) # shape: (292019,)  
# sum: 839.9746897794784; max: 0.030369614809501583; mean: 0.002876438484411899
save_npz("nonverbal_init.npz", csr_matrix(y_nonverbal))
