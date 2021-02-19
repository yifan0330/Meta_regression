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
mask_reshape = mask.reshape((np.prod(dim_mask), ))

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

# remove the foci if its coordinates are beyond (7~82, 8~101, 3~78)
drop_condition2 = foci[(foci["x"]<0)|(foci["x"]>90)|(foci["y"]<0)|(foci["y"]>108)|(foci["z"]<0)|(foci["z"]>90)].index
foci = foci.drop(drop_condition2) #shape: (2223, 5)

# initialize the count of foci as 0
foci_count = np.zeros(shape=(dim_mask[0], dim_mask[1], dim_mask[2])) # (91, 109, 91)
for row_num in range(foci.shape[0]):
    row = foci.iloc[row_num].values.tolist()
    x,y,z = row[1:4]
    foci_count[x,y,z] += 1 # sum: 2223; max: 3

verbal_foci_count = np.zeros(shape=(dim_mask[0], dim_mask[1], dim_mask[2])) # (91,109,91)
for row_num in range(foci.shape[0]): 
    row = foci.iloc[row_num].values.tolist()
    if row[4] == 1:
        x,y,z = row[1:4]
        verbal_foci_count[x,y,z] += 1 # sum: 1286

nonverbal_foci_count = np.zeros(shape=(dim_mask[0], dim_mask[1], dim_mask[2])) # (91,109,91)
for row_num in range(foci.shape[0]): 
    row = foci.iloc[row_num].values.tolist()
    if row[4] == 0:
        x,y,z = row[1:4]
        nonverbal_foci_count[x,y,z] += 1 # sum: 937

# initialize the response vector as 0
y_verbal = np.zeros((dim_mask[0],dim_mask[1],dim_mask[2])) # shape: (91,109,91)
for i in range(foci.shape[0]):
    row = foci.iloc[i].tolist()
    verbal = row[4]
    x,y,z = row[1:4]
    if verbal == 1:
        y_verbal[x,y,z] += 1
y_verbal = y_verbal.astype(int)
y_verbal = y_verbal[mask == 1]

# smooth 3D image of counts with Gaussian kernel
# remove the voxels outside brain mask
# and save to npz file
sigma = 15/2
gaussian_FWHM = np.sqrt(8*np.log(2)) * sigma
print(gaussian_FWHM)
FWHM = np.repeat(gaussian_FWHM, 3)
foci_count_nii = nib.Nifti1Image(foci_count, mask_dil.affine)
smooth_foci_count = nilearn.image.smooth_img(foci_count_nii, FWHM)
smooth_foci_count = np.array(smooth_foci_count.dataobj)
smooth_foci_intensity = smooth_foci_count[mask == 1] # shape: (292019,)
# sum: 2142.516460170923; max: 0.06552637805648526; mean: 0.007336907736040885
save_npz("y_init.npz", csr_matrix(smooth_foci_intensity))

verbal_foci_count_nii = nib.Nifti1Image(verbal_foci_count, mask_dil.affine)
smooth_verbal_foci_count = nilearn.image.smooth_img(verbal_foci_count_nii, FWHM)
smooth_verbal_foci_count = np.array(smooth_verbal_foci_count.dataobj)
smooth_verbal_foci_intensity = smooth_verbal_foci_count[mask == 1] # shape: (292019,)
# sum: 1235.1502880129597; max: 0.04310029364418784; mean: 0.004229691520116704
save_npz("verbal_init.npz", csr_matrix(smooth_verbal_foci_intensity))

nonverbal_foci_count_nii = nib.Nifti1Image(nonverbal_foci_count, mask_dil.affine)
smooth_nonverbal_foci_count = nilearn.image.smooth_img(nonverbal_foci_count_nii, FWHM)
smooth_nonverbal_foci_count = np.array(smooth_nonverbal_foci_count.dataobj)
smooth_nonverbal_foci_intensity = smooth_nonverbal_foci_count[mask == 1] # shape: (292019,)
# sum: 907.3661721579631; max: 0.030369614809501583; mean: 0.00310721621592418
save_npz("nonverbal_init.npz", csr_matrix(smooth_nonverbal_foci_intensity))

# convert to nifti image
smooth_verbal_foci_count[mask == 0] = 0 
image = nib.Nifti1Image(smooth_verbal_foci_count, mask_dil.affine)
image.to_filename('verbal_init.nii.gz')  # Save as NiBabel file

smooth_verbal_foci_count[mask == 0] = 0 
image = nib.Nifti1Image(smooth_verbal_foci_count, mask_dil.affine)
image.to_filename('nonverbal_init.nii.gz')  # Save as NiBabel file