import nibabel as nib
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import scipy.stats
# load the mask and anatomy
mask_dil = nib.load("MNI152_T1_2mm_brain_mask.nii.gz")
mask = np.array(mask_dil.dataobj) # unique values: 0/1

brain_anatomy = nib.load("MNI152_T1_2mm.nii.gz")
brain_anatomy = np.array(brain_anatomy.dataobj)

vebal_mu = nib.load("311_basis/verbal_stimuli_NB/output_image_verbal_NB_fr.nii.gz")
vebal_mu = np.array(vebal_mu.dataobj)
k_verbal = 102

nonvebal_mu = nib.load("311_basis/nonverbal_stimuli_NB/output_image_nonverbal_NB_fr.nii.gz")
nonvebal_mu = np.array(nonvebal_mu.dataobj)
k_nonverbal = 55

Z_mu_diff = nib.load("311_basis/Z_mu_twosample.nii.gz")
Z_mu_diff = np.array(Z_mu_diff.dataobj)

pvalue_mu_diff = nib.load("311_basis/pvalue_mu_twosample.nii.gz")
pvalue_mu_diff = np.array(pvalue_mu_diff.dataobj)

MNI_slices = np.arange(start=-24, stop=49, step=12) 
print(MNI_slices)
slices = (MNI_slices + 72) / 2
print(slices)
# set the threshold for corrected p-values 
corrected_pvalue_mu = 0.0333889
#z_mu_threshold = scipy.stats.norm.ppf(1-corrected_pvalue_mu/2)
fig, axes = plt.subplots(3, 7,figsize=(8,6))
for row in range(3):
    for col in range(7):
        slice_index = int(slices[col])
        anatomy_slice = brain_anatomy[:, :, slice_index]
        mask_slice = mask[:, :, slice_index]
        ax = axes[row, col]
        ax.set_title('z='+ str(int(MNI_slices[col])))
        im = ax.imshow(anatomy_slice.T, cmap="Greys_r", interpolation='nearest', origin="lower")
        # first row: estimated mu for verbal stimuli group
        if row == 0:
            verbal_slice = vebal_mu[:, :, slice_index]/k_verbal
            c = verbal_slice[(mask_slice == 1) & (verbal_slice >= 0.00005)]
            x, y = np.where((mask_slice == 1) & (verbal_slice >= 0.00005))
            sc0 = ax.scatter(x,y, s=0.1, c=c, cmap='viridis', vmin=0.00005, vmax=0.001)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axis('off')
        # second row: estimated mu for non-verbal stimuli group
        if row == 1:
            nonverbal_slice = nonvebal_mu[:, :, slice_index]/k_nonverbal
            print(np.min(nonverbal_slice), np.max(nonverbal_slice))
            c = nonverbal_slice[(mask_slice == 1) & (nonverbal_slice >= 0.00005)]
            x, y = np.where((mask_slice == 1) & (nonverbal_slice >= 0.00005))
            sc1 = ax.scatter(x,y, s=0.1, c=c, cmap='viridis', vmin=0.00005, vmax=0.001)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axis('off')
        # third row: Z for verbal stimuli group
        if row == 2:
            pvalue_mu_diff_slice = pvalue_mu_diff[:, :, slice_index]
            c = Z_mu_diff[:, :, slice_index]
            c = c[(mask_slice == 1) & (pvalue_mu_diff_slice < corrected_pvalue_mu)]
            x, y = np.where((mask_slice == 1) & (pvalue_mu_diff_slice < corrected_pvalue_mu))
            sc2 = ax.scatter(x,y, s=0.1, c=c, cmap='bwr', vmin=-3, vmax=3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axis('off')
    fig.subplots_adjust(left=0.01, right=None, bottom=0.01, top=0.95, wspace=0, hspace=0) #tight_layout(pad=0.5, h_pad=0.5) 
# add colorbar for each row
cbaxes0 = fig.add_axes([0.91, 0.69, 0.02, 0.2]) 
cbar0 = fig.colorbar(sc0, ax=axes[0, :], cax=cbaxes0, format='%.0e')
cbar0.mappable.set_clim(0.00005,0.001)
cbaxes1 = fig.add_axes([0.91, 0.38, 0.02, 0.2]) 
cbar1 = fig.colorbar(sc1, ax=axes[1, :], cax=cbaxes1, format='%.0e')
cbar1.mappable.set_clim(0.00005,0.001)
cbaxes2 = fig.add_axes([0.91, 0.07, 0.02, 0.2]) 
cbar2 = fig.colorbar(sc2, ax=axes[2, :], cax=cbaxes2)
cbar2.mappable.set_clim(-3,3)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('311_basis/plot.png', bbox_inches=0)


