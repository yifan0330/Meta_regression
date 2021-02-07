import numpy as np
import scipy
from scipy.sparse import load_npz, csr_matrix, csc_matrix
from scipy.sparse.linalg import inv
import nibabel as nib
from statsmodels.stats.weightstats import ztest
# load foci and counts from npz files 
X_sparse = load_npz("X.npz")
#y_sparse = load_npz("y.npz").transpose()  # shape: (292019, 1); max: 5; sum: 15131

X = X_sparse.toarray() # shape: (292019, 443)
y = y_sparse.toarray() # sum = 15131

# Null Hypothesis H_0: mu = mu_0
# Alternative test
N = np.sum(y) #15131
V = X.shape[0] # 292019
mu_0 = N / V # 0.051815121618798775
eta_0 = np.log(mu_0) # -2.960073249175136

beta = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_result_1e-6/adjusted_final_beta_NB_fr.txt')
beta = beta.reshape((443,1))
# linear predictor eta = X beta
eta = X @ beta # shape: (292019,)

# mu_i = exp(beta_1 x_i1 + beta_2 x_i2 + ... + beta_k x_ik) = exp(eta_i)
mu = np.exp(eta)
# max: 0.48851712153081045; mean: 0.05250904289049421; sum: 15333.638195839229

# the inverse of the Fisher information matrix is an estimator of the asymptotic covariance matrix
# Fisher information: I = X^T WX 
alpha = 0.6873619734277702786
# w_i = (1 + alpha y_i) mu_i / (1 + alpha mu_i)^2
w = (1 + alpha * y) * mu / (1+alpha*mu)**2
w_sqrt = np.sqrt(w)
X_star = X_sparse.multiply(w_sqrt)# X* = W^(1/2) X; shape: (292019, 443)
XWX = X_star.T @ X_star
# covariance matrix: [I(theta)]^(-1)
cov_beta = np.array(inv(csc_matrix(XWX)).todense())
var_beta = cov_beta.diagonal().reshape((443,1)) # shape: (443, 1)
hadamard_product = np.multiply(X @ cov_beta, X)
# diag(cov(eta)) = (X Cov(beta) @ X^T) 1
var_eta = np.sum(hadamard_product, axis=1).reshape((292019,1))
SE_eta = np.sqrt(var_eta)

# By CLT (eta_hat-eta)/SE_eta -> N(0,1)
# By delta method (exp(eta_hat) - exp(eta))/((e^eta_hat)'SE_e^eta -> N(0,1)
# for mu = exp(eta): asymptotic mean = exp(mean(eta))
#                    asymptotic SE = exp(eta) SE_eta
SE_mu = np.exp(eta) * SE_eta # shape: (292019,1)
# max: 0.6233359140965493; min: 9.369810918134201e-14

# Z-test
Z_eta = (eta - eta_0)/SE_eta 
Z_mu = (mu - mu_0)/SE_mu
#print(np.count_nonzero(Z_eta > 1.645),np.count_nonzero(Z_mu > 1.645))
#print(np.count_nonzero(Z_eta<-1.96), np.count_nonzero(Z_eta>1.96))
#print(np.count_nonzero(Z_mu<-1.96), np.count_nonzero(Z_mu>1.96))
#a = [83939,  84137, 109722, 120782, 121016, 122151, 154497, 188410, 229678, 229798, 241257, 247636, 255744, 263227]
#p_value_mu = ztest(x1=mu, value=mu_0)


# convert the Z-statistics into nifti image 
outside_brain = np.loadtxt('outside_brain.txt') # shape: (250925,)
brain_voxel_index = np.setdiff1d(np.arange(542944), outside_brain)
output_image = np.zeros((91,109,91))
for i in range(brain_voxel_index.shape[0]):
    index = brain_voxel_index[i]
    z_coord = index // (76*94) + 3
    remainder = index % (76*94)
    y_coord = remainder // 76 + 8
    remainder = remainder % 76
    x_coord = remainder % 76 + 7
    response = Z_mu[i]
    output_image[x_coord, y_coord, z_coord] = response

image = nib.Nifti1Image(output_image, np.eye(4))
image.to_filename('z_statistic_mu.nii.gz')