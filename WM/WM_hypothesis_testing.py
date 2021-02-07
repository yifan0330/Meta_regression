import numpy as np
import scipy
from scipy.sparse import load_npz, csr_matrix, csc_matrix
from scipy.sparse.linalg import inv
import nibabel as nib
from statsmodels.stats.weightstats import ztest
# load foci and counts from npz files 
X_sparse = load_npz("X.npz")
#y_sparse = load_npz("y.npz").transpose()  # shape: (292019, 1); max: 5; sum: 15131
y_verbal_sparse = load_npz("y_verbal.npz").transpose()
y_nonverbal_sparse = load_npz("y_non_verbal.npz").transpose()
y_sparse = load_npz("y.npz").transpose()

X = X_sparse.toarray() # shape: (292019, 365)
y_verbal = y_verbal_sparse.toarray() # sum = 1286; max = 3; nonzero counts = 1244
y_nonverbal = y_nonverbal_sparse.toarray() # sum = 937; max = 2; nonzero counts = 923
y = y_sparse.toarray() # sum = 2223; max = 3; nonzero counts = 2125

k_verbal = 102
k_nonverbal = 55

# Alternative test
N_verbal = np.sum(y_verbal) #1286
V = X.shape[0] # 292019
mu_0_verbal = N_verbal / V # 0.0044038230389118515
eta_0_verbal = np.log(mu_0_verbal) # -5.425282242829142

beta_verbal = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/WM/311_basis/verbal_stimuli_NB/verbal_final_beta_NB.txt')
beta_verbal = beta_verbal.reshape((311,1))
# linear predictor eta = X beta
eta_verbal = X @ beta_verbal # shape: (292019,1)
# mu_i = exp(beta_1 x_i1 + beta_2 x_i2 + ... + beta_k x_ik) = exp(eta_i)
mu_verbal = np.exp(eta_verbal)
# max: 0.15196579680124656; mean: 0.004914785566984065; sum: 1435.2107664851198

# the inverse of the Fisher information matrix is an estimator of the asymptotic covariance matrix
# Fisher information: I = X^T WX 
alpha_verbal = 2.823542210162393040
# w_i = (1 + alpha y_i) mu_i / (1 + alpha mu_i)^2
w_verbal = (1 + alpha_verbal * y_verbal) * mu_verbal / (1+alpha_verbal*mu_verbal)**2
w_sqrt_verbal = np.sqrt(w_verbal)
X_star_verbal = X_sparse.multiply(w_sqrt_verbal)# X* = W^(1/2) X; shape: (292019, 443)
XWX_verbal = X_star_verbal.T @ X_star_verbal

# covariance matrix: [I(theta)]^(-1)
cov_beta_verbal = inv(csc_matrix(XWX_verbal)).toarray() # shape: (311, 311)
#var_beta_verbal = cov_beta_verbal.diagonal().reshape((311,1)) # shape: (311, 1)
#X_square = X * X
#var_eta_verbal = X_square @ var_beta_verbal
#SE_eta_verbal = np.sqrt(var_eta_verbal)

hadamard_product_verbal = np.multiply(X @ cov_beta_verbal, X)
# diag(cov(eta)) = (X  Cov(beta) @ X^T) 1
var_eta_verbal = np.sum(hadamard_product_verbal, axis=1).reshape((292019,1))
var_eta_verbal = np.asarray(var_eta_verbal) # convert numpy matrix to array
SE_eta_verbal = np.sqrt(var_eta_verbal) # max: 17.142728108598657; min: 0.16601191357549092

N_nonverbal = np.sum(y_nonverbal) # 937
V = X.shape[0] # 292019
mu_0_nonverbal = N_nonverbal / V # 0.0032086953246192886
eta_0_nonverbal = np.log(mu_0_nonverbal) # -5.741890865388284
beta_nonverbal = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/WM/311_basis/nonverbal_stimuli_NB/nonverbal_final_beta_NB.txt')
beta_nonverbal = beta_nonverbal.reshape((311,1))
# linear predictor eta = X beta
eta_nonverbal = X @ beta_nonverbal # shape: (292019,1)

# mu_i = exp(beta_1 x_i1 + beta_2 x_i2 + ... + beta_k x_ik) = exp(eta_i)
mu_nonverbal = np.exp(eta_nonverbal)
# max: 0.2157275333368224; mean: 0.00395143345081951; sum: 1153.8936448748625

# the inverse of the Fisher information matrix is an estimator of the asymptotic covariance matrix
# Fisher information: I = X^T WX 
alpha_nonverbal = 1.687694324450697891
# w_i = (1 + alpha y_i) mu_i / (1 + alpha mu_i)^2
w_nonverbal = (1 + alpha_nonverbal * y_nonverbal) * mu_nonverbal / (1+alpha_nonverbal*mu_nonverbal)**2
w_sqrt_nonverbal = np.sqrt(w_nonverbal)
X_star_nonverbal = X_sparse.multiply(w_sqrt_nonverbal)# X* = W^(1/2) X; shape: (292019, 443)
XWX_nonverbal = X_star_nonverbal.T @ X_star_nonverbal

# covariance matrix: [I(theta)]^(-1)
cov_beta_nonverbal = inv(csc_matrix(XWX_nonverbal)).todense()

hadamard_product_nonverbal = np.multiply(X @ cov_beta_nonverbal, X)
# diag(cov(eta)) = (X Cov(beta) @ X^T) 1
var_eta_nonverbal = np.sum(hadamard_product_nonverbal, axis=1).reshape((292019,1))
var_eta_nonverbal = np.asarray(var_eta_nonverbal) # convert numpy matrix to array
SE_eta_nonverbal = np.sqrt(var_eta_nonverbal) # max: 25.391250495375232; 0.18952373102975356

# Two sample test
#Z_eta_diff = (eta_verbal - eta_nonverbal) / np.sqrt(SE_eta_verbal**2 + SE_eta_nonverbal**2)

# 311 bases: std=1.3263700078453624; max = 4.252311194331785; min = -3.9659995662020506
#p_eta_diff = scipy.stats.norm.sf(np.abs(Z_eta_diff))*2 #twosided
# 44261 voxels with p-value less than 0.05

# By CLT (eta_hat-eta)/SE_eta -> N(0,1)
# By delta method (exp(eta_hat) - exp(eta))/((e^eta_hat)'SE_e^eta -> N(0,1)
# for mu = exp(eta): asymptotic mean = exp(mean(eta))
#                    asymptotic SE = exp(eta) SE_eta
SE_mu_verbal = np.exp(eta_verbal)*SE_eta_verbal # shape: (292019, 1)
SE_mu_nonverbal = np.exp(eta_nonverbal)*SE_eta_nonverbal # shape: (292019, 1)
# Z-test
#Z_mu_diff = (mu_verbal - mu_nonverbal) / np.sqrt(SE_mu_verbal**2 + SE_mu_nonverbal**2)
# max: 3.8533123475027593; min: -2.4827355867620837; mean: -0.40152295037649777, std: 0.962981753800927
Z_mu_diff = (mu_verbal/k_verbal - mu_nonverbal/k_nonverbal) / np.sqrt(SE_mu_verbal**2/k_verbal**2 + SE_mu_nonverbal**2/k_nonverbal**2)
# max: 2.599024323081608; min: -3.335022319382749; mean: 0.20715701927987287, std: 0.9980280422664004

#p_mu = scipy.stats.norm.sf(np.abs(Z_mu))*2
p_mu_diff = (1-scipy.stats.norm.cdf(x=np.abs(Z_mu_diff)))*2


# convert the Z-statistics into nifti image 
outside_brain = np.loadtxt('311_basis/outside_brain.txt') # shape: (250925,)
brain_voxel_index = np.setdiff1d(np.arange(542944), outside_brain)
output_image = np.zeros((91,109,91))
for i in range(brain_voxel_index.shape[0]):
    index = brain_voxel_index[i]
    z_coord = index // (76*94) + 3
    remainder = index % (76*94)
    y_coord = remainder // 76 + 8
    remainder = remainder % 76
    x_coord = remainder % 76 + 7
    response = Z_mu_diff[i]
    output_image[x_coord, y_coord, z_coord] = response

image = nib.Nifti1Image(output_image, np.eye(4))
image.to_filename('311_basis/Z_mu_twosample.nii.gz')