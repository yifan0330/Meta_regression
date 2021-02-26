import numpy as np
from scipy.optimize import minimize
import nibabel as nib 
from scipy.sparse import load_npz, csr_matrix, csc_matrix
from scipy.sparse.linalg import inv, cg
from scipy.linalg import eigvals

# load image-wise regressors and voxelwise counts from npz files 
X_sparse = load_npz("preprocessed_data/X.npz") 
#y = load_npz("y.npz").transpose()
Y_verbal_sparse = load_npz("preprocessed_data/y_verbal.npz").transpose()
Y_nonverbal_sparse = load_npz("preprocessed_data/y_nonverbal.npz").transpose()
# convert csc matrix to ndarray
X = X_sparse.toarray() # shape: (292019, 443/365/311/277)/ (194369, 562)
#y = y.toarray() # sum = 2223; max = 3; nonzero counts = 2125
Y_verbal = Y_verbal_sparse.toarray() # sum = 1286; max = 3; nonzero counts = 1244
Y_nonverbal = Y_nonverbal_sparse.toarray() # sum = 937; max = 2; nonzero counts = 923
# load study-level regressors and studywise counts from txt files
Z_verbal = np.loadtxt('preprocessed_data/Z_verbal.txt', dtype=float) # shape: (102, 2)
Z_nonverbal = np.loadtxt('preprocessed_data/Z_nonverbal.txt', dtype=float)  # shape: (55,2)
Y_i_verbal = np.loadtxt('preprocessed_data/Y_i_verbal.txt', dtype=int) # shape: (102,); sum: 1286
Y_i_nonverbal = np.loadtxt('preprocessed_data/Y_i_nonverbal.txt', dtype=int) # shape: (55,); sum: 937
Y_i_verbal = Y_i_verbal.reshape((102,1))
Y_i_nonverbal = Y_i_nonverbal.reshape((55,1))

def log_likelihood_beta(beta, gamma, X, Z):
    P, R = X.shape[1], Z.shape[1]
    mu_X = np.exp(X @ beta)
    mu_Z = np.exp(Z @ gamma)
    # compute log-likelihood
    # l = [Y_g]^T * log(mu_X) + [Y_i]^T * log(mu_Z) - sum(mu^Z)*sum(mu^X)
    sum_mu_X = np.sum(mu_X)
    sum_mu_Z = np.sum(mu_Z)
    l = np.sum(Y_verbal * np.log(mu_X)) + np.sum(Y_i_verbal * np.log(mu_Z)) - sum_mu_X * sum_mu_Z
    # compute the penalized term
    # l* = l + 1/2 log(det(I(theta)))
    mu_g = np.sum(mu_Z) * mu_X # shape: (292019, 1)
    mu_i = np.sum(mu_X) * mu_Z # shape: (102,1)
    mu_g_sqrt = np.sqrt(mu_g)
    X_star = csr_matrix(X).multiply(mu_g_sqrt)# X* = W^(1/2) X; W=diag(mu_g)
    XWX = X_star.T @ X_star # XWX = [W^(1/2) X]^T [W^(1/2) X]
    I_beta = XWX.toarray() # shape: (311, 311)
    I_beta_eigens = np.real(eigvals(I_beta))
    log_I_beta_eigens = np.log(I_beta_eigens)
    log_det_I_beta = np.sum(log_I_beta_eigens)
    l_fr = l + 1/2 * log_det_I_beta
    return l_fr

def log_likelihood_gamma(beta, gamma, X, Z):
    P, R = X.shape[1], Z.shape[1]
    mu_X = np.exp(X @ beta)
    mu_Z = np.exp(Z @ gamma)
    # compute log-likelihood
    # l = [Y_g]^T * log(mu_X) + [Y_i]^T * log(mu_Z) - sum(mu^Z)*sum(mu^X)
    sum_mu_X = np.sum(mu_X)
    sum_mu_Z = np.sum(mu_Z)
    l = np.sum(Y_verbal * np.log(mu_X)) + np.sum(Y_i_verbal * np.log(mu_Z)) - sum_mu_X * sum_mu_Z
    # compute the penalized term
    # l* = l + 1/2 log(det(I(theta)))
    mu_g = np.sum(mu_Z) * mu_X # shape: (292019, 1)
    mu_i = np.sum(mu_X) * mu_Z # shape: (102,1)
    mu_i_sqrt = np.sqrt(mu_i)
    Z_star = csr_matrix(Z_verbal).multiply(mu_i_sqrt) # Z* = V^(1/2) Z; V=diag(mu_i)
    ZVZ = Z_star.T @ Z_star # ZVZ = [V^(1/2) Z]^T [V^(1/2) Z]
    I_gamma = ZVZ.toarray() # shape: (2,2)
    det_I_gamma = np.linalg.det(I_gamma)
    log_det_I_gamma = np.log(det_I_gamma)
    l_fr = l + 1/2 * log_det_I_gamma
    #print(l_fr)
    return l_fr


# initialization for beta and gamma => mu_X and mu_Z
beta_i = np.log(np.sum(Y_verbal)/X.shape[0])
beta = np.full(shape=(X.shape[1], 1), fill_value=beta_i) # shape: (425,1)
gamma = np.array([0, 0]).reshape((Z_verbal.shape[1], 1)) # assume no study-wise regressor

l_fr_beta_0 = log_likelihood_beta(beta, gamma, X, Z_verbal)
l_fr_gamma_0 = log_likelihood_gamma(beta, gamma, X, Z_verbal)

diff = np.inf # initializations for the loop
l_fr_beta_prev = -np.inf
l_fr_gamma_prev = -np.inf

beta_2norm_list = list()
gamma_2norm_list = list()

count = 1
l_fr_beta_list = [l_fr_beta_0]
l_fr_gamma_list = [l_fr_gamma_0]
print(l_fr_beta_list)
print(l_fr_gamma_list)

# start with update beta
while diff > 3:
    # compute first order derivative w.r.t beta
    mu_X = np.exp(X @ beta)
    mu_Z = np.exp(Z_verbal @ gamma)
    mu_g = np.sum(mu_Z) * mu_X # shape: (292019, 1)
    mu_g_sqrt = np.sqrt(mu_g)
    X_star = csr_matrix(X).multiply(mu_g_sqrt)# X* = W^(1/2) X; W=diag(mu_g)
    XWX = X_star.T @ X_star # XWX = [W^(1/2) X]^T [W^(1/2) X]
    XWX_inverse = inv(csc_matrix(XWX)) 
    WX = X_sparse.multiply(mu_g) 
    hadamard_product = (WX @ XWX_inverse).multiply(X) # diag(AB^T) = (A @ B) 1
    h = np.sum(hadamard_product, axis=1) # h = (H_11,H_22,...,H_nn)^T, H = WX(X^T WX)^(-1) X^T
    Jacobian_beta = X.transpose() @ (Y_verbal + 1/2*h - mu_g) # shape: (311,1)
    Jacobian_beta = np.asarray(Jacobian_beta)
    # compute second order derivative w.r.t beta
    Hessian_beta = - XWX.toarray() # shape: (311, 311)
    cg_solution_beta = cg(Hessian_beta, b=Jacobian_beta)[0] # shape: (358,)
    beta_update = cg_solution_beta.reshape((X.shape[1],1))
    beta = beta - beta_update
    # 2 norm
    beta_2norm = np.linalg.norm(beta)
    beta_2norm_list.append(beta_2norm)
    print('2 norm of beta is', beta_2norm)
    l_fr_beta = log_likelihood_beta(beta, gamma, X, Z_verbal)
    l_fr_beta_list.append(l_fr_beta)
    print('log-likelihood (beta) is ', l_fr_beta)
    # Then update gamma based on the updated beta
    M, R = Z_verbal.shape
    mu_X = np.exp(X @ beta)
    mu_Z = np.exp(Z_verbal @ gamma)
    mu_i = np.sum(mu_X) * mu_Z
    mu_i_sqrt = np.sqrt(mu_i)
    Z_star = csr_matrix(Z_verbal).multiply(mu_i_sqrt) # Z* = V^(1/2) Z; V=diag(mu_i)
    ZVZ = Z_star.T @ Z_star # ZVZ = [V^(1/2) Z]^T [V^(1/2) Z]
    ZVZ_inverse = np.linalg.inv(ZVZ.toarray())
    V = np.diag(mu_i.reshape((M,))) 
    H_2 = V @ Z_verbal @ ZVZ_inverse @ Z_verbal.T
    h_2 = np.diag(H_2) # extract diagonal elements 
    h_2 = h_2.reshape((M,1))
    Jacobian_gamma = Z_verbal.transpose() @ (Y_i_verbal + 1/2*h_2 - mu_i) # shape: (2,1)
    Hessian_gamma = -ZVZ.toarray() # shape: (2,2)
    cg_solution_gamma = cg(Hessian_gamma, b=Jacobian_gamma)[0] # shape: (358,)
    gamma_update = cg_solution_gamma.reshape((Z_verbal.shape[1],1))
    gamma = gamma - gamma_update
    # 2 norm
    gamma_2norm = np.linalg.norm(gamma)
    gamma_2norm_list.append(gamma_2norm)
    print('2 norm of gamma is', gamma_2norm)
    l_fr_gamma = log_likelihood_gamma(beta, gamma, X, Z_verbal)
    l_fr_gamma_list.append(l_fr_gamma)
    print('log-likelihood (gamma) is ', l_fr_gamma)
    diff = max(l_fr_beta - l_fr_beta_prev, l_fr_gamma - l_fr_gamma_prev)
    print('difference is ', diff, l_fr_beta - l_fr_beta_prev, l_fr_gamma - l_fr_gamma_prev)
    count = count + 1
    l_fr_beta_prev = l_fr_beta
    l_fr_gamma_prev = l_fr_gamma
    print('-------------------this iteration is finished --------------------')

np.savetxt('beta_Poisson_separate.txt', beta)
np.savetxt('gamma_Poisson_separate.txt', gamma)

np.savetxt('beta_2norm_Poisson_separate.txt', np.array(beta_2norm_list))
np.savetxt('gamma_2norm_Poisson_separate.txt', np.array(gamma_2norm_list))

np.savetxt('l_fr_beta_separate.txt', np.array(l_fr_beta_list))
np.savetxt('l_fr_gamma_separate.txt', np.array(l_fr_gamma_list))
exit()



beta = np.loadtxt('beta_Poisson_separate.txt')
gamma = np.loadtxt('gamma_Poisson_separate.txt')
beta = beta.reshape((X.shape[1], 1))
gamma = gamma.reshape((Z_verbal.shape[1], 1))

mu_X = np.exp(X @ beta)
mu_Z = np.exp(Z_verbal @ gamma)
mu_g = np.sum(mu_Z) * mu_X # max: 0.09769839038423539; mean: 0.004409391213367285; sum: 1287.6260127363012
mu_i = np.sum(mu_X) * mu_Z # min: 16.277281891965814; max: 12.623784438591183; mean: 12.623784438591183; sum: 1287.6260127363007
                            # Y_i_verbal: min: 1; max: 41; mean: 12.607843137254902; sum: 1286

# convert the predicted responses into nifti image 
outside_brain = np.loadtxt('preprocessed_data/outside_brain.txt') # shape: (264027,)
image_dim = np.array([72, 90, 76])
x_min, y_min, z_min = 9, 10, 1
brain_voxel_index = np.setdiff1d(np.arange(np.prod(image_dim)), outside_brain) 
output_image = np.zeros((91,109,91))
for i in range(brain_voxel_index.shape[0]):
    index = brain_voxel_index[i]
    z_coord = index // (image_dim[0]*image_dim[1]) + z_min
    remainder = index % (image_dim[0]*image_dim[1])
    y_coord = remainder // image_dim[0] + y_min
    remainder = remainder % image_dim[0] 
    x_coord = remainder + x_min
    response = mu_g[i]
    output_image[x_coord, y_coord, z_coord] = response

mask_MNI152 = nib.load("MNI152_T1_2mm_brain.nii")
image = nib.Nifti1Image(output_image, mask_MNI152.affine)
image.to_filename('verbal_output_image_BGFS_separate.nii.gz')  # Save as NiBabel file