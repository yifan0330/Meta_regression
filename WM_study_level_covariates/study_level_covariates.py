import numpy as np
from scipy.sparse import load_npz, csr_matrix, csc_matrix
from scipy.special import factorial
from scipy.linalg import eigvals
from scipy.sparse.linalg import inv, cg
import nibabel as nib 

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

def Fisher_Information(mu_X, mu_Z, X, Z):
    mu_g = np.sum(mu_Z) * mu_X # shape: (292019, 1)
    mu_i = np.sum(mu_X) * mu_Z # shape: (102,1)
    mu_g_sqrt = np.sqrt(mu_g)
    X_star = csr_matrix(X).multiply(mu_g_sqrt)# X* = W^(1/2) X; W=diag(mu_g)
    XWX = X_star.T @ X_star # XWX = [W^(1/2) X]^T [W^(1/2) X]
    second_order_derivative_beta = - XWX.toarray() # shape: (311, 311)
    mu_i_sqrt = np.sqrt(mu_i)
    Z_star = csr_matrix(Z_verbal).multiply(mu_i_sqrt) # Z* = V^(1/2) Z; V=diag(mu_i)
    ZVZ = Z_star.T @ Z_star # ZVZ = [V^(1/2) Z]^T [V^(1/2) Z]
    second_order_derivative_gamma = -ZVZ.toarray() # shape: (2,2)
    # d^2 l / d beta d gamma  = - [X^T * mu^X] * [(mu^Z)^T * Z]
    second_order_derivative_cross_term = - X.transpose() @ mu_g @ mu_Z.transpose() @ Z_verbal # shape: (311, 2)
    Hessian_top = np.concatenate((second_order_derivative_beta, second_order_derivative_cross_term), axis=1) # shape: (311, 313)
    Hessian_bottom = np.concatenate((second_order_derivative_cross_term.transpose(), second_order_derivative_gamma), axis=1) # shape: (2, 313)
    Hessian = np.concatenate((Hessian_top, Hessian_bottom), axis=0) # shape: (313, 313)
    return Hessian

# Log-linear (Poisson) model
def loglikelihood_Poisson(Y_g, Y_i, mu_X, mu_Z, I):
    # l = [Y_g]^T * log(mu_X) + [Y_i]^T * log(mu_Z) - sum(mu^Z)*sum(mu^X)
    sum_mu_X = np.sum(mu_X)
    sum_mu_Z = np.sum(mu_Z)
    l = np.sum(Y_g * np.log(mu_X)) + np.sum(Y_i * np.log(mu_Z)) - sum_mu_X * sum_mu_Z
    # compute the penalized term
    # l* = l + 1/2 log(det(I(theta)))
    det_I = np.linalg.det(I)
    log_det_I = np.log(det_I)
    l_fr = l + 1/2 * log_det_I
    return l, l_fr

# initialization 
# mu_X only contains the effects of image-wise regressors 
# mu_Z only contains the effects of study-level covariates 
beta_0 = np.log(sum(Y_verbal) / 292019) # assume the occurance of foci is equal likely across whole brain
beta = np.full(shape=(X.shape[1], 1), fill_value=beta_0) # shape: (311,1)
mu_X = np.exp(X @ beta)
gamma = np.array([0,0]).reshape((2,1)) # assume no study-wise regressor
mu_Z = np.exp(Z_verbal @ gamma)

I = Fisher_Information(mu_X, mu_Z, X, Z_verbal) # shape: (313, 313)

# set up variables needed in iteration 
diff = np.inf 
l_fr_prev = -np.inf
count = 0

beta_2norm_list = [np.linalg.norm(beta)]
gamma_2norm_list = [np.linalg.norm(gamma)]

l_0, l_fr_0 = loglikelihood_Poisson(Y_verbal, Y_i_verbal, mu_X, mu_Z, I)
l_list = [l_0]
l_fr_list = [l_fr_0]
print(l_0, l_fr_0)

while diff > 1: 
    # voxelwise intensity summation (over all studies)
    mu_g = np.sum(mu_Z) * mu_X # shape: (292019, 1)
    # study-wise intensity summation (over all voxels)
    mu_i = np.sum(mu_X) * mu_Z # shape: (102,1)
    # first order derivatives 
    first_order_derivative_beta = X.transpose() @ (Y_verbal - mu_g) # shape: (311,1)
    first_order_derivative_gamma = Z_verbal.transpose() @ (Y_i_verbal - mu_i) # shape: (2,1)
    Jacobian = np.concatenate((first_order_derivative_beta, first_order_derivative_gamma), axis=0) # shape: (313, 1)
    # second order derivatives 
    Hessian = Fisher_Information(mu_X, mu_Z, X, Z_verbal)
    cg_solution = cg(A=Hessian, b=Jacobian)[0] # shape: (313,)
    update = cg_solution.reshape((313, 1))
    beta_gamma = np.concatenate((beta, gamma), axis=0) # shape: (313, 1)
    beta_gamma = beta_gamma - update
    # split beta and gamma
    beta = beta_gamma[:311, :] # first 311 elements 
    gamma = beta_gamma[311:, :] # last 2 elements
    beta_2norm = np.linalg.norm(beta)
    beta_2norm_list.append(beta_2norm)
    gamma_2norm = np.linalg.norm(gamma)
    gamma_2norm_list.append(gamma_2norm)
    print('2 norm of beta and gamma are ', beta_2norm, gamma_2norm)
    # update mean intensity corresponding to image-wise and study-level regressors
    mu_X = np.exp(X @ beta) # shape: (292019, 1)
    mu_Z = np.exp(Z_verbal @ gamma) # shape: (102, 1)
    # compute log-likelihood in the current iteration
    l, l_fr = loglikelihood_Poisson(Y_verbal, Y_i_verbal, mu_X, mu_Z, I)
    l_list.append(l)
    l_fr_list.append(l_fr)
    diff = l_fr - l_fr_prev
    print('Firth penalized log-likelihood in the current iteration is', l_fr)
    print('difference of log-likelihood is', diff)
    count += 1
    l_fr_prev = l_fr
    print(count)
    print('------------------------')

print(count)
print('2 norm of beta list:', beta_2norm_list)
print("log-likelihood list", l_list)
#print("penalized log-likelihood list", l_fr_list)
np.savetxt('beta_Poisson.txt', beta, fmt='%f')
np.savetxt('gamma_Poisson.txt', gamma, fmt='%f')
np.savetxt('beta_2norm_Poisson.txt', np.array(beta_2norm_list), fmt='%f')
np.savetxt('gamma_2norm_Poisson.txt', np.array(gamma_2norm_list), fmt='%f')
np.savetxt('l_Poisson.txt', np.array(l_list), fmt='%f')
#np.savetxt('l_fr_Poisson.txt', np.array(l_fr_list), fmt='%f')
exit()

beta_Poisson = np.loadtxt('results/beta_Poisson.txt')
gamma_Poisson = np.loadtxt('results/gamma_Poisson.txt')
mu_X = np.exp(X @ beta_Poisson)
mu_Z = np.exp(Z_verbal @ gamma_Poisson)
mu_g = np.sum(mu_Z) * mu_X # max: 0.2488269212333471; mean: 0.004744023575601006; sum: 1385.3450205234303
mu_i = np.sum(mu_X) * mu_Z # min: 7.022588903175725; max: 30.606897451990292; mean: 13.5818139267003; sum: 1385.3450205234308
                            # Y_i_verbal: min: 1; max: 41; mean: 12.607843137254902; sum: 1286

# convert the predicted responses into nifti image 
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
    response = mu_g[i]
    output_image[x_coord, y_coord, z_coord] = response

image = nib.Nifti1Image(output_image, np.eye(4))
image.to_filename('verbal_output_image_poisson.nii.gz')  # Save as NiBabel file