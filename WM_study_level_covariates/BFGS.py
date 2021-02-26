import numpy as np
from scipy.optimize import minimize
import nibabel as nib 
from scipy.sparse import load_npz, csr_matrix, csc_matrix
from scipy.sparse.linalg import inv
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
    return -Hessian

def nll(x, X, Z):
    P, R = X.shape[1], Z.shape[1]
    beta = x[:P].reshape((P,1))
    gamma = x[P:].reshape((R,1))
    mu_X = np.exp(X @ beta)
    mu_Z = np.exp(Z @ gamma)
    # compute log-likelihood
    # l = [Y_g]^T * log(mu_X) + [Y_i]^T * log(mu_Z) - sum(mu^Z)*sum(mu^X)
    sum_mu_X = np.sum(mu_X)
    sum_mu_Z = np.sum(mu_Z)
    # replace element 0 in mu_X and mu_Z with machine epsilon
    #machine_epsilon = np.finfo(float).eps 
    #mu_X[mu_X == 0] = machine_epsilon
    #mu_Z[mu_Z == 0] = machine_epsilon
    l = np.sum(Y_verbal * np.log(mu_X)) + np.sum(Y_i_verbal * np.log(mu_Z)) - sum_mu_X * sum_mu_Z
    # compute the penalized term
    # l* = l + 1/2 log(det(I(theta)))
    I = Fisher_Information(mu_X, mu_Z, X, Z)
    I_eigens = np.real(eigvals(I))
    log_I_eigens = np.log(I_eigens)
    log_det_I = np.sum(log_I_eigens)
    print('log determinant is', log_det_I)
    l_fr = l + 1/2 * log_det_I
    print(-l, -l_fr)
    return -l

def jac(x, X, Z):
    P = X.shape[1]
    M, R = Z.shape
    beta = x[:P].reshape((P,1))
    gamma = x[P:].reshape((R,1))
    mu_X = np.exp(X @ beta)
    mu_Z = np.exp(Z @ gamma)
    # voxel-wise intensity summation (over all studies within a group)
    mu_g = np.sum(mu_Z) * mu_X # shape: (292019, 1)
    # study-wise intensity summation (over all voxels)
    mu_i = np.sum(mu_X) * mu_Z # shape: (102,1)
    # first order derivatives for beta (with penalized term)
    mu_g_sqrt = np.sqrt(mu_g)
    X_star = csr_matrix(X).multiply(mu_g_sqrt)# X* = W^(1/2) X; W=diag(mu_g)
    XWX = X_star.T @ X_star # XWX = [W^(1/2) X]^T [W^(1/2) X]
    XWX_inverse = inv(csc_matrix(XWX)) 
    WX = X_sparse.multiply(mu_g) 
    hadamard_product = (WX @ XWX_inverse).multiply(X) # diag(AB^T) = (A @ B) 1
    h = np.sum(hadamard_product, axis=1) # h = (H_11,H_22,...,H_nn)^T, H = WX(X^T WX)^(-1) X^T
    first_order_derivative_beta = X.transpose() @ (Y_verbal + 1/2*h - mu_g) # shape: (311,1)
    first_order_derivative_beta = np.asarray(first_order_derivative_beta)
    # first order derivatives for gamma (with penalized term)
    mu_i_sqrt = np.sqrt(mu_i)
    Z_star = csr_matrix(Z_verbal).multiply(mu_i_sqrt) # Z* = V^(1/2) Z; V=diag(mu_i)
    ZVZ = Z_star.T @ Z_star # ZVZ = [V^(1/2) Z]^T [V^(1/2) Z]
    ZVZ_inverse = np.linalg.inv(ZVZ.toarray())
    V = np.diag(mu_i.reshape((M,))) 
    H_2 = V @ Z_verbal @ ZVZ_inverse @ Z_verbal.T
    h_2 = np.diag(H_2) # extract diagonal elements 
    h_2 = h_2.reshape((M,1))
    first_order_derivative_gamma = Z_verbal.transpose() @ (Y_i_verbal + 1/2*h_2 - mu_i) # shape: (2,1)
    Jacobian = np.concatenate((first_order_derivative_beta, first_order_derivative_gamma), axis=0) # shape: (313, 1)
    return -Jacobian.reshape((P+R, ))

"""
# initialization for beta and gamma => mu_X and mu_Z
beta_i = np.log(np.sum(Y_verbal)/X.shape[0])
beta = np.full(shape=(X.shape[1], ), fill_value=beta_i) # shape: (425,)
gamma = np.array([0, 0]).reshape((Z_verbal.shape[1], )) # assume no study-wise regressor
x0 = np.concatenate((beta, gamma), axis=0) # shape:(427, )

#mu_X = np.exp(X @ beta)
#mu_Z = np.exp(Z_verbal @ gamma)
#I = Fisher_Information(mu_X, mu_Z, X, Z_verbal)
#print(I.shape)
#exit()
# minimization of negative log-likelihood with BFGS
minimizer = minimize(nll, x0, args=(X, Z_verbal), method='BFGS', jac=jac, tol=100)
np.savetxt('minimizer_x.txt', minimizer.x)
"""

x = np.loadtxt('minimizer_x.txt')
P, R = X.shape[1], Z_verbal.shape[1]
beta = x[:P].reshape((P,1)) # shape: (425, 1)
gamma = x[P:].reshape((R,1)) # shape: (2,1)

mu_X = np.exp(X @ beta)
mu_Z = np.exp(Z_verbal @ gamma)
mu_g = np.sum(mu_Z) * mu_X # max: 0.12666795965265917; mean: 0.006456670009214881; sum: 1475.0456336151672
mu_i = np.sum(mu_X) * mu_Z # min: 16.277281891965814; max: 121.46846116484669; mean: 14.46123170210948; sum: 1475.045633615167
                            # Y_i_verbal: min: 1; max: 39; mean: 12.607843137254902; sum: 1235

# convert the predicted responses into nifti image 
outside_brain = np.loadtxt('preprocessed_data/outside_brain.txt') # shape: (264027,)
image_dim = np.array([72, 90, 76])
x_min, y_min, z_min = 9, 10, 1
brain_voxel_index = np.setdiff1d(np.arange(np.prod(image_dim)), outside_brain) # shape: (228453, )

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
image.to_filename('verbal_output_image_BGFS.nii.gz')  # Save as NiBabel file