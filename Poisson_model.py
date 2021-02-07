import numpy as np
import pandas as pd
import scipy 
from scipy.special import factorial
from scipy.sparse import load_npz, csr_matrix, csc_matrix, hstack, vstack
from scipy.sparse.linalg import inv
from scipy.optimize import minimize
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
import nibabel as nib

# load foci and counts from npz files 
X_sparse = load_npz("X.npz")
y_sparse = load_npz("y.npz").transpose()  # shape: (292019, 1); max: 5; sum: 15131

X = X_sparse.toarray() # shape: (292019, 443)
y = y_sparse.toarray()

X = np.float32(X)
y = np.int32(y)

"""
# IRLS for Poisson regression
# Problem: The step size can get extremely large easily if there is no regularization term
beta_i = np.log(np.sum(y_sparse)/X.shape[0])
beta = np.full(shape=(X.shape[1],), fill_value=beta_i).reshape((X.shape[1],1)) # shape: (443,1)
# sum_i (log(y_i!))
#sum_logy_factorial = sum(log(factorial(y)))
sum_logy_factorial = np.sum(np.log(factorial(y)))
diff = np.inf # initializations for the loop
l_beta_prev = -np.inf
count = 0

for i in range(50):
    g_mu = np.matmul(X, beta)
    mu = np.exp(g_mu) # mu: mean vector (log link)
    mu_sqrt = np.sqrt(mu)
    # compute the log-likelihood in the current iteration
    # l(beta) = sum(y_i*log(mu_i)-mu_i-log(y_i!))
    log_mu = g_mu
    y_log_mu = y_sparse.multiply(log_mu)
    l_beta = np.sum(y_log_mu) - np.sum(mu) - sum_logy_factorial
    diff = l_beta - l_beta_prev
    print(l_beta, diff, count)
    # compute the update
    X_star = X_sparse.multiply(mu_sqrt)# X* = W^(1/2) X
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    print(np.linalg.cond(XWX.toarray()), np.min(g_mu), np.max(g_mu))
    XWX_inverse = inv(csc_matrix(XWX)) 
    update = XWX_inverse @ X.T @ (y-mu) #(XWX)^(-1) X^T (y-mu)
    print('2 norm of update', np.linalg.norm(update))
    beta += update
    count += 1
    l_beta_prev = l_beta
    print('-----------------')
# takes 15 iterations
np.savetxt('beta.txt', beta, fmt='%f')
l_poisson = l_beta # -54084.26918761416
print(l_poisson)
exit()
"""

# Firth regression for log-linear model
def loglikelihood_Poisson(y, mu):
    #sum_logy_factorial = sum(log(factorial(y)))
    sum_logy_factorial = np.sum(np.log(factorial(y)))
    # l*(beta) = l(beta) + 1/2 log(|I(theta)|)
    #          = sum(y_i*log(mu_i)-mu_i-log(y_i!)) + 1/2 * log(det(XWX))
    log_mu = np.log(mu)
    y_log_mu = np.multiply(y, log_mu)
    l = np.sum(y_log_mu) - np.sum(mu) - sum_logy_factorial
    # compute the penalized term
    # l* = l + 1/2 log(det(X^T WX))
    mu_sqrt = np.sqrt(mu)
    X_star = csr_matrix(X).multiply(mu_sqrt)# X* = W^(1/2) X
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    XWX_eigens = np.real(eigvals(XWX.todense()))
    log_XWX_eigens = np.log(XWX_eigens)
    log_det_XWX = np.sum(log_XWX_eigens)
    l_fr = l + 1/2 * log_det_XWX
    return l, l_fr
    
beta_i = np.log(np.sum(y_sparse)/X.shape[0])
beta = np.full(shape=(X.shape[1],), fill_value=beta_i).reshape((X.shape[1],1)) # shape: (443,1)

log_mu = np.matmul(X, beta)
mu = np.exp(log_mu)

diff = np.inf # initializations for the loop
l_fr_prev = -np.inf
count = 0

beta_2norm_list = [np.linalg.norm(beta)]

l_0, l_fr_0 = loglikelihood_Poisson(y, mu)
l_list = [l_0]
l_fr_list = [l_fr_0]
print(l_0, l_fr_0)

while diff > 1e-6:
    g_mu = np.matmul(X, beta)
    mu = np.exp(g_mu) # mu: mean vector (log link)
    mu_sqrt = np.sqrt(mu)
    print('min and mean and max of mu', np.min(mu), np.mean(mu), np.max(mu), np.mean(mu)/np.min(mu), np.max(mu)/np.mean(mu))
    # compute the update
    X_star = X_sparse.multiply(mu_sqrt)# X* = W^(1/2) X
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    XWX_array = XWX.toarray() #type: numpy array
    XWX_eigens = np.real(eigvals(XWX.todense()))
    print('eigen value is in the range of', np.min(XWX_eigens), np.max(XWX_eigens), np.prod(XWX_eigens))
    XWX_inverse = inv(csc_matrix(XWX)) 
    # compute the update
    WX = X_sparse.multiply(mu) # shape: (292019, 443)
    hadamard_product = (WX @ XWX_inverse).multiply(X) # diag(AB^T) = (A @ B) 1
    h = np.sum(hadamard_product, axis=1) # h = (H_11,H_22,...,H_nn)^T, H = WX(X^T WX)^(-1) X^T
    update = XWX_inverse @ np.transpose(X) @ (y + 1/2*h - mu)
    beta = beta + update
    beta_2norm = np.linalg.norm(beta)
    beta_2norm_list.append(beta_2norm)
    print('2 norm of beta is', beta_2norm)
    # compute (penalized) log-likelihood and save to list
    updated_log_mu = np.matmul(X, beta)
    updated_mu = np.exp(updated_log_mu)
    l, l_fr = loglikelihood_Poisson(y, updated_mu)
    l_list.append(l)
    l_fr_list.append(l_fr)
    diff = l_fr - l_fr_prev
    print('log-likelihood in the current iteration is', l, l_fr)
    print('difference of log-likelihood is', diff)
    count += 1
    l_fr_prev = l_fr
    print('-----------------')
# takes 42 iterations
print(count)
print('2 norm of beta list:', beta_2norm_list)
print("log-likelihood list", l_list)
print("penalized log-likelihood list", l_fr_list)
np.savetxt('beta_Poisson.txt', beta, fmt='%f')
np.savetxt('beta_2norm_Poisson.txt', np.array(beta_2norm_list), fmt='%f')
np.savetxt('l_Poisson.txt', np.array(l_list), fmt='%f')
np.savetxt('l_fr_Poisson.txt', np.array(l_fr_list), fmt='%f')
exit()


# load the convergent beta result
beta_poisson_fr = np.loadtxt('Poisson_FR_result_1e-6/beta_Poisson.txt')
log_mu_poisson_fr = X @ beta_poisson_fr
y_pred_poisson_fr = np.exp(log_mu_poisson_fr) # sum: 15352.529954222287; max: 0.5190707002550571; mean: 0.052573736483661294 ~ 15131/292019 
print(np.min(y_pred_poisson_fr), np.mean(y_pred_poisson_fr), np.max(y_pred_poisson_fr), np.sum(y_pred_poisson_fr))

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
    response = y_pred_poisson_fr[i]
    output_image[x_coord, y_coord, z_coord] = response

image = nib.Nifti1Image(output_image, np.eye(4))
image.to_filename('output_image_poisson_fr.nii.gz')  # Save as NiBabel file


beta_2norm_Poisson_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/Poisson_FR_result_1e-6/beta_2norm_Poisson.txt')
l_Poisson_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/Poisson_FR_result_1e-6/l_Poisson.txt')
l_fr_Poisson_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/Poisson_FR_result_1e-6/l_fr_Poisson.txt')
print(len(beta_2norm_Poisson_fr), len(l_Poisson_fr), len(l_fr_Poisson_fr))

x = np.arange(10, len(beta_2norm_Poisson_fr)) # x-axis: from 5 to 131

fig, axs = plt.subplots(2, 2)
fig.suptitle('Convergence plot in Log-linear model (tol=10^-6)')

axs[0, 0].plot(x, l_Poisson_fr[10:], 'tab:green')
axs[0, 0].set_title('Log-likelihood')
axs[0, 1].plot(x, l_fr_Poisson_fr[10:], 'tab:red')
axs[0, 1].set_title('Penalized log-likelihood')
axs[1, 0].plot(x, beta_2norm_Poisson_fr[10:], 'tab:orange')
axs[1, 0].set_title('L2-norm of beta')

for ax in axs.flat:
    ax.set(xlabel='Number of iteration')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

#plt.show()
fig.savefig('/Users/yifan/Documents/GitHub/Meta_regression/Poisson_FR_result_1e-6/convergence_plot.png')

l_Poisson_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/Poisson_FR_result_1e-6/l_Poisson.txt')
# compute measurements of goodness-of-fit
n,k = X.shape # n = number of data points; k = number of estimated parameters
log_L = l_Poisson_fr[-1]
AIC = 2 * k - 2 * log_L # 109134.51417
BIC = k * np.log(n) - 2 * log_L # 113823.48051739862

# compute Pearson residual
beta_poisson_fr = np.loadtxt('Poisson_FR_result_1e-6/beta_Poisson.txt')
log_mu_poisson_fr = X @ beta_poisson_fr
y_pred_poisson_fr = np.exp(log_mu_poisson_fr) # sum: 15352.529954222287; max: 0.5190707002550571; mean: 0.052573736483661294 ~ 15131/292019 
r_pearson = (y - y_pred_poisson_fr)/np.sqrt(y_pred_poisson_fr)
print(r_pearson.shape)

