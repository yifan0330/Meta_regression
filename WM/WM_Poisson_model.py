import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
from scipy.special import factorial
from scipy.sparse import load_npz, csr_matrix, csc_matrix, hstack, vstack
from scipy.sparse.linalg import inv, cg
from scipy.linalg import eigvals
from scipy.optimize import minimize, fsolve, root, brute
import patsy
import statsmodels.api as sm
import nibabel as nib
from datetime import datetime

# load foci and counts from npz files 
X_sparse = load_npz("X.npz") 
y = load_npz("y.npz").transpose()
y_verbal_sparse = load_npz("y_verbal.npz").transpose()
y_nonverbal_sparse = load_npz("y_non_verbal.npz").transpose()

X = X_sparse.toarray() # shape: (292019, 443/365/311/277)/ (194369, 562)
y = y.toarray() # sum = 2223; max = 3; nonzero counts = 2125
y_verbal = y_verbal_sparse.toarray() # sum = 1286; max = 3; nonzero counts = 1244
                                    # sum = 1158; max = 3; nonzero counts = 1121
y_nonverbal = y_nonverbal_sparse.toarray() # sum = 937; max = 2; nonzero counts = 923

print(np.sum(y), np.sum(y_verbal), np.sum(y_nonverbal))


exit()
# Firth regression for log-linear model
now = datetime.now()
print(now)
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
    #print(np.min(XWX_eigens), np.max(XWX_eigens))
    log_XWX_eigens = np.log(XWX_eigens)
    log_det_XWX = np.sum(log_XWX_eigens)
    l_fr = l + 1/2 * log_det_XWX
    return l, l_fr


beta_i = -5.425282242829142 #np.log(np.sum(y)/X.shape[0]) # -5.425282242829142 / -5.123064035139061
beta = np.full(shape=(X.shape[1],1), fill_value=beta_i) # shape: (365,1) / (562, 1)

log_mu = np.matmul(X, beta)
mu = np.exp(log_mu)

diff = np.inf # initializations for the loop
l_fr_prev = -np.inf
count = 0

beta_2norm_list = [np.linalg.norm(beta)]

l_0, l_fr_0 = loglikelihood_Poisson(y, mu)
l_list = [l_0]
l_fr_list = [l_fr_0]

print(np.sum(np.log(factorial(y))))
print(np.sum(np.log(factorial(y_verbal))))
print(np.sum(np.log(factorial(y_nonverbal))))

exit()
while diff > 1e-2:
    g_mu = np.matmul(X, beta)
    mu = np.exp(g_mu) # mu: mean vector (log link)
    mu_sqrt = np.sqrt(mu)
    # compute the update
    X_star = X_sparse.multiply(mu_sqrt)# X* = W^(1/2) X
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    XWX_inverse = inv(csc_matrix(XWX)) 
    # compute the update
    WX = X_sparse.multiply(mu) # shape: (292019, 365)
    hadamard_product = (WX @ XWX_inverse).multiply(X) # diag(AB^T) = (A @ B) 1
    h = np.sum(hadamard_product, axis=1) # h = (H_11,H_22,...,H_nn)^T, H = WX(X^T WX)^(-1) X^T
    first_order_derivarive = X.transpose() @ (y + 1/2*h -mu)
    # start with gradient descent at the beginning
    if diff > 1:
        step_size = 0.07
        update = step_size * first_order_derivarive # shape: (365,1)
        print('2 norm of update', np.linalg.norm(update))
        beta = beta + update
    # switch back to Newton's method
    else: 
        print('switch back to Newton method')
        second_order_derivarive = - XWX 
        cg_solution = cg(second_order_derivarive, b=first_order_derivarive)[0] # shape: (365,)
        update = cg_solution.reshape((X.shape[1],1))
        beta = beta - update
    beta_2norm = np.linalg.norm(beta)
    beta_2norm_list.append(beta_2norm)
    print('2 norm of beta is', beta_2norm)
    # compute (penalized) log-likelihood and save to list
    updated_log_mu = np.matmul(X, beta)
    updated_mu = np.exp(updated_log_mu)
    print(updated_mu.shape)
    l, l_fr = loglikelihood_Poisson(y, updated_mu)
    l_list.append(l)
    l_fr_list.append(l_fr)
    diff = l_fr - l_fr_prev
    print('log-likelihood in the current iteration is', l, l_fr)
    print('difference of log-likelihood is', diff)
    count += 1
    l_fr_prev = l_fr
    print(count)
    print('------------------------')
# takes 61 iterations for gradient descent
print(count)
print('2 norm of beta list:', beta_2norm_list)
print("log-likelihood list", l_list)
print("penalized log-likelihood list", l_fr_list)
np.savetxt('beta_Poisson.txt', beta, fmt='%f')
np.savetxt('beta_2norm_Poisson.txt', np.array(beta_2norm_list), fmt='%f')
np.savetxt('l_Poisson.txt', np.array(l_list), fmt='%f')
np.savetxt('l_fr_Poisson.txt', np.array(l_fr_list), fmt='%f')
now = datetime.now()
print(now)
exit()


# load the convergent beta result
beta_poisson_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/WM/311_basis/nonverbal_stimuli_Poisson/nonverbal_beta_Poisson.txt')
log_mu_poisson_fr = X @ beta_poisson_fr
y_pred_poisson_fr = np.exp(log_mu_poisson_fr) # verbal(365): sum: 1470.5259707385887; max: 0.18864370051745882; mean: 0.0050357201782712385
                                            # non-verbal(365): sum: 1121.0836906286936; max: 0.10618059468741041; mean: 0.003839077904618171
                                            # verbal(311): sum: 1443.3986676337433; max: 0.13415673569698255; mean: 0.004942824499891252
                                            # non-verbal(311): sum: 1094.1353494726702; max: 0.07157430830104584; mean: 0.003746795069747757
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
image.to_filename('/Users/yifan/Documents/GitHub/Meta_regression/WM/311_basis/nonverbal_stimuli_Poisson/nonverbal_output_image_poisson_fr.nii.gz')  # Save as NiBabel file
exit()


beta_2norm_Poisson_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/WM/verbal_stimuli_Poisson/verbal_beta_2norm_Poisson.txt')
l_Poisson_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/WM/verbal_stimuli_Poisson/verbal_l_Poisson.txt')
l_fr_Poisson_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/WM/verbal_stimuli_Poisson/verbal_l_fr_Poisson.txt')
print(len(beta_2norm_Poisson_fr), len(l_Poisson_fr), len(l_fr_Poisson_fr))

x = np.arange(5, len(beta_2norm_Poisson_fr)) # x-axis: from 5 to 131

fig, axs = plt.subplots(2, 2)
fig.suptitle('Convergence plot in Log-linear model (tol=10^-6)')

axs[0, 0].plot(x, l_Poisson_fr[5:], 'tab:green')
axs[0, 0].set_title('Log-likelihood')
axs[0, 1].plot(x, l_fr_Poisson_fr[5:], 'tab:red')
axs[0, 1].set_title('Penalized log-likelihood')
axs[1, 0].plot(x, beta_2norm_Poisson_fr[5:], 'tab:orange')
axs[1, 0].set_title('L2-norm of beta')

for ax in axs.flat:
    ax.set(xlabel='Number of iteration')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

#plt.show()
fig.savefig('/Users/yifan/Documents/GitHub/Meta_regression/WM/verbal_stimuli_Poisson/convergence_plot_skip5.png')
exit()
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