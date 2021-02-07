import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
from scipy.special import factorial
from scipy.sparse import load_npz, csr_matrix, csc_matrix, hstack, vstack
from scipy.sparse.linalg import inv
from scipy.linalg import eigvals
from scipy.optimize import minimize, fsolve, root, brute
import patsy
import statsmodels.api as sm
import nibabel as nib

# load foci and counts from npz files 
X_sparse = load_npz("X.npz")
y_sparse = load_npz("y.npz").transpose()  # shape: (292019, 1); max: 5; sum: 15131

X = X_sparse.toarray() # shape: (292019, 443)
y = y_sparse.toarray() # sum = 15131

""""
# Firstly, leave alpha fixed at zero (essentially Poisson model)
# and let beta get estimated
# and the stopping criterion is set as: Penalized log-likelihood changes fall below 10^-2
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

# initialization for beta
beta_i = np.log(np.sum(y_sparse)/X.shape[0])
beta = np.full(shape=(X.shape[1],), fill_value=beta_i).reshape((X.shape[1],1)) # shape: (443,1)

diff = np.inf # initializations for the loop
l_fr_prev = -np.inf
count = 0

while diff > 1e-2:
    g_mu = np.matmul(X, beta)
    mu = np.exp(g_mu) # mu: mean vector (log link)
    mu_sqrt = np.sqrt(mu)
    # compute the update
    X_star = X_sparse.multiply(mu_sqrt)# X* = W^(1/2) X
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    XWX_array = XWX.toarray() #type: numpy array
    XWX_inverse = inv(csc_matrix(XWX)) 
    # compute the update
    WX = X_sparse.multiply(mu) # shape: (292019, 443)
    hadamard_product = (WX @ XWX_inverse).multiply(X) # diag(AB^T) = (A @ B) 1
    h = np.sum(hadamard_product, axis=1) # h = (H_11,H_22,...,H_nn)^T, H = WX(X^T WX)^(-1) X^T
    update = XWX_inverse @ np.transpose(X) @ (y + 1/2*h - mu)
    beta = beta + update
    beta_2norm = np.linalg.norm(beta)
    print('2 norm of beta is', beta_2norm)
    # compute (penalized) log-likelihood and save to list
    l, l_fr = loglikelihood_Poisson(y, mu)
    diff = l_fr - l_fr_prev
    print('log-likelihood in the current iteration is', l, l_fr)
    print('difference of log-likelihood is', diff)
    count += 1
    l_fr_prev = l_fr
    print('-----------------')
# takes 26 iterations （～30 min）
print(count)
np.savetxt('optimized_beta.txt', beta)
"""

"""
# identify optimized beta and alpha as the starting points
def negative_loglikelihood_NB(alpha, *args):
    X, y, mu = args[0], args[1], args[2]
    y_max = np.max(y)
    first_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(np.log(array + 1/alpha))
        first_term += summation * count
    expression =  - np.log(scipy.special.gamma(y+1)) + np.log(alpha*mu) * y - np.multiply(y+1/alpha, np.log(1+alpha*mu))
    l = first_term + np.sum(expression)
    # compute the penalized term
    # l* = l + 1/2 log(det(X^T WX))
    mu_sqrt = np.sqrt(mu)
    X_star = csr_matrix(X).multiply(mu_sqrt)# X* = W^(1/2) X
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    XWX_eigens = np.real(eigvals(XWX.todense()))
    log_XWX_eigens = np.log(XWX_eigens)
    log_det_XWX = np.sum(log_XWX_eigens)
    l_fr = l + 1/2 * log_det_XWX
    print(alpha, l_fr)
    return - l_fr

optimized_beta = np.loadtxt('optimized_beta.txt').reshape((443,1)) # shape: (443,1)
optimized_beta = np.float32(optimized_beta)
optimized_log_mu = np.matmul(X, optimized_beta)
optimized_mu = np.exp(optimized_log_mu) # shape: (292019,1)

# find the optimal initial value alpha_0 using low-resolution grid search 
#machine_epsilon = np.finfo(float).eps # use machine epilson instead of zero
alpha_range = [slice(0, 5, 0.2)]
alpha_brute = brute(negative_loglikelihood_NB, alpha_range, args=(X, y, optimized_mu))
alpha_0 = alpha_brute[0] # 0.7129375
"""


# IRLS for Negative Binomial model (coordinate ascent: just beta & just alpha)
"""
# log-likelihood in Negative Binomial model
def loglikelihood_NB(log_alpha, y, mu):
    alpha = np.exp(log_alpha)
    y_max = np.max(y)
    first_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(np.log(array + 1/alpha))
        first_term += summation * count
    alpha_mu = np.multiply(alpha, mu)
    expression =  - np.log(scipy.special.gamma(y+1)) + np.multiply(np.log(alpha_mu), y) - np.multiply(y+1/alpha, np.log(1+alpha_mu))
    l = first_term + np.sum(expression)
    return l

def loglikelihood_NB_fr(log_alpha, y, mu, X):
    l = loglikelihood_NB(log_alpha, y, mu)
    # compute the penalized item
    w_numerator = np.multiply(1+alpha*y, mu)
    w_denominator = np.multiply(1 + alpha * mu, 1 + alpha * mu)
    w = np.multiply(w_numerator, 1/w_denominator)
    w_sqrt = np.sqrt(w)
    X_star = csr_matrix(X).multiply(w_sqrt)# X* = W^(1/2) X; shape: (292019, 443)
    XWX = X_star.T @ X_star
    XWX_eigens = np.real(eigvals(XWX.todense()))
    log_XWX_eigens = np.log(XWX_eigens)
    log_det_XWX = np.sum(log_XWX_eigens)
    l_fr = l + 1/2 * log_det_XWX
    return l_fr

# first-order partial derivative: d l(beta, alpha)/d alpha
def first_order_derivative_alpha_fr(log_alpha, y, mu, h):
    alpha = np.exp(log_alpha)
    y_max = np.max(y)
    add_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(1/(array + 1/alpha))
        add_term += -1/alpha**2 * summation * count
    alpha_mu = alpha*mu # shape: (292019, 1)
    expression = 1/alpha**2 * np.log(1+alpha_mu) + 1/alpha * np.multiply(y - mu, 1/(1+alpha_mu))
    # firth regression term in first order derivative of alpha
    # 1/2 * sum[(y_i-alpha*y_i*mu_i-2*mu_i)/(1+alpha*mu_i)*(1+alpha*y_i) H_ii]
    #fr_term_numerator = y - alpha*np.multiply(y,mu) - 2*mu
    #fr_term_denominator = np.multiply(1+alpha_mu, 1+alpha*y)
    #fr_term_frac = np.multiply(fr_term_numerator, 1/fr_term_denominator)
    #fr_term_alpha = 1/2 * np.sum(np.multiply(fr_term_frac, h))
    dl_d_alpha = add_term + np.sum(expression) #+ fr_term_alpha
    #print(add_term+np.sum(expression), fr_term_alpha, dl_d_alpha)
    #print('==================')
    return dl_d_alpha


# second-order partial derivative: d^2 l(beta, alpha)/d alpha^2
def second_order_derivative_alpha(log_alpha, y, mu, h):
    alpha = np.exp(log_alpha)
    y_max = np.max(y)
    add_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(np.multiply(array/(1+alpha*array), array/(1+alpha*array)))
        add_term += -1 * summation * count
    alpha_mu = np.multiply(alpha, mu)
    expression_last_numerator = np.multiply(y+1/alpha, np.multiply(mu,mu))
    expression_last_denominator = np.multiply(1+alpha_mu, 1+alpha_mu)
    expression = -2/alpha**3 * np.log(1+alpha_mu) + 2/alpha**2 * np.multiply(mu, 1/(1+alpha_mu)) + np.multiply(expression_last_numerator, 1/expression_last_denominator)
    # firth regression term in second order derivative of alpha
    # 1/2 * sum[(alpha^2*mu^2*y^2 - y^2 + 2*mu^2 - 2*alpha*mu*y^2 + 4*alpha*mu^2*y) / (1+alpha*mu)^2 * (1+alpha*y)^2 H_ii]
    #mu_times_y = np.multiply(mu, y)
    #fr_term_numerator = alpha**2 * np.multiply(mu_times_y,mu_times_y) - np.multiply(y,y) + 2*np.multiply(mu,mu) - 2*alpha*np.multiply(mu_times_y, y) + 4*alpha*np.multiply(mu_times_y, mu)
    #fr_term_denominator = np.multiply(np.multiply(1+alpha_mu, 1+alpha_mu), np.multiply(1+alpha*y, 1+alpha*y))
    #fr_term_frac = np.multiply(fr_term_numerator, 1/fr_term_denominator)
    #fr_term_alpha = 1/2 * np.sum(np.multiply(fr_term_frac, h))
    dl2_d_alpha2 = add_term + np.sum(expression) #+ fr_term_alpha
    #print(add_term+np.sum(expression), fr_term_alpha, dl2_d_alpha2)
    #print('==================')
    return dl2_d_alpha2

optimized_beta = np.loadtxt('optimized_beta.txt').reshape((443,1)) # shape: (443,1)
optimized_beta = np.float32(optimized_beta)
optimized_log_mu = np.matmul(X, optimized_beta)
optimized_mu = np.exp(optimized_log_mu) # shape: (292019,1)

# initialization for beta and alpha
beta = optimized_beta
alpha = 0.8 #0.7129375 # results from Poisson model
log_alpha = np.log(alpha) # -0.3383176882953604

# initialization for l_prev and l_diff
l_fr_prev = - np.inf
l_fr_diff = np.inf

count = 0
beta_2norm_list = [np.linalg.norm(beta)]
alpha_list = [np.exp(log_alpha)]
# compute (penalized) log-likelihood in the 0th iteration (for initialized value)
log_mu = np.matmul(X, beta)
mu = np.exp(log_mu)

l_list = [loglikelihood_NB(log_alpha, y, mu)]
l_fr_list = [loglikelihood_NB_fr(log_alpha, y, mu, X)]

while l_fr_diff >= 1e-6:
    alpha = np.exp(log_alpha)
    log_mu = np.matmul(X, beta) # log_mu = beta_0 + beta_1 x1 + ... + beta_p x_p
    mu = np.exp(log_mu) # shape: (292019, 1)
    # w_i = (1+alpha y_i)*mu_i/(1+alpha mu_i)^2
    w_numerator = np.multiply(1+alpha*y, mu)
    w_denominator = np.multiply(1 + alpha * mu, 1 + alpha * mu)
    w = np.multiply(w_numerator, 1/w_denominator)
    # compute diagonal elements on H = WX (X^TWX)^(-1) X^T
    w_sqrt = np.sqrt(w)
    X_star = X_sparse.multiply(w_sqrt)# X* = W^(1/2) X; shape: (292019, 443)
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    # compute H = WX(X^T WX)^(-1) X^T
    XWX_inverse = inv(csc_matrix(XWX))
    WX = X_sparse.multiply(w) # shape: (292019, 443)
    hadamard_product = (WX @ XWX_inverse).multiply(X) # diag(AB^T) = (A @ B) 1; @: hadamard product / elementwise product
    h = np.sum(hadamard_product, axis=1) # h = (H_11,H_22,...,H_nn)^T
    # first order derivative w.r.t beta (firth regression)
    s = np.multiply(y - mu, 1/(1+alpha*mu)) # shape: (292019, 1)
    fr_term_beta = 1/2 * np.multiply(h, (1-alpha*mu)/(1+alpha*mu))
    beta_first_derivative = X.T @ (s + fr_term_beta) #dl(alpha, beta)/d beta = X^T (s + firth_regression_term); shape: (443, 1)
    # second order derivative w.r.t beta
    # d^2 l/d beta^2 = - X^T V X
    v_numerator = np.multiply(1+alpha*h+alpha*y, mu)
    v_denominator = np.multiply(1+alpha*mu, 1+alpha*mu)
    v = np.multiply(v_numerator, 1/v_denominator) # shape: (292019, 1)
    v_sqrt = np.sqrt(v)
    X_star2 = X_sparse.multiply(v_sqrt)# X* = V^(1/2) X
    XVX = X_star2.T @ X_star2 # XVX = (V^(1/2) X)^T (V^(1/2) X)
    beta_second_derivative = - XVX # shape: (443, 443)
    print("condition number is", np.linalg.cond(beta_second_derivative.todense()))
    beta_Hessian_inverse = inv(csc_matrix(beta_second_derivative)) # shape: (443,443)
    beta_update =  beta_Hessian_inverse @ beta_first_derivative
    beta = beta - beta_update
    # save the 2 norm of beta in each iteration
    beta_2norm = np.linalg.norm(beta)
    beta_2norm_list.append(beta_2norm)
    # then maximize log-likelihood w.r.t alpha with updated beta
    updated_log_mu = np.matmul(X, beta)
    updated_mu = np.exp(updated_log_mu)
    log_alpha_first_derivative = alpha * first_order_derivative_alpha_fr(log_alpha, y, updated_mu, h)
    log_alpha_second_derivative = alpha**2 * second_order_derivative_alpha(log_alpha, y, updated_mu, h) + 2 * log_alpha_first_derivative
    log_alpha_update = 1/log_alpha_second_derivative * log_alpha_first_derivative
    log_alpha = log_alpha - log_alpha_update
    count += 1
    # save the value of alpha in each iteration
    alpha_list.append(np.exp(log_alpha))
    # compute the log-likelihood
    l = loglikelihood_NB(log_alpha, y, updated_mu)
    l_fr = loglikelihood_NB_fr(log_alpha, y, updated_mu, X)
    # difference in log-likelihood
    l_fr_diff = l_fr - l_fr_prev
    print('difference in log-likelihood is', l_fr_diff)
    l_list.append(l)
    l_fr_list.append(l_fr)
    l_fr_prev = l_fr
    print('Log-likelihood and firth log-likelihood of firth regression in the current iteration is', l, l_fr)
    print("alpha is", np.exp(log_alpha))
    #print(alpha_list, beta_2norm_list, l_list, l_fr_list)
    print('--------------------The ' + str(count) + "th iteration is done")

# take  iterations (~ ) to limit the difference in log-likelihood below 1e-6
print(count)
print("The list of alpha values is", alpha_list)
print("The list of 2 norm of beta is", beta_2norm_list)
print("The list of log-likelihood is", l_list)
print("The list of firth-log-likelihood is", l_fr_list)
print('----------------------------------------------')
np.savetxt('adjusted_final_beta_NB_fr.txt', beta)
np.savetxt('adjusted_alpha_NB_fr.txt', np.array(alpha_list))
np.savetxt('adjusted_beta_2norm_NB_fr.txt', np.array(beta_2norm_list))
np.savetxt('adjusted_l_NB_fr.txt', np.array(l_list))
np.savetxt('adjusted_l_fr_NB_fr.txt', np.array(l_fr_list))
"""
"""
# load the convergent beta result
beta_NB_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_result_1e-6/adjusted_final_beta_NB_fr.txt')
log_mu_NB_fr = X @ beta_NB_fr
y_pred_NB_fr = np.exp(log_mu_NB_fr) # 1e-6: sum: 15332.912167062772; max: 0.4879301389032836; mean: 0.05250655665235061
print(np.min(y_pred_NB_fr), np.mean(y_pred_NB_fr), np.max(y_pred_NB_fr), np.sum(y_pred_NB_fr))


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
    response = y_pred_NB_fr[i]
    output_image[x_coord, y_coord, z_coord] = response

image = nib.Nifti1Image(output_image, np.eye(4))
image.to_filename('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_result_1e-6/output_image_adjusted_NB_fr.nii.gz')
"""
"""
alpha_adjusted_NB_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_result_1e-6/adjusted_alpha_NB_fr.txt')
beta_2norm_adjusted_NB_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_result_1e-6/adjusted_beta_2norm_NB_fr.txt')
l_adjusted_NB_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_result_1e-6/adjusted_l_NB_fr.txt')
l_fr_adjusted_NB_fr = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_result_1e-6/adjusted_l_fr_NB_fr.txt')
print(len(alpha_adjusted_NB_fr), len(beta_2norm_adjusted_NB_fr), len(l_adjusted_NB_fr), len(l_fr_adjusted_NB_fr)) # 89


x = np.arange(10, len(alpha_adjusted_NB_fr)) # x-axis: from 0 to 45

fig, axs = plt.subplots(2, 2)
fig.suptitle('Convergence plot in Negative Binomial model (tol=10^-6)')
axs[0, 0].plot(x, alpha_adjusted_NB_fr[10:])
axs[0, 0].set_title('alpha')
axs[0, 1].plot(x, beta_2norm_adjusted_NB_fr[10:], 'tab:orange')
axs[0, 1].set_title('L2-norm of beta')
axs[1, 0].plot(x, l_adjusted_NB_fr[10:], 'tab:green')
axs[1, 0].set_title('Log-likelihood')
axs[1, 1].plot(x, l_fr_adjusted_NB_fr[10:], 'tab:red')
axs[1, 1].set_title('Penalized log-likelihood')

for ax in axs.flat:
    ax.set(xlabel='Number of iteration')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

#plt.show()
fig.savefig('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_result_1e-6/convergence_plot_skip10.png')
exit()
"""

# IRLS for Negative Binomial model (moment estimation)
"""
# log-likelihood in Negative Binomial model
def loglikelihood_NB(alpha, y, mu):
    y_max = np.max(y)
    first_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(np.log(array + 1/alpha))
        first_term += summation * count
    alpha_mu = np.multiply(alpha, mu)
    expression =  - np.log(scipy.special.gamma(y+1)) + np.multiply(np.log(alpha_mu), y) - np.multiply(y+1/alpha, np.log(1+alpha_mu))
    l = first_term + np.sum(expression)
    return l

def loglikelihood_NB_fr(alpha, y, mu, X):
    l = loglikelihood_NB(alpha, y, mu)
    # compute the penalized item
    w_numerator = np.multiply(1+alpha*y, mu)
    w_denominator = np.multiply(1 + alpha * mu, 1 + alpha * mu)
    w = np.multiply(w_numerator, 1/w_denominator)
    w_sqrt = np.sqrt(w)
    X_star = csr_matrix(X).multiply(w_sqrt)# X* = W^(1/2) X; shape: (292019, 443)
    XWX = X_star.T @ X_star
    XWX_eigens = np.real(eigvals(XWX.todense()))
    log_XWX_eigens = np.log(XWX_eigens)
    log_det_XWX = np.sum(log_XWX_eigens)
    l_fr = l + 1/2 * log_det_XWX
    return l_fr

def alpha_moment_estimation(alpha, y, mu):
    #alpha = alpha.item()
    pearson_residual_numerator = np.multiply(y-mu, y-mu)
    pearson_residual_denominator = np.multiply(mu, 1+alpha*mu)
    pearson_residual = np.multiply(pearson_residual_numerator, 1/pearson_residual_denominator)
    sum_pearson_square = np.sum(pearson_residual)
    return sum_pearson_square - (n-p)

n,p = X.shape # n=292019; p=443
# initialization for beta and alpha
optimized_beta = np.loadtxt('optimized_beta.txt').reshape((443,1)) # shape: (443,1)
optimized_log_mu = np.matmul(X, optimized_beta)
optimized_mu = np.exp(optimized_log_mu) # shape: (292019,1)
beta = optimized_beta
alpha = 0.8 #0.7129375

l_fr_diff = np.inf
l_fr_prev = -np.inf
count = 0

beta_2norm_list = [np.linalg.norm(beta)]
alpha_list = [alpha]


l_list = [loglikelihood_NB(alpha, y, optimized_mu)]
l_fr_list = [loglikelihood_NB_fr(alpha, y, optimized_mu, X)]

while l_fr_diff >= 1e-6:
    log_mu = np.matmul(X, beta) # log_mu = beta_0 + beta_1 x1 + ... + beta_p x_p
    mu = np.exp(log_mu) # shape: (292019, 1)
    # maximize log-likelihood w.r.t beta with fixed alpha
    # w_i = (1+alpha y_i)*mu_i/(1+alpha mu_i)^2
    w_numerator = np.multiply(1+alpha*y, mu)
    w_denominator = np.multiply(1 + alpha * mu, 1 + alpha * mu)
    w = np.multiply(w_numerator, 1/w_denominator)
    # compute diagonal elements on H = WX (X^TWX)^(-1) X^T
    w_sqrt = np.sqrt(w)
    X_star = X_sparse.multiply(w_sqrt)# X* = W^(1/2) X; shape: (292019, 443)
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    # compute H = WX(X^T WX)^(-1) X^T
    XWX_inverse = inv(csc_matrix(XWX))
    WX = X_sparse.multiply(w) # shape: (292019, 443)
    hadamard_product = (WX @ XWX_inverse).multiply(X) # diag(AB^T) = (A @ B) 1
    h = np.sum(hadamard_product, axis=1) # h = (H_11,H_22,...,H_nn)^T
    # first order derivative w.r.t beta (firth regression)
    s = (y - mu)/(1+alpha*mu) # shape: (292019, 1)
    fr_term = 1/2 * np.multiply(h, (1-alpha*mu)/(1+alpha*mu))
    beta_first_derivative = X.T @ (s + fr_term) #dl(alpha, beta)/d beta = X^T (s + firth_regression_term); shape: (443, 1)
    # second order derivative w.r.t beta
    # d^2 l/d beta^2 = - X^T V X
    v_numerator = np.multiply(1+alpha*h+alpha*y, mu)
    v_denominator = np.multiply(1+alpha*mu, 1+alpha*mu)
    v = np.multiply(v_numerator, 1/v_denominator) # shape: (292019, 1)
    v_sqrt = np.sqrt(v)
    X_star2 = X_sparse.multiply(v_sqrt)# X* = V^(1/2) X
    XVX = X_star2.T @ X_star2 # XVX = (V^(1/2) X)^T (V^(1/2) X)
    beta_second_derivative = - XVX # shape: (443, 443)
    beta_Hessian_inverse = inv(csc_matrix(beta_second_derivative)) # shape: (443,443)
    beta_update =  beta_Hessian_inverse @ beta_first_derivative
    beta = beta - beta_update
    # save the 2 norm of beta in each iteration
    beta_2norm = np.linalg.norm(beta)
    beta_2norm_list.append(beta_2norm)
    # then use moment estimation of alpha
    # Newton's method
    updated_log_mu = np.matmul(X, beta)
    updated_mu = np.exp(updated_log_mu)
    #for i in range(10):
    f_alpha = alpha_moment_estimation(alpha, y, optimized_mu)
    f_alpha_derivative_numerator = np.multiply(y-optimized_mu, y-optimized_mu)
    f_alpha_derivative_denominator = np.multiply(1+alpha*optimized_mu, 1+alpha*optimized_mu)
    f_alpha_derivative = -np.sum(np.multiply(f_alpha_derivative_numerator, 1/f_alpha_derivative_denominator))
    updated_alpha = alpha - f_alpha/f_alpha_derivative
    #updated_alpha = root(fun = alpha_moment_estimation, x0=alpha, args=(y, mu), method='lm')
    #print(updated_alpha)
    #updated_alpha = updated_alpha.x.item()
    print('updated alpha is', updated_alpha)
    alpha = updated_alpha
    alpha_list.append(alpha)
    # compute the updated log-likelihood and penalized log-likelihood
    l = loglikelihood_NB(alpha, y, updated_mu)
    # compute penalized log-likelihood
    l_fr = loglikelihood_NB_fr(alpha, y, updated_mu, X)
    l_list.append(l)
    l_fr_list.append(l_fr)
    print('The (penalized) log-likelihood in the current iteration is', l, l_fr)
    #l_star = l + 1/2 * np.log(np.linalg.det(XWX.toarray()))
    l_fr_diff = l_fr - l_fr_prev
    print(l_fr_diff, alpha, count)
    l_fr_prev = l_fr
    count += 1
    print('--------------------The ' + str(count) + "th iteration is done")

# take  iterations (~  miniutes) to limit the difference in log-likelihood below 1e-2
# take  iterations (~  miniutes) to limit the difference in log-likelihood below 1e-4
print(count)
print("The list of alpha values is", alpha_list)
print("The list of 2 norm of beta is", beta_2norm_list)
print("The list of log-likelihood is", l_list)
print("The list of firth-log-likelihood is", l_fr_list)
print('----------------------------------------------')
np.savetxt('final_beta_NB_fr_ME.txt', beta)
np.savetxt('alpha_NB_fr_ME.txt', np.array(alpha_list))
np.savetxt('beta_2norm_NB_fr_ME.txt', np.array(beta_2norm_list))
np.savetxt('l_NB_fr_ME.txt', np.array(l_list))
np.savetxt('l_fr_NB_fr_ME.txt', np.array(l_fr_list))
exit()
"""
"""
# IRLS for Negative Binomial model (moment estimation)
# log-likelihood in Negative Binomial model
def loglikelihood_NB(log_alpha, y, mu):
    alpha = np.exp(log_alpha)
    y_max = np.max(y)
    first_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(np.log(array + 1/alpha))
        first_term += summation * count
    alpha_mu = np.multiply(alpha, mu)
    expression =  - np.log(scipy.special.gamma(y+1)) + np.multiply(np.log(alpha_mu), y) - np.multiply(y+1/alpha, np.log(1+alpha_mu))
    l = first_term + np.sum(expression)
    return l

def loglikelihood_NB_fr(log_alpha, y, mu, X):
    alpha = np.exp(log_alpha)
    l = loglikelihood_NB(alpha, y, mu)
    # compute the penalized item
    w_numerator = np.multiply(1+alpha*y, mu)
    w_denominator = np.multiply(1 + alpha * mu, 1 + alpha * mu)
    w = np.multiply(w_numerator, 1/w_denominator)
    w_sqrt = np.sqrt(w)
    X_star = csr_matrix(X).multiply(w_sqrt)# X* = W^(1/2) X; shape: (292019, 443)
    XWX = X_star.T @ X_star
    XWX_eigens = np.real(eigvals(XWX.todense()))
    log_XWX_eigens = np.log(XWX_eigens)
    log_det_XWX = np.sum(log_XWX_eigens)
    l_fr = l + 1/2 * log_det_XWX
    return l_fr

def alpha_moment_estimation(log_alpha, y, mu):
    alpha = np.exp(log_alpha)
    pearson_residual_numerator = np.multiply(y-mu, y-mu)
    pearson_residual_denominator = np.multiply(mu, 1+alpha*mu)
    pearson_residual = np.multiply(pearson_residual_numerator, 1/pearson_residual_denominator)
    sum_pearson_square = np.sum(pearson_residual)
    return sum_pearson_square - (n-p)

n,p = X.shape # n=292019; p=443
# initialization for beta and alpha
optimized_beta = np.loadtxt('optimized_beta.txt').reshape((443,1)) # shape: (443,1)
optimized_log_mu = np.matmul(X, optimized_beta)
optimized_mu = np.exp(optimized_log_mu) # shape: (292019,1)
beta = optimized_beta
alpha = 0.8 #0.7129375
log_alpha = np.log(alpha)

l_fr_diff = np.inf
l_fr_prev = -np.inf
count = 0

beta_2norm_list = [np.linalg.norm(beta)]
alpha_list = [np.exp(log_alpha)]

#l_0 = loglikelihood_NB(log_alpha, y, mu)
l_list = [loglikelihood_NB(log_alpha, y, optimized_mu)]
l_fr_list = [loglikelihood_NB_fr(log_alpha, y, optimized_mu, X)]

while l_fr_diff >= 1e-6:
    alpha = np.exp(log_alpha)
    log_mu = np.matmul(X, beta) # log_mu = beta_0 + beta_1 x1 + ... + beta_p x_p
    mu = np.exp(log_mu) # shape: (292019, 1)
    # maximize log-likelihood w.r.t beta with fixed alpha
    # w_i = (1+alpha y_i)*mu_i/(1+alpha mu_i)^2
    w_numerator = np.multiply(1+alpha*y, mu)
    w_denominator = np.multiply(1 + alpha * mu, 1 + alpha * mu)
    w = np.multiply(w_numerator, 1/w_denominator)
    # compute diagonal elements on H = WX (X^TWX)^(-1) X^T
    w_sqrt = np.sqrt(w)
    X_star = X_sparse.multiply(w_sqrt)# X* = W^(1/2) X; shape: (292019, 443)
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    # compute H = WX(X^T WX)^(-1) X^T
    XWX_inverse = inv(csc_matrix(XWX))
    WX = X_sparse.multiply(w) # shape: (292019, 443)
    hadamard_product = (WX @ XWX_inverse).multiply(X) # diag(AB^T) = (A @ B) 1
    h = np.sum(hadamard_product, axis=1) # h = (H_11,H_22,...,H_nn)^T
    # first order derivative w.r.t beta (firth regression)
    s = (y - mu)/(1+alpha*mu) # shape: (292019, 1)
    fr_term = 1/2 * np.multiply(h, (1-alpha*mu)/(1+alpha*mu))
    beta_first_derivative = X.T @ (s + fr_term) #dl(alpha, beta)/d beta = X^T (s + firth_regression_term); shape: (443, 1)
    # second order derivative w.r.t beta
    # d^2 l/d beta^2 = - X^T V X
    v_numerator = np.multiply(1+alpha*h+alpha*y, mu)
    v_denominator = np.multiply(1+alpha*mu, 1+alpha*mu)
    v = np.multiply(v_numerator, 1/v_denominator) # shape: (292019, 1)
    v_sqrt = np.sqrt(v)
    X_star2 = X_sparse.multiply(v_sqrt)# X* = V^(1/2) X
    XVX = X_star2.T @ X_star2 # XVX = (V^(1/2) X)^T (V^(1/2) X)
    beta_second_derivative = - XVX # shape: (443, 443)
    beta_Hessian_inverse = inv(csc_matrix(beta_second_derivative)) # shape: (443,443)
    beta_update =  beta_Hessian_inverse @ beta_first_derivative
    beta = beta - beta_update
    # save the 2 norm of beta in each iteration
    beta_2norm = np.linalg.norm(beta)
    beta_2norm_list.append(beta_2norm)
    # then use moment estimation of alpha
    # fixed point: log_alpha = log_alpha/(n-p) * sum a*(y_i-mu_i)^2/[mu_i(1+alpha*mu_i)]
    updated_log_mu = np.matmul(X, beta)
    updated_mu = np.exp(updated_log_mu)
    #l_fr_diff_alpha = np.inf
    #l_fr_prev_alpha = loglikelihood_NB_fr(log_alpha, y, updated_mu, X)
    #print(l_fr_prev_alpha)
    #while l_fr_diff_alpha > 1e-6:
    sum_term_numerator = np.multiply(y-updated_mu, y-updated_mu)
    sum_term_denominator = np.multiply(updated_mu, 1+alpha*updated_mu)
    updated_log_alpha = np.log(alpha / (n-p) * np.sum(np.multiply(sum_term_numerator, 1/sum_term_denominator)))
    #l_fr_alpha = loglikelihood_NB_fr(updated_log_alpha, y, updated_mu, X)
    #l_fr_diff_alpha = l_fr_alpha - l_fr_prev_alpha
    #print(alpha, l_fr_alpha, l_fr_diff_alpha)
    #l_fr_prev_alpha = l_fr_alpha
    log_alpha = updated_log_alpha
    alpha = np.exp(log_alpha)
    alpha_list.append(alpha)
    # compute the updated log-likelihood and penalized log-likelihood
    l = loglikelihood_NB(log_alpha, y, updated_mu)
    # compute penalized log-likelihood
    #l_fr = l_fr_alpha
    l_fr = loglikelihood_NB_fr(log_alpha, y, mu, X)
    l_list.append(l)
    l_fr_list.append(l_fr)
    print('The (penalized) log-likelihood in the current iteration is', l, l_fr)
    #l_star = l + 1/2 * np.log(np.linalg.det(XWX.toarray()))
    l_fr_diff = l_fr - l_fr_prev
    print(l_fr_diff, np.exp(updated_log_alpha), count)
    l_fr_prev = l_fr
    count += 1
    print('--------------------The ' + str(count) + "th iteration is done")


print(count)
print("The list of alpha values is", alpha_list)
print("The list of 2 norm of beta is", beta_2norm_list)
print("The list of log-likelihood is", l_list)
print("The list of firth-log-likelihood is", l_fr_list)
print('----------------------------------------------')
np.savetxt('final_beta_NB_fr_ME.txt', beta)
np.savetxt('alpha_NB_fr_ME.txt', np.array(alpha_list))
np.savetxt('beta_2norm_NB_fr_ME.txt', np.array(beta_2norm_list))
np.savetxt('l_NB_fr_ME.txt', np.array(l_list))
np.savetxt('l_fr_NB_fr_ME.txt', np.array(l_fr_list))
exit()


# load the convergent beta result
adjusted_beta_NB_fr_ME = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_ME_result_1e-6/final_beta_NB_fr_ME.txt')
adjusted_log_mu_NB_fr_ME = X @ adjusted_beta_NB_fr_ME
adjusted_y_pred_NB_fr_ME = np.exp(adjusted_log_mu_NB_fr_ME) # 1e-6: sum: 15336.471983965062; max: 0.49106178184442784; mean: 0.05251874701291718
print(np.min(adjusted_y_pred_NB_fr_ME), np.mean(adjusted_y_pred_NB_fr_ME), np.max(adjusted_y_pred_NB_fr_ME), np.sum(adjusted_y_pred_NB_fr_ME))

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
    response = adjusted_y_pred_NB_fr_ME[i]
    output_image[x_coord, y_coord, z_coord] = response

image = nib.Nifti1Image(output_image, np.eye(4))
image.to_filename('output_image_NB_fr_ME.nii.gz')
"""
"""
alpha_NB_fr_ME = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_ME_result_1e-6/alpha_NB_fr_ME.txt')
beta_2norm_NB_fr_ME = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_ME_result_1e-6/beta_2norm_NB_fr_ME.txt')
l_NB_fr_ME = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_ME_result_1e-6/l_NB_fr_ME.txt')
l_fr_NB_fr_ME = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_ME_result_1e-6/l_fr_NB_fr_ME.txt')
print(len(alpha_NB_fr_ME), len(beta_2norm_NB_fr_ME), len(l_NB_fr_ME), len(l_fr_NB_fr_ME))

x = np.arange(5, len(alpha_NB_fr_ME)) # x-axis: from 0 to 45

fig, axs = plt.subplots(2, 2)
fig.suptitle('Convergence plot in Negative Binomial model (tol=10^-6)')
axs[0, 0].plot(x, alpha_NB_fr_ME[5:])
axs[0, 0].set_title('alpha')
axs[0, 1].plot(x, beta_2norm_NB_fr_ME[5:], 'tab:orange')
axs[0, 1].set_title('L2-norm of beta')
axs[1, 0].plot(x, l_NB_fr_ME[5:], 'tab:green')
axs[1, 0].set_title('Log-likelihood')
axs[1, 1].plot(x, l_fr_NB_fr_ME[5:], 'tab:red')
axs[1, 1].set_title('Penalized log-likelihood')

for ax in axs.flat:
    ax.set(xlabel='Number of iteration')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

#plt.show()
fig.savefig('convergence_plot.png')
"""

"""
alpha_list, beta_2norm_list, l_list, l_fr_list = list(), list(), list(), list()
# define the negative log likelihood function
# minimize negative log likelihood function -> maximize log likelihood function
def nll(x, X, y):
    # x = np.array([beta_1, ..., beta_443, log_alpha])
    beta = x[:-1].reshape((X.shape[1],1)) # shape: (443, 1)
    beta_2norm = np.linalg.norm(beta)
    log_alpha = x[-1]
    alpha = np.exp(log_alpha)
    # log_mu = beta_0 + beta_1 x1 + ... + beta_p x_p
    log_mu = np.matmul(X, beta) 
    mu = np.exp(log_mu) 
    # compute log-likelihood
    y_max = np.max(y)
    first_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(np.log(array + 1/alpha))
        first_term += summation * count
    expression =  - np.log(scipy.special.gamma(y+1)) + y*np.log(mu) + y * log_alpha - (y+1/alpha)*np.log(1+alpha*mu)
    l = first_term + np.sum(expression)
    # compute the penalized term
    # l* = l + 1/2 log(det(X^T WX))
    w_numerator = np.multiply(1+alpha*y, mu)
    w_denominator = np.multiply(1 + alpha * mu, 1 + alpha * mu)
    w = np.multiply(w_numerator, 1/w_denominator)
    w_sqrt = np.sqrt(w)
    X_star = X_sparse.multiply(w_sqrt)# X* = W^(1/2) X; shape: (292019, 443)
    XWX = X_star.T @ X_star
    XWX_eigens = np.real(eigvals(XWX.todense()))
    log_XWX_eigens = np.log(XWX_eigens)
    log_det_XWX = np.sum(log_XWX_eigens)
    l_fr = l + 1/2 * log_det_XWX
    # save to list
    alpha_list.append(alpha)
    beta_2norm_list.append(beta_2norm)
    l_list.append(l)
    l_fr_list.append(l_fr)
    print('alpha and log-likelihood is', alpha, beta_2norm, l, l_fr)
    return -l_fr

# Jacobian: the matrix of all its first-order partial derivatives
def jac(x, X, y):
    # x = np.array([beta_1, ..., beta_443, log_alpha])
    beta = x[:-1].reshape((X.shape[1],1)) # shape: (443, 1)
    log_alpha = x[-1].item()
    alpha = np.exp(log_alpha)
    # log_mu = beta_0 + beta_1 x1 + ... + beta_p x_p
    log_mu = np.matmul(X, beta) 
    mu = np.exp(log_mu)

    # compute H = WX (X^T WX)^(-1) X^T
    w_numerator = np.multiply(1+alpha*y, mu)
    w_denominator = np.multiply(1 + alpha * mu, 1 + alpha * mu)
    w = np.multiply(w_numerator, 1/w_denominator)
    # compute diagonal elements on H = WX (X^TWX)^(-1) X^T
    w_sqrt = np.sqrt(w)
    X_star = csr_matrix(X).multiply(w_sqrt)# X* = W^(1/2) X; shape: (292019, 443)
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    # compute H = WX(X^T WX)^(-1) X^T
    XWX_inverse = inv(csc_matrix(XWX))
    WX = csr_matrix(X).multiply(w) # shape: (292019, 443)
    hadamard_product = (WX @ XWX_inverse).multiply(X) # diag(AB^T) = (A @ B) 1; @: hadamard product / elementwise product
    h = np.sum(hadamard_product, axis=1) # h = (H_11,H_22,...,H_nn)^T
    h = np.array(h) # convert numpy matrix to array
    # compute the first order derivative (vector) of beta
    s = np.multiply(y-mu, 1/(1+alpha*mu)) # s_i = (y_i - mu_i)/(1+alpha*mu_i)
    fr_term_beta = np.asarray(1/2 * h * (1-alpha*mu)/(1+alpha*mu)) # firth regression term: 1/2 H_ii (1-alpha*mu_i)/(1+alpha*mu_i)
    beta_first_derivative = X.T @ (s + fr_term_beta) #dl(alpha, beta)/d beta = X^T s; shape: (443, 1)
    # compute the first order derivative of log_alpha
    y_max = np.max(y)
    add_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(1/(array + 1/alpha))
        add_term += -1/alpha**2 * summation * count
    expression = 1/alpha**2 * np.log(1+alpha*mu) + 1/alpha * (y - mu)/(1+alpha*mu)
    # firth regression term in first order derivative of alpha
    # 1/2 * sum[(y_i-alpha*y_i*mu_i-2*mu_i)/(1+alpha*mu_i)*(1+alpha*y_i) H_ii]
    #fr_term_numerator = y - alpha*np.multiply(y,mu) - 2*mu
    #fr_term_denominator = np.multiply(1+alpha*mu, 1+alpha*y)
    #fr_term_frac = np.multiply(fr_term_numerator, 1/fr_term_denominator)
    #fr_term_alpha = 1/2 * np.sum(np.multiply(fr_term_frac, h))
    dl_d_alpha = add_term + np.sum(expression) #+ fr_term_alpha
    #dl_d_alpha = add_term + np.sum(expression)
    dl_d_log_alpha = dl_d_alpha * alpha
    dl_d_log_alpha_vector = np.array([dl_d_log_alpha]).reshape((1,1))
    jac = np.concatenate((beta_first_derivative, dl_d_log_alpha_vector), axis=0) #shape: (444,1)
    return -jac.reshape((444,))

# initialization for beta and log_alpha
optimized_beta = np.loadtxt('optimized_beta.txt').reshape((443,))
optimized_log_mu = np.matmul(X, optimized_beta)
optimized_mu = np.exp(optimized_log_mu) # shape: (292019,1)
beta = optimized_beta
alpha = 0.8
log_alpha = np.log(alpha)

log_alpha_vec = np.array(log_alpha).reshape((1,))
x0 = np.concatenate((beta, log_alpha_vec), axis=0) # shape:(444,)
# minimization of negative log-likelihood with BFGS
minimizer = minimize(nll, x0, args=(X,y), method='BFGS', jac=jac, tol=1e-6)
np.savetxt('minimizer_x.txt', minimizer.x)
beta_ast = minimizer.x[:-1]
log_alpha_ast = minimizer.x[-1]
alpha_ast = np.exp(log_alpha_ast) 
print(alpha_ast)
print(alpha_list, beta_2norm_list, l_list, l_fr_list)

np.savetxt('beta_2norm_NB_BFGS.txt', np.array(beta_2norm_list))
np.savetxt('l_NB_BFGS.txt', np.array(l_list))
np.savetxt('l_fr_NB_BFGS.txt', np.array(l_fr_list))
"""

"""
minimizer_x = np.loadtxt("adjusted_NB_fr_BFGS_result_1e-6/minimizer_x.txt") # after 757 iterations
beta = minimizer_x[:-1] # shape: (443,)
log_alpha = minimizer_x[-1] # log_alpha = -0.313834
alpha = np.exp(log_alpha) # alpha = 0.7151511344556296
print(alpha)
#log_mu = beta_0 + beta_1 x1 + ... + beta_p x_p
log_mu = X @ beta
y_pred_BFGS = np.exp(log_mu) # max: 0.5385744239419953; sum: 15332.226896260328; mean: 0.05250420998722798
print(np.max(y_pred_BFGS), np.sum(y_pred_BFGS), np.mean(y_pred_BFGS))

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
    response = y_pred_BFGS[i]
    output_image[x_coord, y_coord, z_coord] = response

image = nib.Nifti1Image(output_image, np.eye(4))
image.to_filename('adjusted_NB_fr_BFGS_result_1e-6/output_image_NB_BFGS.nii.gz')
"""

alpha_NB_BFGS = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_BFGS_result_1e-6/alpha_NB_BFGS.txt')
beta_2norm_NB_BFGS = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_BFGS_result_1e-6/beta_2norm_NB_BFGS.txt')
l_Poisson_NB_BFGS = np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_BFGS_result_1e-6/l_NB_BFGS.txt')
l_fr_NB_BFGS= np.loadtxt('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_BFGS_result_1e-6/l_fr_NB_BFGS.txt')
print(len(alpha_NB_BFGS), len(beta_2norm_NB_BFGS), len(l_Poisson_NB_BFGS), len(l_fr_NB_BFGS))


x = np.arange(10, len(alpha_NB_BFGS)) # x-axis: from 5 to 131

fig, axs = plt.subplots(2, 2)
fig.suptitle('Convergence plot in Log-linear model (tol=10^-6)')

axs[0, 0].plot(x, alpha_NB_BFGS[10:], 'tab:orange')
axs[0, 0].set_title('alpha')
axs[0, 1].plot(x, beta_2norm_NB_BFGS[10:], 'tab:orange')
axs[0, 1].set_title('L2-norm of beta')
axs[1, 0].plot(x, l_Poisson_NB_BFGS[10:], 'tab:green')
axs[1, 0].set_title('Log-likelihood')
axs[1, 1].plot(x, l_fr_NB_BFGS[10:], 'tab:red')
axs[1, 1].set_title('Penalized log-likelihood')


for ax in axs.flat:
    ax.set(xlabel='Number of iteration')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

#plt.show()
fig.savefig('/Users/yifan/Documents/GitHub/Meta_regression/adjusted_NB_fr_BFGS_result_1e-6/convergence_plot_skip10.png')
exit()