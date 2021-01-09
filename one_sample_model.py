import numpy as np
import pandas as pd
import scipy 
from scipy.special import factorial
from scipy.sparse import load_npz, csr_matrix, csc_matrix, hstack, vstack
from scipy.sparse.linalg import inv
from scipy.optimize import minimize
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
beta_i = np.log(np.sum(y_sparse)/X.shape[0])
beta = np.full(shape=(X.shape[1],), fill_value=beta_i).reshape((X.shape[1],1)) # shape: (443,1)
# sum_i (log(y_i!))
#sum_logy_factorial = sum(log(factorial(y)))
sum_logy_factorial = np.sum(np.log(factorial(y)))
diff = np.inf # initializations for the loop
l_beta_prev = -np.inf
count = 0
while diff >= 0.5:
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
    print(np.linalg.cond(XWX.todense()))
    
    XWX_inverse = inv(csc_matrix(XWX)) 
    update = XWX_inverse @ X.T @ (y-mu) #(XWX)^(-1) X^T (y-mu)
    beta += update
    count += 1
    l_beta_prev = l_beta
# takes 15 iterations
np.savetxt('beta.txt', beta, fmt='%f')
l_poisson = l_beta # -54084.26918761416


# compute BIC = -2*ln(L) + k*ln(n)
#k = beta.shape[0] # 443
#n = X.shape[0] # 292019
#BIC_poisson = -2*np.log(l_poisson) + k * np.log(n)
#print(k,n,BIC_poisson)


beta = np.loadtxt('beta.txt')
g_mu = X @ beta
y_pred = np.exp(g_mu) # max: 0.4223341367479041; mean: 0.051815815234713014 ~ 15131/292019
mu = np.exp(g_mu)
# compute Pearson residual


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
    response = y_pred[i]
    output_image[x_coord, y_coord, z_coord] = response

image = nib.Nifti1Image(output_image, np.eye(4))
image.to_filename('output_image.nii.gz')  # Save as NiBabel file
"""
"""
# BFGS method for minimizing negative log-likehood with Negative Binomial model
# define the negative log likelihood function
# minimize negative log likelihood function -> maximize log likelihood function
def nll(x, X, y):
    # x = np.array([beta_1, ..., beta_443, log_alpha])
    beta = x[:-1].reshape((X.shape[1],1)) # shape: (443, 1)
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
    print(-l, np.min(mu))
    return -l

# Jacobian: the matrix of all its first-order partial derivatives
def jac(x, X, y):
    # x = np.array([beta_1, ..., beta_443, log_alpha])
    beta = x[:-1].reshape((X.shape[1],1)) # shape: (443, 1)
    log_alpha = x[-1].item()
    alpha = np.exp(log_alpha)
    # log_mu = beta_0 + beta_1 x1 + ... + beta_p x_p
    log_mu = np.matmul(X, beta) 
    mu = np.exp(log_mu)
    # compute the first order derivative (vector) of beta
    s = (y - mu)/(1+alpha*mu) # s_i = (y_i - mu_i)/(1+alpha*mu_i)
    beta_first_derivative = X.T @ s #dl(alpha, beta)/d beta = X^T s; shape: (443, 1)
    # compute the first order derivative of log_alpha
    y_max = np.max(y)
    add_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(1/(array + 1/alpha))
        add_term += -1/alpha**2 * summation * count
    expression = 1/alpha**2 * np.log(1+alpha*mu) + 1/alpha * (y - mu)/(1+alpha*mu)
    dl_d_alpha = add_term + np.sum(expression)
    dl_d_log_alpha = dl_d_alpha * alpha
    dl_d_log_alpha_vector = np.array([dl_d_log_alpha]).reshape((1,1))
    jac = np.concatenate((beta_first_derivative, dl_d_log_alpha_vector), axis=0) #shape: (444,1)
    return -jac.reshape((444,))

# initialization for beta and log_alpha
beta_i = np.log(np.sum(y_sparse)/X.shape[0])
beta = np.full(shape=(X.shape[1],), fill_value=beta_i).reshape((X.shape[1],)) # shape: (443,)
log_alpha = np.log(2)
log_alpha_vec = np.array(log_alpha).reshape((1,))
x0 = np.concatenate((beta, log_alpha_vec), axis=0) # shape:(444,)
# minimization of negative log-likelihood with BFGS
minimizer = minimize(nll, x0, args=(X,y), method='BFGS', jac=jac, tol=1e-2)
np.savetxt('minimizer_x.txt', minimizer.x, fmt='%f')
beta_ast = minimizer.x[:-1]
log_alpha_ast = minimizer.x[-1]
alpha_ast = np.exp(log_alpha_ast) #0.7169196743432069
print(alpha_ast)
exit()

# the final minimized negative log-likelihood is 53943.639996281156
minimizer_x = np.loadtxt("minimizer_x.txt") # after 757 iterations
beta = minimizer_x[:-1] # shape: (443,)
log_alpha = minimizer_x[-1] # log_alpha = -0.313834
alpha = np.exp(log_alpha) # alpha = 0.7306403043837475
#log_mu = beta_0 + beta_1 x1 + ... + beta_p x_p
log_mu = X @ beta
y_pred = np.exp(log_mu) # max: 0.42743450015748446; sum: 15120.125010428963; mean: 0.05177788092702517 ~ 15131/292019 = 0.051815121618798775

print(np.max(y_pred), np.sum(y_pred), np.mean(y_pred))

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
    response = y_pred[i]
    output_image[x_coord, y_coord, z_coord] = response

image = nib.Nifti1Image(output_image, np.eye(4))
image.to_filename('output_image_optimization_NB.nii.gz')  # Save as NiBabel file
"""


# IRLS for Negative Binomial model
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
    expression =  - np.log(scipy.special.gamma(y+1)) + y*np.log(mu) + y * log_alpha - (y+1/alpha)*np.log(1+alpha*mu)
    l = first_term + np.sum(expression)
    return l

# first-order partial derivative: d l(beta, alpha)/d alpha
def first_order_derivative_alpha(log_alpha, y, mu):
    alpha = np.exp(log_alpha)
    y_max = np.max(y)
    add_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(1/(array + 1/alpha))
        add_term += -1/alpha**2 * summation * count
    expression = 1/alpha**2 * np.log(1+alpha*mu) + 1/alpha * (y - mu)/(1+alpha*mu)
    dl_d_alpha = add_term + np.sum(expression)
    dl_d_log_alpha = dl_d_alpha 
    return np.array(dl_d_log_alpha).reshape(1,1)


# second-order partial derivative: d^2 l(beta, alpha)/d alpha^2
def second_order_derivative_alpha(log_alpha, y, mu):
    alpha = np.exp(log_alpha)
    y_max = np.max(y)
    add_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum((array/(1+alpha*array))**2)
        add_term += -1 * summation * count
    expression = -2/alpha**3 * np.log(1+alpha*mu) + 2/alpha**2 * mu / (1+alpha*mu) + (y+1/alpha) * mu**2 / (1+alpha*mu)**2
    dl2_d_alpha2 = add_term + np.sum(expression)
    return np.array(dl2_d_alpha2).reshape(1,1)

# initialization for beta and alpha
beta_i = np.log(np.sum(y_sparse)/X.shape[0])
beta = np.full(shape=(X.shape[1],), fill_value=beta_i).reshape((X.shape[1],1)) # shape: (443,1)
log_alpha = np.log(2)

l_prev = np.inf
log_alpha_update = np.inf
count = 0
while log_alpha_update >= 1e-4:
    alpha = np.exp(log_alpha)
    log_mu = np.matmul(X, beta) # log_mu = beta_0 + beta_1 x1 + ... + beta_p x_p
    mu = np.exp(log_mu) # shape: (292019, 1)
    # log-likelihood and difference to the previous iteration
    #l = loglikelihood_NB(log_alpha, y, mu)
    #diff = l - l_prev 
    #print(l ,diff)
    # maximize log-likelihood w.r.t beta with fixed alpha
    # first order derivative w.r.t beta
    s = (y - mu)/(1+alpha*mu) # s_i = (y_i - mu_i)/(1+alpha*mu_i); shape: (292019, 1)
    beta_first_derivative = X.T @ s #dl(alpha, beta)/d beta = X^T s; shape: (443, 1)
    # second order derivative w.r.t beta
    w = (1 + alpha * y)* mu / (1 + alpha * mu)**2 # w_i = (1+alpha y_i)*mu_i/(1+alpha mu_i)
    w_sqrt = np.sqrt(w)
    X_star = X_sparse.multiply(w_sqrt)# X* = W^(1/2) X; shape: (292019, 443)
    beta_second_derivative = -X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X); shape: (443,443)
    print("condition number is", np.linalg.cond(beta_second_derivative.todense()))
    beta_Hessian_inverse = inv(csc_matrix(beta_second_derivative)) # shape: (443,443)
    beta_update =  beta_Hessian_inverse @ beta_first_derivative
    beta_update_2norm = np.linalg.norm(beta_update)
    print("2 norm of beta update is ", beta_update_2norm)
    beta = beta - beta_update
    # then maximize log-likelihood w.r.t alpha with updated beta
    updated_log_mu = np.matmul(X, beta)
    updated_mu = np.exp(updated_log_mu)
    log_alpha_first_derivative = alpha * first_order_derivative_alpha(log_alpha, y, updated_mu)
    log_alpha_second_derivative = alpha**2 * second_order_derivative_alpha(log_alpha, y, updated_mu) + 2 * log_alpha_first_derivative
    log_alpha_update = 1/log_alpha_second_derivative * log_alpha_first_derivative
    print('The first derivative of log_alpha is', log_alpha_first_derivative)
    log_alpha = log_alpha - log_alpha_update
    count += 1
    print(count, log_alpha, np.min(mu))
    print('-------------------------------')

g_mu = X @ beta
y_pred = np.exp(g_mu) # max: 0.428027478648621; mean: 0.05178027803359981 ~ 15131/292019
print(np.max(y_pred), np.mean(y_pred), np.min(y_pred))

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
    response = y_pred[i]
    output_image[x_coord, y_coord, z_coord] = response

image = nib.Nifti1Image(output_image, np.eye(4))
image.to_filename('output_image_NB.nii.gz')  # Save as NiBabel file
exit()





# IRLS for Negative Binomial model
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
    expression =  - np.log(scipy.special.gamma(y+1)) + y*np.log(mu) + y * log_alpha - (y+1/alpha)*np.log(1+alpha*mu)
    l = first_term + np.sum(expression)
    return l

# first-order partial derivative: d l(beta, alpha)/d alpha
def first_order_derivative_alpha(log_alpha, y, mu):
    alpha = np.exp(log_alpha)
    y_max = np.max(y)
    add_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum(1/(array + 1/alpha))
        add_term += -1/alpha**2 * summation * count
    expression = 1/alpha**2 * np.log(1+alpha*mu) + 1/alpha * (y - mu)/(1+alpha*mu)
    dl_d_alpha = add_term + np.sum(expression)
    dl_d_log_alpha = dl_d_alpha 
    return np.array(dl_d_log_alpha).reshape(1,1)


# second-order partial derivative: d^2 l(beta, alpha)/d alpha^2
def second_order_derivative_alpha(log_alpha, y, mu):
    alpha = np.exp(log_alpha)
    y_max = np.max(y)
    add_term = 0
    for y_i in range(1,y_max+1):
        count = np.count_nonzero(y == y_i)
        array = np.arange(y_i)
        summation = np.sum((array/(1+alpha*array))**2)
        add_term += -1 * summation * count
    expression = -2/alpha**3 * np.log(1+alpha*mu) + 2/alpha**2 * mu / (1+alpha*mu) + (y+1/alpha) * mu**2 / (1+alpha*mu)**2
    dl2_d_alpha2 = add_term + np.sum(expression)
    return np.array(dl2_d_alpha2).reshape(1,1)

# initialization for beta and alpha
beta_i = np.log(np.sum(y_sparse)/(X.shape[0])) # -2.96

beta = np.full(shape=(X.shape[1],), fill_value=beta_i).reshape((X.shape[1],1)) # shape: (443,1)
log_alpha = np.log(2)

update_2norm = np.inf
count = 0

while update_2norm >= 1e-2:
    alpha = np.exp(log_alpha)
    print('---------------')
    print("alpha=", alpha)
    print('---------------')
    # concatenate to a column array
    log_alpha_vec = np.array(log_alpha).reshape((1,1))
    beta_log_alpha = np.concatenate((beta, log_alpha_vec), axis=0) # shape: (443, 1)
    log_mu = np.matmul(X, beta) # log_mu = beta_1 x1 + ... + beta_p x_p
    mu = np.exp(log_mu) 
    # log-likelihood and difference to the previous iteration
    l = loglikelihood_NB(log_alpha, y, mu)
    #diff = l - l_prev 
    print("Log-likelihood in the current iteration is", l)
    #print(diff)
    s = (y - mu)/(1+alpha*mu) # s_i = (y_i - mu_i)/(1+alpha*mu_i)
    beta_first_derivative = X.T @ s #dl(alpha, beta)/d beta = X^T s; shape: (444, 1)
    # dl/d(log_alpha) = alpha * dl/d alpha
    log_alpha_first_derivative = alpha * first_order_derivative_alpha(log_alpha, y, mu)
    # Jacobian matrix (all the first-order partial derivatives)
    Jacobian_matrix = np.concatenate((beta_first_derivative,log_alpha_first_derivative), axis=0) # shape: (444,1)
    # second-order partial derivative of beta
    w = (1 + alpha * y)* mu / (1 + alpha * mu)**2 # w_i = (1+alpha y_i)*mu_i/(1+alpha mu_i)
    w_sqrt = np.sqrt(w)
    X_star = X_sparse.multiply(w_sqrt)# X* = W^(1/2) X
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X); shape: (443,443)
    # d^2 l /d log_alpha d beta= dl / d beta + alpha * d^2 l/ d alpha d beta
    t = (mu - y) * mu / (1+alpha*mu)**2 
    derivative_alpha_beta = X.T @ t
    derivative_log_alpha_beta = csr_matrix(alpha * derivative_alpha_beta + beta_first_derivative) # shape: (443,1)
    log_alpha_second_derivative = alpha**2 * second_order_derivative_alpha(log_alpha, y, mu) + 2 * log_alpha_first_derivative
    # Hessian matrix (second-order partial derivatives)
    Hessian_matrix = hstack([XWX, derivative_log_alpha_beta]) # shape: (443, 444)
    Hessian_last_row = hstack([derivative_log_alpha_beta.T, csr_matrix(log_alpha_second_derivative)]) # shape: (1, 444)
    Hessian_matrix = vstack([Hessian_matrix, Hessian_last_row])
    #print("determinant is", np.linalg.det(Hessian_matrix.todense()))
    print("condition number is", np.linalg.cond(Hessian_matrix.todense()))
    #print("eiganvalues are ",np.linalg.eigvals(Hessian_matrix.todense())[-10:])
    Hessian_inverse = inv(csc_matrix(Hessian_matrix)) 
    # update beta and alpha
    update = Hessian_inverse @ Jacobian_matrix # shape: (444, 1)
    update_2norm = np.linalg.norm(update)
    print("2 norm is", update_2norm)
    beta_log_alpha += update
    beta = beta_log_alpha[:-1, :]
    log_alpha = beta_log_alpha[-1, :].item()
    count += 1
    print(count)
    print(np.min(mu))
    print('-----------------------')


final_log_mu = np.matmul(X, beta) # log_mu = beta_1 x1 + ... + beta_p x_p
final_mu = np.exp(log_mu) 
final_l = loglikelihood_NB(log_alpha, y, final_mu)
print(final_l)










