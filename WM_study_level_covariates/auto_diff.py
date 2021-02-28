import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, hessian
from scipy.sparse import load_npz, csr_matrix, csc_matrix
from scipy.sparse.linalg import inv, cg
from scipy.special import factorial

# load foci and counts from npz files 
X_sparse = load_npz("preprocessed_data/X.npz") 
y_verbal_sparse = load_npz("preprocessed_data/y_verbal.npz").transpose()

X = X_sparse.toarray() # shape: (228453, 315)
y_verbal = y_verbal_sparse.toarray() 
y_verbal = y_verbal.astype(dtype=float)

X_jax = jnp.array(X) # shape: (228453, 425)
y_verbal_jax = jnp.array(y_verbal) # shape: (228453, 1)

#X = np.random.rand(1000, 425) / 100.
#X = jnp.array(X)
#y_verbal = np.random.rand(1000, 1)
#y_verbal = jnp.array(y_verbal)

X = X_jax[200000: 201000, :]
y_verbal= jnp.array(y_verbal_jax[200000: 201000, :])

def loglikelihood_Poisson(beta):
    y = y_verbal
    beta = beta.reshape((425,1))
    log_mu = X @ beta
    mu = jnp.exp(log_mu)
    #sum_logy_factorial = sum(log(factorial(y)))
    sum_logy_factorial = jnp.sum(jnp.log(factorial(y)))
    # l*(beta) = l(beta) + 1/2 log(|I(theta)|)
    #          = sum(y_i*log(mu_i)-mu_i-log(y_i!)) + 1/2 * log(det(XWX))
    y_log_mu = jnp.multiply(y, log_mu)
    l = jnp.sum(y_log_mu) - jnp.sum(mu) - sum_logy_factorial
    # compute the penalized term
    # l* = l + 1/2 log(det(X^T WX))
    mu_sqrt = jnp.sqrt(mu)
    mu_sqrt = mu_sqrt.reshape(y.shape[0],)
    X_star = jnp.einsum("ij,i->ij", X, mu_sqrt) # row-wise multiplication 
    XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)
    #det_XWX = jnp.linalg.det(XWX)
    #log_det_XWX = jnp.log(det_XWX)
    XWX_eigens = jnp.linalg.eigh(XWX)[0]
    log_XWX_eigens = jnp.log(XWX_eigens)
    print(log_XWX_eigens.shape)
    log_det_XWX = jnp.sum(log_XWX_eigens)
    l_fr = l + 1/2 * log_det_XWX
    return l_fr

# initialization
# initialization for beta and gamma => mu_X and mu_Z
#beta_i = np.log(np.sum(y_verbal)/X.shape[0])
#beta = np.full(shape=(X.shape[1], ), fill_value=beta_i) # shape: (425, 1)


beta = np.random.randn(425) 
#beta = jnp.array(beta)


X_sparse = csr_matrix(X)
beta = beta.reshape(425, 1)
g_mu = np.matmul(X, beta)
mu = np.exp(g_mu) # mu: mean vector (log link)
mu_sqrt = np.sqrt(mu)
# compute the update
X_star = csr_matrix(X).multiply(mu_sqrt)# X* = W^(1/2) X
XWX = X_star.T @ X_star # XWX = (W^(1/2) X)^T (W^(1/2) X)

XWX = XWX.toarray()
XWX_eigens = np.linalg.eigh(XWX)[0]
det_XWX = np.linalg.det(XWX)
print(np.min(XWX_eigens), np.max(XWX_eigens))
print(det_XWX)
exit()
log_XWX_eigens = jnp.log(XWX_eigens)
exit()
XWX_inverse = inv(csc_matrix(XWX)) 
# compute the update
WX = X_sparse.multiply(mu) 
hadamard_product = (WX @ XWX_inverse).multiply(X) # diag(AB^T) = (A @ B) 1
h = np.sum(hadamard_product, axis=1) # h = (H_11,H_22,...,H_nn)^T, H = WX(X^T WX)^(-1) X^T
first_order_derivarive = X.transpose() @ (y_verbal + 1/2*h - mu)
first_order_derivarive = np.asarray(first_order_derivarive)
second_order_derivarive = - XWX 
second_order_derivarive = second_order_derivarive.toarray()
#print(second_order_derivarive[0:10, 201])


Jacobian_fun = jacfwd(loglikelihood_Poisson)
Jacobian_beta = Jacobian_fun(beta)
print(Jacobian_beta[0][0:10])
exit()

Hessian_fun = hessian(loglikelihood_Poisson)
Hessian_beta = Hessian_fun(beta_jax)
print(Hessian_beta[0:10, 201])
exit()