import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as tdist
from scipy.sparse import load_npz

# p = tdist.poisson.Poisson(0.01)
# size = [100, 100, 100]
# count = p.sample(sample_shape=size)
# grid = np.meshgrid(*[np.arange(s) for s in size])
# coords = np.stack(grid, axis=-1).reshape([np.prod(size),len(size)])
# count = np.reshape(count, [np.prod(size),1])

# load foci and counts from npz files 
X_sparse = load_npz("preprocessed_data/X.npz") 
y_verbal_sparse = load_npz("preprocessed_data/y_verbal.npz").transpose()
X = X_sparse.toarray() # shape: (228453, 315)
y_verbal = y_verbal_sparse.toarray() 
y_verbal = y_verbal.astype(dtype=float)

def rff(inputs, n_feature=128):
    b, d = inputs.shape
    B = np.random.randn(n_feature, d)
    A = 2*np.pi * inputs @ B.T
    outputs = np.concatenate([np.cos(A), np.sin(A)], axis=1)
    return torch.Tensor(outputs)

class GLMPoisson(torch.nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.linear = torch.nn.Linear(self.in_dim, 1, bias=False)
        torch.nn.init.uniform_(self.linear.weight, a=-0.01, b=0.01)
        # torch.nn.init.constant_(self.linear.weight, -1.22)
        
    def forward(self, x, c):
        log_mu = self.linear(x)
        mu = log_mu.exp()
        p = tdist.poisson.Poisson(mu)
        log_prob = p.log_prob(c)
        obj = -log_prob.mean()
        return obj

# n_feature = 128
# inputs = rff(coords, n_feature=n_feature)

## Spline
# inputs = torch.tensor(X, dtype=torch.float32)
# count = torch.tensor(y_verbal, dtype=torch.float32)
# model = GLMPoisson(425)
## RFF
n_feature = 128
inputs = rff(coords, n_feature=n_feature)
count = torch.tensor(y_verbal, dtype=torch.float32)
model = GLMPoisson(n_feature*2)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)

for step in range(50):
    def closure():
        optimizer.zero_grad()
        optimizer.zero_grad()
        loss = model(inputs, count)
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    print("step {0}: loss {1}".format(step, loss))

