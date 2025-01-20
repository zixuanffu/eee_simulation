# ## Figures

# ### Likelihood
# 1. MLE (true)
# 2. Oracle D (true)
# 3. Correctly specified D (trained)
# 4. Neural network D (trained) 
# 5. Misspecified D (true and trained)
# 6. quasi-MLE (true)

# - Figure 1: 1 & 2 & 3
# - Figure 2: 1 & 2 & 4
# - Figure 3: 6 & 5 (the misspecified D has two versions: the truly misspecified and the trained)
# - I want to compare: 2 & 3 & 4
# ### Moments
# 1. Oracle true D (What is the oracle D?) (true)
# 2. Misspecified D (The logistic location model with increasing numbers of inputs. The curvature of the cross-entropy loss is very close to the log likelihood up to 7 moments and is still good for 11 moments.) (trained)
# 3. SMM loglikelihood (true)
# 4. Neural network D (trained) (to be trained haha)

# - Figure 4: 1 & 2 & 3
# - I want to compare: 1 & 4 & 3


import numpy as np
from scipy.integrate import quad
from time import time

# import functions
from .functions import logisticcdf


# Setup
n = 300
M = np.array([n, 2 * n])  # Sample sizes
S = 500  # Simulation size
T = lambda th, x: th + x  # Transformation map
g = np.array([20, 250])  # Neural network configuration
node = 5  # Number of nodes
th0 = 0  # True parameter


# Check size consistency
assert len(M) == len(g), "Size of M and g must be the same."

# Analytic calculation
log_p0 = lambda x: loglogisticpdf(x, th0, 1)
log_p = lambda th, x: loglogisticpdf(x, th, 1)

ld = lambda th, x: -1 + 2 * logisticcdf(x - th, 0, 1)
ldd = lambda th, x: -2 * logisticcdf(x - th, 0, 1) * logisticcdf(-(x - th), 0, 1)

log_D = lambda th, x: log_p0(x) - logaddexp(log_p0(x), log_p(th, x))
log_UmD = lambda th, x: log_p(th, x) - logaddexp(log_p0(x), log_p(th, x))
p0 = lambda x: np.exp(log_p0(x))
p = lambda th, x: np.exp(log_p(th, x))
D = lambda th, x: np.exp(log_D(th, x))

# Precompute constants for integration
meat_func = lambda x: 4 * np.exp(log_D(th0, x) + log_UmD(th0, x)) * ld(th0, x)**2
Meat_1 = quad(lambda x: meat_func(x) * p(th0, x), -np.inf, np.inf, epsabs=1e-20)[0]
Meat_2 = quad(lambda x: meat_func(x) * p0(x), -np.inf, np.inf, epsabs=1e-20)[0]
V = Meat_1 + (n / M) * Meat_2

bun_func = lambda x: 2 * (D(th0, x) * ld(th0, x)**2 + (ldd(th0, x) + ld(th0, x)**2) * log_UmD(th0, x))
I_tilde = quad(lambda x: bun_func(x) * p(th0, x), -np.inf, np.inf, epsabs=1e-20)[0]

# Asymptotic variances
V_Adv = np.linalg.inv(I_tilde) @ V @ np.linalg.inv(I_tilde)
V_MLE = -1 / quad(lambda x: ldd(th0, x) * p0(x), -np.inf, np.inf, epsabs=1e-20)[0]

# Orthogonality
K = 50
th_grid = np.linspace(-0.7, 0.7, K)

# Generate real and latent data
X = logisticrnd(0, 1, (n, 1))
Z = logisticrnd(0, 1, (max(M), 1))

# Precompute Z slices
Z_slices = {m: Z[:m] for m in M}

# Initialize arrays for results
LL_grid = -np.mean([log_p(th, X) for th in th_grid], axis=1)
oD_grid = np.zeros((len(M), K))
cD_grid = np.zeros((len(M), K))
NND_grid = np.zeros((len(M), K))

# Vectorized computation over th_grid and M
start_time = time()
for m_idx, m_size in enumerate(M):
    Z_m = Z_slices[m_size]
    T_Z = np.array([T(th, Z_m) for th in th_grid])
    oD_grid[m_idx, :] = np.mean([log_D(th, X) + log_UmD(th, T_Z[i]) for i, th in enumerate(th_grid)], axis=1)
    cD_grid[m_idx, :] = np.mean([loss(X, T_Z[i]) for i in range(K)], axis=1)
    NND_grid[m_idx, :] = np.mean([NND(X.T, T_Z[i].T, g[m_idx], node) for i in range(K)], axis=1)

# Precompute for th0
LL0_grid = -np.mean(log_p(th0, X))
oD0_grid = np.zeros(len(M))
cD0_grid = np.zeros(len(M))
NND0_grid = np.zeros(len(M))

for m_idx, m_size in enumerate(M):
    Z_m = Z_slices[m_size]
    oD0_grid[m_idx] = np.mean(log_D(th0, X) + log_UmD(th0, T(th0, Z_m)))
    cD0_grid[m_idx] = loss(X, T(th0, Z_m))
    NND0_grid[m_idx] = NND(X.T, T(th0, Z_m).T, g[m_idx], node)

end_time = time()

# Print elapsed time
print(f"Optimized elapsed time: {end_time - start_time:.2f} seconds")