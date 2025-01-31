import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim


def logisticcdf(x, mu=0, sigma=1, uflag=None):
    """
    Logistic cumulative distribution function (CDF).

    Parameters:
    x : scalar or array_like
        Values at which to evaluate the CDF.
    mu : float, optional
        Location parameter (default is 0).
    sigma : float, optional
        Scale parameter (default is 1).
    uflag : str, optional
        Specify 'upper' for complementary CDF (default is None).

    Returns:
    y : scalar or ndarray
        CDF values of the logistic distribution.
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    if np.isscalar(sigma) and sigma <= 0 or np.any(sigma <= 0):
        raise ValueError("Scale parameter 'sigma' must be positive.")

    if np.isscalar(x):
        if not np.isreal(x):
            raise ValueError("Input x must be real.")
    elif not np.isreal(x).all():
        raise ValueError("Input x must be real.")

    if uflag is not None and uflag.lower() == 'upper':
        x, mu = -x, -mu

    y = 1 / (1 + np.exp(-(x - mu) / sigma))
    return y.item() if np.isscalar(x) else y


def logisticrnd(m=0, s=1, *size):
    if len(size) == 0:
        size = np.broadcast(np.array(m), np.array(s)).shape
    elif len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])

    u = np.random.rand(*size)
    return logisticinv(u, m, s)


def logisticinv(u, m, s):
    u = np.asarray(u)
    m = np.asarray(m)
    s = np.asarray(s)
    return m + s * np.log(u / (1 - u))


def loglogisticpdf(x, mu=0, sigma=1):
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    if np.isscalar(sigma) and sigma <= 0 or np.any(sigma <= 0):
        raise ValueError("Scale parameter 'sigma' must be positive.")

    z = (x - mu) / sigma
    y = -np.log(sigma) - softplus(-z) - softplus(z)
    return y.item() if np.isscalar(x) else y


def softplus(x):
    x = np.asarray(x)
    return np.log1p(np.exp(x))


def loglogisticcdf(x, mu=0, sigma=1, uflag=None):
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    if np.isscalar(sigma) and sigma <= 0 or np.any(sigma <= 0):
        raise ValueError("Scale parameter 'sigma' must be positive.")

    if np.isscalar(x):
        if not np.isreal(x):
            raise ValueError("Input x must be real.")
    elif not np.isreal(x).all():
        raise ValueError("Input x must be real.")

    if uflag is not None and uflag.lower() == 'upper':
        x, mu = -x, -mu

    y = -softplus(-(x - mu) / sigma)
    return y.item() if np.isscalar(x) else y


def logaddexp(x, y):
    x, y = np.asarray(x), np.asarray(y)
    m = np.maximum(np.real(x), np.real(y))

    if np.isscalar(m):
        if np.isinf(m):
            m = 0
    else:
        m[np.isinf(m)] = 0

    z = np.log(np.exp(x - m) + np.exp(y - m)) + m
    return z.item() if np.isscalar(x) and np.isscalar(y) else z


def loss(X1, X2):
    def objective(l):
        return (np.mean(softplus(indexf(X1, l))) +
                np.mean(softplus(-indexf(X2, l))))

    initial_guess = np.array([1, 2, 1])
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    beta = result.x
    v = -result.fun
    return v, beta


def indexf(x, l):
    x = np.asarray(x)
    l = np.asarray(l)
    return l[0] + l[1] * (softplus(-x) - softplus(-x + l[2]))


class ShallowNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ShallowNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)


def NND(x, y, g=1, numhidden=10):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    n = x.shape[1]
    m = y.shape[1]

    input_data = torch.cat((x, y), dim=1).T
    labels = torch.cat((torch.ones(n, 1), torch.zeros(m, 1))).squeeze()

    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = input_data.shape[1]
    D_sum = torch.zeros(n + m).to(device)

    for i in range(g):
        model = ShallowNN(input_dim=input_dim, hidden_dim=numhidden).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        model.train()
        input_data = input_data.to(device)
        labels = labels.to(device)
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(input_data).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            D_sum += model(input_data).squeeze()

    D_avg = D_sum / g
    v = torch.mean(torch.log(D_avg[:n])) + torch.mean(torch.log(1 - D_avg[n:]))
    return v.item()
