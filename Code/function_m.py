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
    # Ensure x is an array for consistent processing
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    # Ensure valid scale parameter
    if np.any(sigma <= 0):
        raise ValueError("Scale parameter 'sigma' must be positive.")

    # Ensure inputs are real
    if not np.isreal(x).all() or not np.isreal(mu).all() or not np.isreal(sigma).all():
        raise ValueError("Inputs must be real.")

    # Adjust for complementary CDF
    if uflag is not None and uflag.lower() == 'upper':
        x = -x
        mu = -mu

    # Calculate the logistic CDF
    y = 1 / (1 + np.exp(-(x - mu) / sigma))

    # Handle edge cases for division by zero
    zero_mask = (x == mu) & (sigma == 0)
    if np.isscalar(y):
        if zero_mask:
            y = 1
    else:
        y[zero_mask] = 1

    # Return scalar if input is scalar, otherwise array
    return y.item() if np.isscalar(x) else y

def logisticrnd(m=0, s=1, *size):
    """
    Generate logistic random numbers.
    
    Parameters:
    m : float or array_like, optional
        Location parameter (default is 0).
    s : float or array_like, optional
        Scale parameter (default is 1).
    size : int or tuple of ints, optional
        Size of the output array (default is scalar if no size is provided).
        
    Returns:
    ndarray
        Logistic random numbers.
    """
    # if np.any(np.asarray(s) <= 0):
    #     raise ValueError("Scale parameter 's' must be positive.")
    # if not np.isreal(m) or not np.isreal(s):
    #     raise ValueError("Input parameters must be real numbers.")

    if len(size) == 0:
        size = np.broadcast(np.array(m), np.array(s)).shape
    elif len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])

    u = np.random.rand(*size)
    return logisticinv(u, m, s)

def logisticinv(u, m, s):
    """
    Inverse CDF of the logistic distribution.
    
    Parameters:
    u : array_like
        Uniform random numbers in [0, 1].
    m : float or array_like
        Location parameter.
    s : float or array_like
        Scale parameter.
        
    Returns:
    ndarray
        Logistic random numbers.
    """
    return m + s * np.log(u / (1 - u))



def loglogisticpdf(x, mu=0, sigma=1):
    """
    Log of logistic probability density function (pdf).
    
    Parameters:
    x : array_like
        Values at which to evaluate the pdf.
    mu : float or array_like, optional
        Location parameter (default is 0).
    sigma : float or array_like, optional
        Scale parameter (default is 1).
    
    Returns:
    y : ndarray
        Log pdf values.
    """
    # Ensure valid scale parameter
    sigma = np.asarray(sigma)
    if np.any(sigma <= 0):
        raise ValueError("Scale parameter 'sigma' must be positive.")

    # Compute z
    z = (x - mu) / sigma

    # Compute log pdf
    y = -np.log(sigma) - softplus(-z) - softplus(z)
    return y

def softplus(x):
    """
    Smooth approximation to the rectifier function.
    Computes log(1 + exp(x)).
    
    Parameters:
    x : array_like
        Input values.
    
    Returns:
    ndarray
        Softplus-transformed values.
    """
    return np.log1p(np.exp(x))

def loglogisticcdf(x, mu=0, sigma=1, uflag=None):
    """
    Log of logistic cumulative distribution function (cdf).
    
    Parameters:
    x : array_like
        Values at which to evaluate the cdf.
    mu : float or array_like, optional
        Location parameter (default is 0).
    sigma : float or array_like, optional
        Scale parameter (default is 1).
    uflag : str, optional
        Specify 'upper' for the complementary cdf (default is None).
    
    Returns:
    y : ndarray
        Log cdf values.
    """
    # Determine if complementary cdf ('upper') is specified
    upper = uflag is not None and uflag.lower() == 'upper'

    # Ensure valid scale parameter
    sigma = np.asarray(sigma)
    if np.any(sigma <= 0):
        raise ValueError("Scale parameter 'sigma' must be positive.")

    if not np.isreal(x).all() or not np.isreal(mu) or not np.isreal(sigma).all():
        raise ValueError("Inputs must be real.")

    # Adjust for complementary cdf
    if upper:
        x = -x
        mu = -mu

    # Compute log cdf
    y = -softplus(-(x - mu) / sigma)
    return y

def softplus(x):
    """
    Smooth approximation to the rectifier function.
    Computes log(1 + exp(x)).
    
    Parameters:
    x : array_like
        Input values.
    
    Returns:
    ndarray
        Softplus-transformed values.
    """
    return np.log1p(np.exp(x))



def logaddexp(x, y):
    """
    Calculates log(exp(x) + exp(y)).
    Minimizes overflow or underflow during computation.
    
    Parameters:
    x : array_like
        First input.
    y : array_like
        Second input.
    
    Returns:
    z : ndarray
        Result of log(exp(x) + exp(y)).
    """
    x, y = np.asarray(x), np.asarray(y)  # Ensure inputs are arrays
  # Compute the element-wise maximum of x and y
    m = np.maximum(np.real(x), np.real(y))
    # Handle infinite values in m
    if np.isscalar(m):  # If m is a scalar
        if np.isinf(m):
            m = 0
    else:  # If m is an array
        m[np.isinf(m)] = 0
    # Compute the numerically stable log(exp(x) + exp(y))
    z = np.log(np.exp(x - m) + np.exp(y - m)) + m

    return z

def loss(X1, X2):
    """
    Classification accuracy with logistic discriminator.
    
    Parameters:
    X1 : array_like
        First set of inputs (data).
    X2 : array_like
        Second set of inputs (data).
    
    Returns:
    v : float
        The minimized loss value (negative of the result).
    beta : ndarray
        The optimized parameters of the nonlinear index function.
    """
    # Define the objective function
    def objective(l):
        return (np.mean(softplus(indexf(X1, l))) +
                np.mean(softplus(-indexf(X2, l))))

    # Initial guess for optimization
    initial_guess = np.array([1, 2, 1])

    # Optimize using scipy's minimize (equivalent to MATLAB's fminsearch)
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    beta = result.x
    v = -result.fun  # Negate the result to match MATLAB's behavior

    return v, beta

def indexf(x, l):
    """
    Nonlinear index for correctly specified discriminator.
    
    Parameters:
    x : array_like
        Input values.
    l : array_like
        Parameters of the index function.
    
    Returns:
    idx : ndarray
        Computed index values.
    """
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
    """
    Shallow neural network discriminator with cross-entropy loss.

    Parameters:
    x : ndarray
        Actual data (d x n).
    y : ndarray
        Simulated data (d x m).
    g : int, optional
        Number of neural networks to average over (default = 1).
    numhidden : int, optional
        Number of nodes in the hidden layer (default = 10).

    Returns:
    v : float
        Value of the maximized cross-entropy loss.
    """
    # Convert inputs to tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Sample sizes
    n = x.shape[1]
    m = y.shape[1]

    # Combined data and labels
    input_data = torch.cat((x, y), dim=1).T  # Transpose to match PyTorch convention (samples x features)
    labels = torch.cat((torch.ones(n, 1), torch.zeros(m, 1))).squeeze()

    # Cross-entropy loss
    criterion = nn.BCELoss()

    # Initialize and train g networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = input_data.shape[1]
    D_sum = torch.zeros(n + m).to(device)

    for i in range(g):
        model = ShallowNN(input_dim=input_dim, hidden_dim=numhidden).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Training
        model.train()
        input_data = input_data.to(device)
        labels = labels.to(device)
        for epoch in range(100):  # Train for 100 epochs (adjust as needed)
            optimizer.zero_grad()
            outputs = model(input_data).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Collect discriminator output
        model.eval()
        with torch.no_grad():
            D_sum += model(input_data).squeeze()

    # Average discriminator outputs
    D_avg = D_sum / g

    # Compute the cross-entropy loss value
    v = torch.mean(torch.log(D_avg[:n])) + torch.mean(torch.log(1 - D_avg[n:]))
    return v.item()

