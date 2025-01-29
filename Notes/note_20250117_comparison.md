
Let's say the true distribution (pdf) is logistic

$$ p_r(x) = \Lambda(x) (1-\Lambda(x))$$

we specify the model to be normal
$$ p_g(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}(x-\theta)^2\right)$$

We generate data from the generator of the form $G_{\theta}(Z) = \theta + Z$ where $Z \sim \text{logistic}(0,1)$. Note that $E_Z[G_{\theta}(Z)] = \theta$.

Now we compare the **curvature** of the objective function in three different methods. 

### MLE
In MLE, we minimize 
$$\min_\theta L_\theta =  -\frac{1}{2n} \sum_{i=1}^n \log p(x_i;\theta)$$
where 
$$p(x;\theta) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}(x-\theta)^2\right)$$

### Adversarial estimator
In Adversarial, we minimize
$$ \min_\theta M_\theta(D) = \frac{1}{n} \sum_{i=1}^n \log D(x_i) + \frac{1}{m} \sum_{i=1}^m \log (1-D(G_{\theta}(Z_i)))$$

In the case of an oracle discriminator,
$D_\text{oracle}(x) = \frac{p_r(x)}{p_r(x) + p_g(x)} = \Lambda(\log(\sqrt{2\pi})-x+\frac{1}{2}(x-\theta)^2-2\log(1+e^{-x}))$

In the case of correctly specified disctriminator yet to be trained $D(x;\lambda)$ is
$$D(x;\lambda) = \Lambda(\lambda_0 +\lambda_1 x + \lambda_2 x^2+\lambda_3 \log(1+e^{-x}))$$ 
parametrized by $\lambda \in \mathbb{R}^4$

In the case of non-parametric discriminator (shallow neural network), we obviously don't have a closed form of the $D$ but the neural network is described by one input layer, one hidden layer with three nodes, and one output layer (Check description of neural network).

Given a $\theta$ denote the oracle discriminator as $D_{\theta}$, and the correctly specified trained discriminator by $\hat{D}_{\theta}$.

### SMM
- $Y_i, X_i$ are observed data
- $W_i$ is the instruments.
- $Z_i$ is the input (usually uniform), here we assume standard logistic distribution (which is kind of abnormal, no?)

#### General case 
Recall that $f(Y_i,X_i|\theta)$ is some moment condition such that 
$$ \mathbb{E}_{Y_i|X_i}(f(Y_i,X_i|\theta)|X_i) = 0$$
This $f$ may be hard to calculate. Therefore, we want to replace this $f$ with a generator $G$ such that
$$\mathbb{E}_{Z}(G(Y_i,X_i,Z|\theta)|Y_i,X_i) = f(Y_i,X_i|\theta)$$
Therefore the theoretical moment condition we will be using is the following
$$ \mathbb{E}_{Y_i}(\mathbb{E}_{Z}(G(Y_i,X_i,Z|\theta)|Y_i,X_i)\times W_i) = 0$$
In SMM, we minimize
$$ \min_\theta S_\theta = \left\{\frac{1}{n}\sum_i^n W_i \frac{1}{s} \sum_{j=1}^s G(Y_i,X_i,Z_j|\theta)\right\}^\top \Omega \left\{\frac{1}{n} \sum_i^n W_i \frac{1}{s} \sum_{j=1}^s G(Y_i,X_i,Z_j|\theta)\right\}
    $$ 
The generated sample size $m = ns$ is much larger than the observed sample size $n$.

#### One dimensional case
We only have one observation which is $X_i$, there's no **relationship** between $Y_i$ and $X_i$ that we want to discover. We only want to know something about the distribution of $X_i$. Therefore, we can simplify the moment condition to be
$$\mathbb{E}_{X}(f(X|\theta)) = \theta$$
where $f(X|\theta)$ is some moment condition.
We replace this $\theta$ with a generator $G$ such that
$$\mathbb{E}_{Z}(G(Z|\theta)) = \theta $$
Therefore, the theoretical moment condition we will be using is the following
$$ \mathbb{E}_{X}(f(X|\theta)-\mathbb{E}_{Z}(G(Z|\theta))) = 0$$
In SMM, we minimize
$$ \min_\theta S_\theta = \left\{\frac{1}{n}\sum_i^n \left\{f(X_i)-\frac{1}{m} \sum_{j=1}^m G(Z_j|\theta)\right\} \right\}^\top \Omega \left\{\frac{1}{n} \sum_i^n \left\{f(X_i)-\frac{1}{m} \sum_{j=1}^m G(Z_j|\theta)\right\} \right\}$$
Or equivalently
$$ \min_\theta S_\theta = \left\{\frac{1}{n}\sum_i^n f(X_i) - \frac{1}{m} \sum_{j=1}^m G_\theta(Z_j) \right\}^\top \Omega \left\{\frac{1}{n} \sum_i^n f(X_i) - \frac{1}{m} \sum_{j=1}^m G_\theta(Z_j) \right\}$$
where real data has size $n$ and generated data has size $m$.

Simplest case is to match moments of different order. In this case,

$$f(X)=X^\text{order} \quad G(Z|\theta) = \theta + Z$$

Note that one should estimate the optimal weighting matrix $\Omega$ in SMM.

### SML (Simulated Maximum Likelihood)

The likelihood function is
$$\min_\theta L_\theta =  -\frac{1}{2n} \sum_{i=1}^n \log p(x_i;\theta)$$

Since we don't have a closed form of the likelihood function, we simulate it by 
$$ E_Z[f(x_i,G(Z);\theta)]=p(x_i;\theta) $$
where $p(x_i;\theta)$ is the likelihood of observing the real data and $f(x_i,G(Z);\theta)$ is the likelihood of observing the real data **and** the generated data.

Then the objective function is 
$$\min_\theta L^s_\theta =  -\frac{1}{2n} \sum_{i=1}^n \log \left[\frac{1}{m} \sum_{j=1}^m f(x_i,G(Z_j);\theta)\right]$$

*Remark*: an unbiased estimator of likelihood $p$ is not an unbiased estimator of the log-likelihood $\log p$.
Now let's compare the three objective function 
#### MLE
$$ \min_\theta L_\theta =  -\frac{1}{2n} \sum_{i=1}^n \log p(x_i;\theta)$$

  - in mispecified case: $p_\text{mis}(x;\theta) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}(x-\theta)^2\right)$

- correctly specified case: $p_\text{true}(x;\theta) = \Lambda(x-\theta) (1-\Lambda(x-\theta))$
#### Adversarial
$$ \min_\theta M_\theta(D) = \frac{1}{n} \sum_{i=1}^n \log D(x_i) + \frac{1}{m} \sum_{j=1}^m \log (1-D(G_{\theta}(Z_j)))$$
- in correctly specified case of $p_g$: 
  - oracle $D(x) = \Lambda(-\theta-2log(1+e^{-x})+2\log(1+e^{-x+\theta}))$
  - correctly specified $D(x;\lambda)  = \Lambda(\lambda_0-2\log(1+e^{-x})+2\log(1+e^{-x+\lambda_1}))$
  - non-parametric $D$ (shallow neural network)
- in wrongly specified case of $p_g$: 
  - oracle $D(x) = \Lambda(\log(\sqrt{2\pi})-x+\frac{1}{2}(x-\theta)^2-2\log(1+e^{-x}))$
  - correctly specified $D(x;\lambda) = \Lambda(\lambda_0 +\lambda_1 x + \lambda_2 x^2+\lambda_3 \log(1+e^{-x}))$
  - non-parametric $D$ (shallow neural network)
#### SMM
$$ \min_\theta S_\theta = \left\{\frac{1}{n}\sum_i^n f(X_i) - \frac{1}{m} \sum_{j=1}^m G_\theta(Z_j) \right\}^\top \Omega \left\{\frac{1}{n} \sum_i^n f(X_i) - \frac{1}{m} \sum_{j=1}^m G_\theta(Z_j) \right\}$$
- equal weight
- optimal weight
