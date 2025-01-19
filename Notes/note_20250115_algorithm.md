
In adversarial approach, we need to switch perspective.

## Discriminator
The first one to move. Given only data points. An oracle one would be such that it knows $p_r$ and $p_g$

# Ideas
## Generator
The second one to move. An oracle one would be such that it knows $p_r$, then it can perfectly generate $p_g = p_r$. However it does not. Given a certain discriminator $D$, it tries to find a $p_g$ from which to generate data points. When $p_g$ is parametric, the generator is trying to find the optimal $\theta^*$. 

Finding the optimal $G$ is equivalent to finding the optimal $p_g$, which is equivalent to finding the optimal $\theta^*$.

Thus, the generator is an estimator. Let's say the true parameter is $\theta_0$. The generator generates $\hat{\theta}$, which is an estimator of $\theta_0$. Consistency is when $\hat{\theta}\to \theta_0$ as $n\to \infty$. Let's say the generator somehow knows that the true distribution $p_r$ is indeed normal, characterized by $\theta=(\mu, \sigma)$, which it is trying to estimate. When the generator knows the true parametric **form**, we say the model is correctly specified. And if the given discriminator is indeed the oracle one, then the generator will give us $\hat{\theta}=\theta_0$.

## Proposition 1

### Oracle $D$
$$D^*_{\text{oracle}}=\frac{p_r(x)}{p_r(x)+p_g(x)}$$

Given **oracle discriminator** and the generator **knowing the form of the true distribution**, **and there are more generated data than real data**, the generator estimator is efficient. 

## Proposition 2

### Parametric $D$
If we don't have an oracle estimator but impose a logistic function form on the discriminator $D(x)$ for example $D(x|\beta)=\Lambda(\beta_0+\beta_1x)=\frac{1}{1+e^{-(\beta_0+\beta_1 x)}}$, the discriminator is the full log maximizing likelihood.
$$ \sum_i^{n+m} t_i\log D(x_i)+(1-t_i)\log(1-D(x_i))$$
And the optimal $D^*$ is characterized by the optimal $\beta^*$. 

Given **the optimal $D^*=\Lambda(\beta^{*T}x)$**, and **the generator knowing the form of the true distritbuion**, and **there are more fake data than real data**, the generator estimator is asymptotically **equivalent** to the optimally weihgted SMM with moments $E(X_i)$.

In practice, we use a **sieve** of logistic form $D_\text{logistic}$ to approximate the oracle $D_\text{oracle}$.

As we can see, the only difference between the two statements is the type of discriminator that we use, which gives rise to different asymptotic properties of the generator estimator.

- $D_\text{oracle}$: efficient
- $D_\text{logistic}$: equivalent to optimally weighted SMM
- $D_\text{neural}$: ???


# Algorithm
1. Pick an initial estimate $\theta_1$ so that **Generator** can generate $p_{g_1}$. 
2. Train the **Discriminator** $D_1$ to completion using the data points $\{x_i\}$ from $p_r$ and $\{x_{j}\}$ from $p_{g_1}$. We get a value of the entropy.
3. We take the gradient of the entropy with respect to $\theta_1$. Update the $\theta_1$ to $\theta_2$ to minimize the entropy.
4. Repeat 2-3 until convergence, where changes in $\theta$ are small and the loss function is bounded away from 0. 

**Note that the loss function is bounded between $0$ and $-\log2$.** When the discriminator is ignorant, the entropy is $-\log2$. When the discriminator is oracle and the generator is perfect $p_g=p_r$, the entropy is $-\log2$ as well. The idea is that if we have a dumb discriminator, the generator is also dumb and generates anything. If we have an oracle discriminator, then the generator is also trying to be perfect in mimicing the true distribution pushing the entropy to $-\log2$ again.

## Training $D$: Adam stochastic gradient descent with minibatch
Adam algorithm: combines the current and previous estimates of gradient to update and *information about the second moment*. Note that we don't have an oracle discriminator because we don't know $p_r$. 

If we impose a logistic form, then we will take the gradient of $\frac{1}{n}\sum_i \log(\Lambda(\beta x))+\frac{1}{m}\sum_j \log(1-\Lambda(\beta x))$
with respect to $x$ and update $\beta$.

In Adam algorithm, there are four tuning parameters.
- learning rate $l_r^D$
- exponential decay reates of the first moment $\beta_1$
- exponential decay rates of the second moment $\beta_2$
- a small constant $\epsilon$ to avoid division by zero.
## Training $G$: gradient descent with adaptive learning


## Issues with convergence

### Solution
1. feature matching: whether the generator's ouput matches some moments of the real sample. 
$$ E_{p_r}[f(x)]-E_{p_g}[f(x)]$$ 

2. Minibatch discrimination: discriminate in one batch rather than one point. we need to define a new function $c(x_i, x_j)$ that measures the similarity between $x_i$ and $x_j$. (?) what is the point of defining how close it is to other data points in the same batch?
3. Historical averaging
4. One-sided label smoothing: 
5. Virtual batch normalization: set a reference batch
6. Adding noise to the input of the discriminator to create artifical overlaps between the real  and generated data (between $p_r$ and $p_g$).
7. Better metric of distribution similarity: the loss function fo the the vanilla GAN $=2D_{JS}(p_r||p_m)-2\log2$ where $D_{JS}$ is the Jensen-Shannon divergence. This metric fails to provide a meaningful value when the two distributions are disjoint (infinity is not a good value...)


### Wasserstein GAN

Wasserstein distance: the earth mover's distance. The distance between two distributions is the minimum amount of work/energy cost required to transform one pile of dirt in the shape of $p_1$ to another pile of dirt in the shape of $p_2$. The cost is defined by amount of dirt moved $\times$ distance moved.

