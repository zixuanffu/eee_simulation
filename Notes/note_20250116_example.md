Consider
$$Z \sim \text{logistic}(0,1)$$
$$G_{\theta}(Z) = \theta + Z \Leftrightarrow p_g(x) = \Lambda(x-\theta) (1-\Lambda(x-\theta))$$
$$p_r(x) = \Lambda(x) (1-\Lambda(x))$$ 

This is correctly specified model!

Closed form for the optimal discriminator $D_\theta$!

The generator estimator is efficient!

Yet of course, in reality, the discriminator doesn't know the true distribution $p_r$ nor the $p_g$. We need to train the discriminator.

Let's impose a form of the discriminator. 

$$D(x;\lambda) = \Lambda(\lambda_0 -2\log(1+e^{-x}+2\log(1+e^{-x+\lambda_1}))$$ 
The discriminator is parametrized by $\lambda = (\lambda_0, \lambda_1)$

The training of the discriminator boils down to finding the optimal $\lambda$ that maminizes the entropy.

This is essentially a nuisance parameter because we don't really care about $\lambda$.

- Given a $\theta^1$, we train the discriminator to find the optimal $\lambda^1$. We denote this discriminator as $\hat{D}_{\theta^1}$ while the oracle discriminator as $D_{\theta^1}$.
- With $\lambda^1$, we train the generator to find the optimal $\theta^2$ so on and so forth.

We plot the entropy for each $\theta$, using 
- the oracle discriminator $D_{\theta}$,
- the trained discriminator $\hat{D}_{\theta}$,

They are very close to each other.

Remark:
for a logistic discriminator, and a **differentiable** $G_{\theta}$, the estimated entropy $M_\theta(\hat{D}_{\theta})$ will be smooth in $\theta$ if 
- $Z_i$ is fixed 
- the exact maximum is attained at the inner step.
  

Things to read:

- empirical likelihood vs GMM
- estimation in SMM. why does it perform bad when the number of moments is large?
- pseudo mle vs wrongly specified model.
- how to calculate theoretical standard deviation for adversarial estimator?
  

Questions:
- what is empirical likelihood estimation and how does it compare to GMM? (Imbens, 2002)
- when we have oracle d, then asymptotically equivalent to MLE?
- when we have logistcis d, then asymptotically equivalent to SMM?

