
# Divergence between two distribution
- KL divergence: [Why is it non-negative?](https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative)
- JS divergnece:
- Wasserstein distance:
# Ignorant discriminator
$$ D(x) =\frac{1}{2}$$
$$ Loss = 2\log(1/2)$$
# Oracle discriminator

$$ D(x)=\frac{p_r(x)}{p_r(x)+p_g(x)}$$
$$ Loss= $$ 

# Question 
1. The first term doesn't not matter for $G$ in the minimization gradient descent. Yet minimizing with the oracle discriminator is shown to be almost equivalent to MLE while taking into account the first term.
2. To connect to MSM. Why is minimizing with a logistic discriminator equivalent to the case when the moment is $E(X_i)=0$ in MSM? To understand MSM first. 
This is really important to understand asymptotic distribution of this adversarial estimator.