## Figures

### Likelihood
1. MLE (true)
2. Oracle D (true)
3. Correctly specified D (trained)
4. Neural network D (trained) 
5. Misspecified D (true and trained)
6. quasi-MLE (true)

- Figure 1: 1 & 2 & 3
- Figure 2: 1 & 2 & 4
- Figure 3: 6 & 5 (the misspecified D has two versions: the truly misspecified and the trained)
- I want to compare: 2 & 3 & 4
### Moments
1. Oracle true D (What is the oracle D?) (true)
2. Misspecified D (The logistic location model with increasing numbers of inputs. The curvature of the cross-entropy loss is very close to the log likelihood up to 7 moments and is still good for 11 moments.) (trained)
3. SMM loglikelihood (true)
4. Neural network D (trained) (to be trained haha)

- Figure 4: 1 & 2 & 3
- I want to compare: 1 & 4 & 3

## Code

```python
# Example Usage
x = np.random.rand(5, 100)  # Actual data (5 features, 100 samples)
y = np.random.rand(5, 100)  # Simulated data (5 features, 100 samples)
v = NND(x, y, g=5, numhidden=10)
print("Cross-entropy loss value:", v)
```