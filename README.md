# Randomized (low rank) Hessian approximation of JAX function

Sample a few Gaussian vectors and project the Hessian on it to get an approximation of the Hessian

```python
import numpy as onp
import jax.numpy as np

from randomized_hessian import approximate_hessian

# create a simple quadratic function
dim = 5
S = onp.random.normal(0,1,size=(dim,dim))
S = (S + S.T) / 2.
def func(x):
    return 0.5 * np.dot(x, S @ x)

# approximate its Hessian at u
u = onp.random.normal(0,1,size=(dim))
rank_approx = 6
H_approx = approximate_hessian(func, u, rank_approx)
```
