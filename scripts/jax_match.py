#%%
import jax.numpy as jnp
import graspologic as gl
import numpy as np

#%%
n = 10
p = 0.3
rho = 0.9
A1, B1 = gl.simulations.er_corr(n, p, rho)
A2, B2 = gl.simulations.er_corr(n, p, rho)

A = np.stack([A1, A2])
B = np.stack([B1, B2])
A = jnp.array(A)
B = jnp.array(B)
#%%
from jax import jit



def objective_func(P):
    n_layers = A.shape[0]
    obj_value = 0
    for layer in range(n_layers):
        obj_value += jnp.linalg.norm(A[layer] - P @ B[layer] @ P.T) ** 2
    return obj_value


P = np.full((n, n), 1 / n)

objective_func(P)

#%%
import jax
import jax.scipy.optimize as jopt

objective_gradient = jax.grad(objective_func)


grad_at_P = objective_gradient(P)
from scipy.optimize import linear_sum_assignment

row_inds, col_inds = linear_sum_assignment(grad_at_P)

Q = np.eye(*P.shape)
Q = Q[:, col_inds]


def combination_objective(alpha):
    return objective_func(alpha[0] * P + (1 - alpha[0]) * Q)


combination_objective_gradient = jax.grad(combination_objective)

jopt.minimize(combination_objective, jnp.array([0.5]), method="BFGS")

#%%


def projection(P, *args):
    _, cols = linear_sum_assignment(P)
    Q = jnp.eye(*P.shape)
    Q = Q[:, cols]
    return Q


from jaxopt import ProjectedGradient

pg = ProjectedGradient(objective_func, projection)
pg.run(jnp.array(P))

# %%
