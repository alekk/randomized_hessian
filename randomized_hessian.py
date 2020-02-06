#ordinary numpy
import numpy as onp

#jax
import jax
from jax import grad, hessian, jit, vmap
import jax.numpy as np


def create_hessian_vector_product_func(func):
    """ 
    args:
        func: function with a single argument
        output: function that implements the hessian-vector product
    """
    grad_func = jax.grad(func)
    auxiliary_func = lambda u,v: np.dot(grad_func(u), v)
    return jax.grad(auxiliary_func)


def randomized_hessian_approximation(hess_vec_func, u, rank_approx):
    """    
    args:
        hess_vec_func: implements hess_vec_func(x,y) = hessian(x) @ y
        u: vector where to approximate Hessian
    """
    hess_vec_func_u_fixed = lambda v: hess_vec_func(u,v)
    hess_vec_func_u_fixed_batch = jax.jit( jax.vmap(hess_vec_func_u_fixed, in_axes=0, out_axes=0) )

    ell = rank_approx
    dim_input = len(u)
    gauss_probing = onp.random.normal(0,1,size=(dim_input, ell)) / np.sqrt(dim_input)
    Y = hess_vec_func_u_fixed_batch(gauss_probing.T).T
    Q,R = onp.linalg.qr(Y)
    B = onp.array( hess_vec_func_u_fixed_batch(Q.T) )
    return Q @ Q.T @ B.T @ Q.T  #symmetric approximate Hessian


def approximate_hessian(func, u, rank_approx):
    """    
    args:
        func: univariate function with a single argument
        u: vector where to approximate Hessian
        rank_approx: rank of the approximation
    """
    hess_vec_func = create_hessian_vector_product_func(func)
    return randomized_hessian_approximation(hess_vec_func, u, rank_approx)


# # ====================
# # create a simple quadratic function
# dim = 5
# S = onp.random.normal(0,1,size=(dim,dim))
# S = (S + S.T) / 2.
# def func(x):
#     return 0.5 * np.dot(x, S @ x)

# # apprixmate its Hessian at u
# u = onp.random.normal(0,1,size=(dim))
# rank_approx = 6
# H_approx = approximate_hessian(func, u, rank_approx)

# # sanity check
# print(S - H_approx)

