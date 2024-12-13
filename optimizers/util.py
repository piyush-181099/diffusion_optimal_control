import torch
from torch.utils._pytree import tree_map

def inv(A, eps=1e-7):
    """
    Convenience function for computing the inverse of a function A with Tikhonov
    regularization constant eps.
    
    Args:
        A:     matrix to be inverted
        eps:   Tikhonov regularization constant added to the diagonal of A 
               before inversion
    Returns:
        A_inv: the inverse of A
    """
    extra_dims = A.shape[:-2]
    k = min(*A.shape[-2:])
    eye = torch.diag_embed(torch.ones(size=(*extra_dims, k), device=A.device)) * eps
    A_inv = torch.inverse(A[..., :k, :k] + eye)
    return A_inv
    
def get_Q(dims, k, device):
        """
        Computes a random projection matrix Q given batch / time step dimensions dims
        and projection dimension k.
        
        Args:
            dims:   list[int]
            k:      int
            device: torch.device
            
        Returns:
            Q: tensor of shape (k, *dims)
        """
        
        if not isinstance(dims, tuple):
            dims = tuple(dims)
        
        if dims[-1] > k:
            R = torch.randn((*dims, k), device=device)
            Q, _ = torch.linalg.qr(R)
        else:
            *extra_dims, rows = dims
            Q = torch.eye(dims[-1], k, device=device)
            Q = Q.reshape([1 for _ in extra_dims] + [rows, k]).tile(extra_dims + [1, 1])
                
        return Q

def get_mjp_fn(fn, *xs, return_output=False, chunk_size=None, randomness='same'):
    """
    Returns a function computing the matrix-jacobian product of a function fn(*xs) 
    evaluated at its k inputs xs = (x1, x2, ..., xk)

    Args:
        fn:     torch differentiable function - the function to take the jacobian of
        xs:     tuple of inputs xs to fn()

    Returns: 
        mjp_fn: a new function computing the matrix jacobian product of fn w.r.t. a matrix m
    """
    output, vjp_fn = torch.func.vjp(fn, *xs)
    
    def mjp_fn(ms):
        """
        Computes the matrix-jacobian product of fn w.r.t. to the matrix m.
        
        Args:
            m:          a matrix of shape (fn(*xs).shape) + (k,)
            chunk_size: the number of columns of m to compute at a time (e.g. batch size)
            randomness: how to deal with randomness in fn across batches ('error', 'same',
                        'different')
        Returns:
            mjp:        the matrix-jacobian product
        """
        vmap_vjp_fn = torch.vmap(vjp_fn, in_dims=-1, out_dims=1, 
                                 randomness=randomness, chunk_size=chunk_size)
                
        return vmap_vjp_fn(ms)
    
    if return_output:
        return output, mjp_fn
    
    return mjp_fn
  
def get_jmp_fn(fn, *xs, fallback=False, chunk_size=None, randomness='same'):
    """
    Returns a function computing the jacobian-matrix product of a function fn(*xs) 
    evaluated at its k inputs xs = (x1, x2, ..., xk)

    Args:
        fn:     torch differentiable function - the function to take the jacobian of
        xs:     tuple of inputs xs to fn()

    Returns: 
        mjp_fn: a new function computing the matrix jacobian product of fn w.r.t. a matrix m
    """
    if fallback:
        jvp_fn = fallback_jvp(fn, *xs)
    else:
        jvp_fn = lambda ms: torch.func.jvp(fn, xs, ms)[1]
    
    def jmp_fn(ms):
        """
        Computes the jacobian-matrix product of fn w.r.t. to the matrix m.
        
        Args:
            m:          a matrix of shape (x.shape for x in xs) + (k,)
            chunk_size: the number of columns of m to compute at a time (e.g. batch size)
            randomness: how to deal with randomness in fn across batches ('error', 'same',
                        'different')
        Returns:
            jmp:        the matrix-jacobian product
        """
        vmap_jvp_fn = torch.vmap(jvp_fn, in_dims=-1, out_dims=-1, 
                                 randomness=randomness, chunk_size=chunk_size)
                  
        return vmap_jvp_fn(ms)

    return jmp_fn
  

def fallback_jvp(fn, *xs):
    """
    If forward mode AD is not implemented, we can rely on a fallback algorithm that produces a jvp computation
    at the cost of two vjp computations (i.e., twice the cost of the original algorithm, but without requiring
    specialized forward mode AD code). See:
    https://j-towns.github.io/2017/06/12/A-new-trick.html
    """
    # step 1. construct vjp_fn, which computes the vjp w.r.t. x: v^T d fn / dx (x)
    u, vjp_fn = torch.func.vjp(fn, *xs)
    
    if not isinstance(u, tuple):
        u = (u,)
        
    def jvp_fn(*vs):
        # step 2. construct vjp_v_fn, which computes the vjp w.r.t. v: v^T d fn / dx (v)
        _, vjp_v_fn = torch.func.vjp(vjp_fn, *u)

        return vjp_v_fn(*vs)
          
    return jvp_fn
    
  
  
# --------------------------------------------------------- #
#                       UNIT TESTING CODE                   #
# --------------------------------------------------------- #

def get_jacobian(fn):
    """
    Returns a function computing the jacobian of a function fn(*xs) evaluated at its 
    k inputs xs = (x1, x2, ..., xk)

    Args:
        fn:     torch differentiable function - the function to take the jacobian of

    Returns: 
        j_fn:    a new function computing the jacobian of fn
    """
    # for converting and unconverting batch dimension into vmap dimension
    add_vmap_dim = lambda pytree: tree_map(lambda x: x.unsqueeze(1), pytree)
    rm_vmap_dim = lambda pytree: tree_map(lambda x: x.squeeze(1), pytree)
    def j_fn(*xs):
        argnums = tuple(i for i in range(len(xs)))
        # torch.vmap: vmap(func) returns a new function that maps func over some dimension of the inputs.
        vmap_jacobian = torch.vmap(torch.func.jacrev(fn, argnums=argnums))
        # argnums: Optional, integer or tuple of integers, saying which arguments to get the Jacobian with respect to. Default: 0.
        
        xs = add_vmap_dim(xs)
        jacobian = rm_vmap_dim(vmap_jacobian(*xs))
        
        return jacobian if len(xs) > 1 else jacobian[0]
      
    return j_fn
  
def true_jacobian(fn, x):
    """
    Ground truth jacobian function. Only works for single input / output
    functions.
    """
    js = []
    for i in range(len(x)):
        j = torch.autograd.functional.jacobian(fn, x[i:i+1])
        js.append(j.squeeze())
    return torch.stack(js)
  
class Function(torch.nn.Module):
    """
    A function f: R^in_dim -> R^out_dim. Could be nonlinear, depending on the flag.
    """
    def __init__(self, in_dim, out_dim, nonlinear=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinear = nonlinear
        self.w = torch.randn((out_dim, in_dim))
        if nonlinear:
            self.w2 = torch.randn((out_dim, out_dim))
              
    def forward(self, x):
        x = x.reshape(-1, self.in_dim, 1)
        x = self.w @ x
        if self.nonlinear:
            self.w2 @ torch.nn.functional.relu(x)
        return x.reshape(-1, self.out_dim)
      
class MultiIOFunction(torch.nn.Module):
    """
    A multi-input and multi-output function f: (R^in_dim)^n_inputs -> (R^out_dim)^n_outputs.
    In other words, maps a n_inputs-tuple of R^in_dim variables to an n_outputs-tuple of
    R^out_dim variables. Could be nonlinear, depending on the flag.
    """
    def __init__(self, n_inputs, n_outputs, in_dim, out_dim, nonlinear=False):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.out_dim = out_dim
        self.fn = Function(in_dim, out_dim * n_outputs, nonlinear)
        self.ws = [self.fn.w[i * out_dim:(i + 1) * out_dim] for i in range(n_outputs)]
        
    def forward(self, *xs):
        x = torch.sum(torch.stack(xs), dim=0)
        x = self.fn(x)
        return torch.split(x, self.out_dim, dim=1)
  
def test_jacobian(n=8, in_dim=784, out_dim=10, nonlinear=False):
    """
    Verify that the jacobian function is correct.
    """
    x = torch.randn((n, in_dim))
    fn = Function(in_dim, out_dim, nonlinear)
    out = get_jacobian(fn)(x)
    
    w_hat = get_jacobian(fn)(x).reshape(n, out_dim, in_dim)
    w_true = true_jacobian(fn, x).reshape(n, out_dim, in_dim)
    assert torch.allclose(fn.w, w_hat)
    assert torch.allclose(w_true, w_hat)
    
def test_mjp(n=10, k=5, in_dim=784, out_dim=10, nonlinear=False):
    """
    Verify that the matrix-jacobian product function is correct.
    """
    x = torch.randn((n, in_dim))
    fn = Function(in_dim, out_dim, nonlinear)
    m = torch.randn((n, k, out_dim))
    
    mjp_true = m @ fn.w.reshape(1, out_dim, in_dim)
    mjp_hat = get_mjp_fn(fn, x)(m.mT)[0]
    assert torch.allclose(mjp_true, mjp_hat)
      
def test_multi_jacobian(n=8, n_inputs=2, n_outputs=2, input_dim=784, output_dim=10):
    """
    Verify that the jacobian function is correct for multi-input /-output functions.
    """
    multifunc = MultiIOFunction(n_inputs, n_outputs, input_dim, output_dim)
    xs = tuple(torch.randn((n, input_dim)) for _ in range(n_inputs))
    j = get_jacobian(multifunc)(*xs)
    for input_idx in range(n_inputs):
        for output_idx in range(n_outputs):
            j_hat = j[output_idx][input_idx].reshape(n, output_dim, input_dim)
            j_true = multifunc.ws[output_idx]
            assert torch.allclose(j_hat, j_true)
            
def test_multi_mjp(n=8, k=2, n_inputs=2, n_outputs=2, input_dim=784, output_dim=10):
    """
    Verify that the matrix-jacobian product function is correct for 
    multi-input /-output functions.
    """
    multifunc = MultiIOFunction(n_inputs, n_outputs, input_dim, output_dim)
    xs = tuple(torch.randn((n, input_dim)) for _ in range(n_inputs))
    vs = tuple(torch.randn((n, output_dim, k)) for _ in range(n_outputs))
    mjp_fn = get_mjp_fn(multifunc, *xs)
    j_fn_vs = mjp_fn(vs)

    j_true = sum([(vs[i].mT @ multifunc.ws[i]) for i in range(n_outputs)])
    for input_idx in range(n_inputs):
        assert torch.allclose(j_fn_vs[input_idx], j_true, atol=1e-5)
  
if __name__ == '__main__':
    # test linear jacobian correctness
    test_jacobian()
    test_mjp()
    
    # test nonlinear jacobian correctness
    test_jacobian(nonlinear=True)
    test_mjp(nonlinear=True)
    
    # test multi-input and multi-output linear function correctness
    test_multi_jacobian()
    test_multi_mjp()
    print("All unit tests passed!")
