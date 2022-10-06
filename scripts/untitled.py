from numba import cuda

@cuda.jit(device=True)
def fun_jit_gpu(x, params):
    """Bisection function"""

    return scalar


@cuda.jit(device=True)
def bisect_jit_gpu(x, params):
    
    return scalar


@cuda.jit
def simplex_jit_gpu(X, params, Y):
    
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if i < x.shape[0]: 

        fun_jit_gpu(X[i], params) 
            
        c = bisect_jit_gpu(X[i], params)
        
        for j in range(coeffs.shape[0]):
            y[i][j] = min( max(x[i][j]))