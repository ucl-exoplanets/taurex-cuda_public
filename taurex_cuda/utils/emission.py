from functools import lru_cache
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray
import numpy as np
import math
import pycuda.driver as drv

@lru_cache(maxsize=1)
def _blackbody_module():
    from taurex.constants import PLANCK, SPDLIGT, KBOLTZ, PI
    code = f"""
    
    __global__ void black_body(double* dest,
                               const double* __restrict__ lamb,
                               const double* __restrict__ temperature,
                               const int grid_length, const int nlayers)
    {{
            unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
            unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
            
            if ( i >= grid_length )
                return;
            
            if ( j >= nlayers )
                return;     
            
            double wl = 10000*1e-6/lamb[i];
            double wl_5 = wl*wl*wl*wl*wl;
            double exponent = exp(({PLANCK * SPDLIGT}) / (wl * {KBOLTZ} * temperature[j]));
            double bb = ({PI*2.0*PLANCK*SPDLIGT**2}/(wl_5)) * (1.0/(exponent-1));
            dest[j*grid_length + i] = bb*1e-6;
    }}
    """
    mod = SourceModule(code)
    return mod.get_function('black_body')

def cuda_blackbody(lamb, temperature, out=None):
    my_out = out
    if my_out is None:
        my_out = GPUArray(shape=(temperature.shape[0],lamb.shape[0]),dtype=np.float64)
    
    grid_length = lamb.shape[0]
    nlayers = temperature.shape[0]
    
    THREAD_PER_BLOCK_X = 16
    THREAD_PER_BLOCK_Y = 16

    NUM_BLOCK_X = int(math.ceil(grid_length/THREAD_PER_BLOCK_X))
    NUM_BLOCK_Y = int(math.ceil(nlayers/THREAD_PER_BLOCK_Y))   
    
    _blackbody_module()(my_out,drv.In(lamb), drv.In(temperature),np.int32(grid_length),np.int32(nlayers),
                        block=(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y,1), grid=(NUM_BLOCK_X, NUM_BLOCK_Y,1))
    return my_out