from functools import lru_cache
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray

@lru_cache(maxsize=1)
def layer_sum_kernal():
    pass