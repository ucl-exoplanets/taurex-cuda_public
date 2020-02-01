from .cudacontribution import CudaContribution
import numpy as np
from ..opacity.cudacache import CudaOpacityCache
from pycuda.gpuarray import GPUArray, to_gpu, zeros
from pycuda.tools import DeviceMemoryPool
from pycuda.compiler import SourceModule
from functools import lru_cache
import math
from taurex.core import fitparam

def _cloud_kernal(grid_size):

    code = f"""

    __global__ void simple_cloud(double* dest, const int end_layer)
    {{
        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
        unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (i >= {grid_size})
            return;
        
        if (j >= end_layer)
            return;
        
        dest[j*{grid_size} + i] = INFINITY;
    }}
    """
    mod = SourceModule(code)
    return mod.get_function('simple_cloud')


class SimpleCloudsCuda(CudaContribution):
    """
    Computes the contribution to the optical depth
    occuring from molecular Rayleigh.
    """

    def __init__(self, clouds_pressure=1e3):
        super().__init__('SimpleClouds')
        self._memory_pool = DeviceMemoryPool()
        self._cloud_pressure = clouds_pressure
    def build(self, model):
        super().build(model)

    def prepare_each(self, model, wngrid):
        """
        Prepares each molecular opacity by weighting them
        by their mixing ratio in the atmosphere

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            Name of molecule and weighted opacity

        """

        self.debug('Preparing model with %s', wngrid.shape)
        self._ngrid = wngrid.shape[0]

        self._sigma_xsec = zeros(shape=(model.nLayers, wngrid.shape[0]), dtype=np.float64, allocator=self._memory_pool.allocate)

        end_layer = max(0,np.argmax(model.pressureProfile < self._cloud_pressure))
        cloud_kernel = _cloud_kernal(self._ngrid)

        THREAD_PER_BLOCK_X = 16
        THREAD_PER_BLOCK_Y = 16
        
        NUM_BLOCK_X = int(math.ceil(self._ngrid/THREAD_PER_BLOCK_X))
        NUM_BLOCK_Y = int(math.ceil(model.nLayers/THREAD_PER_BLOCK_Y))

        cloud_kernel(self._sigma_xsec,np.int32(end_layer),
            block=(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y,1), grid=(NUM_BLOCK_X, NUM_BLOCK_Y,1) )

        yield 'Clouds', self._sigma_xsec


    @fitparam(param_name='clouds_pressure',
              param_latex='$P_\mathrm{clouds}$',
              default_mode='log',
              default_fit=False, default_bounds=[1e-3, 1e6])
    def cloudsPressure(self):
        """
        Cloud top pressure in Pascal
        """
        return self._cloud_pressure

    @cloudsPressure.setter
    def cloudsPressure(self, value):
        self._cloud_pressure = value



    @classmethod
    def input_keywords(cls):
        return ['SimpleCloudsCUDA', 'ThickCloudsCUDA' ]