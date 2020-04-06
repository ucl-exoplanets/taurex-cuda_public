from .cudacontribution import CudaContribution
import numpy as np
from ..opacity.cudacache import CudaOpacityCache
from pycuda.gpuarray import GPUArray, to_gpu
from pycuda.tools import DeviceMemoryPool
from pycuda.compiler import SourceModule
from functools import lru_cache
import math



def __rayleigh_kernal(molecule_computation_code):

    code = f"""

    __global__ void compute_rayleigh_sigma(double* __restrict__ dest, const double* __restrict__ wngrid,
                                       const double* __restrict__ mix, const int nlayers, const int grid_size)
    {{


        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
        unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (i >= grid_size)
            return;
        
        if (j >= nlayers)
            return;

        double _wn = wngrid[i];
        double _mix = mix[j];

        {molecule_computation_code}

        dest[j*grid_size + i] = result*_mix;
    }}

    """
    return code
@lru_cache(maxsize=None)
def get_rayleigh_kernal(molecule):
    from ..utils.scattering import _molecule_func_code
    try:
        molecule_code = _molecule_func_code[molecule]
    except KeyError:
        return None

    kernal_code = __rayleigh_kernal(molecule_code)
    mod = SourceModule(kernal_code)

    return mod.get_function('compute_rayleigh_sigma')



class RayleighCuda(CudaContribution):
    """
    Computes the contribution to the optical depth
    occuring from molecular Rayleigh.
    """

    def __init__(self):
        super().__init__('Rayleigh')
        self._memory_pool = DeviceMemoryPool()

    def build(self, model):
        super().build(model)
        self._mix_array = GPUArray(shape=(model.nLayers,), dtype=np.float)

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
        from taurex.util.scattering import rayleigh_sigma_from_name

        self.debug('Preparing model with %s', wngrid.shape)
        self._ngrid = wngrid.shape[0]
        gpu_wngrid = to_gpu(wngrid,allocator=self._memory_pool.allocate)
        sigma_xsec = GPUArray(shape=(model.nLayers, wngrid.shape[0]), dtype=np.float64, allocator=self._memory_pool.allocate)

        THREAD_PER_BLOCK_X = 16
        THREAD_PER_BLOCK_Y = 16
        
        NUM_BLOCK_X = int(math.ceil(self._ngrid/THREAD_PER_BLOCK_X))
        NUM_BLOCK_Y = int(math.ceil(model.nLayers/THREAD_PER_BLOCK_Y))

        molecules = model.chemistry.activeGases + model.chemistry.inactiveGases

        for gasname in molecules:
            sigma = get_rayleigh_kernal(gasname)

            if sigma is not None:

                self._mix_array.set(model.chemistry.get_gas_mix_profile(gasname))

                sigma(sigma_xsec, gpu_wngrid, self._mix_array, np.int32(model.nLayers),np.int32(self._ngrid),
                      block=(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y,1), grid=(NUM_BLOCK_X, NUM_BLOCK_Y,1) )

                self.sigma_xsec = sigma_xsec
                yield gasname, sigma_xsec

    @classmethod
    def input_keywords(cls):
        return ['RayleighCUDA', ]