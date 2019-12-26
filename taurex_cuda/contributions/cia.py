from .cudacontribution import CudaContribution
import numpy as np
from taurex.cache import OpacityCache
from ..opacity.cudacache import CudaCiaCache
from pycuda.gpuarray import GPUArray
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceMemoryPool
from functools import lru_cache

import math
@lru_cache(maxsize=4)
def _contribute_cia_kernal(nlayers, grid_size, with_sigma_offset=False):
    
    extra = '+layer'
    if with_sigma_offset:
        extra = ''


    code = f"""
    
    __global__ void contribute_cia(double* dest, const double* __restrict__ sigma, 
                                   const double* __restrict__ density, const double* __restrict__ path,
                                   const int* __restrict__ startK, const int* __restrict__ endK,
                                   const int* __restrict__ density_offset)
    {{
        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
        
        if ( i >= {grid_size} )
            return;
        
        for (unsigned int layer=0; layer<={nlayers}; layer++)
        {{
            unsigned int _startK = startK[layer];
            unsigned int _endK = endK[layer];
            for (unsigned int k = _startK; k < _endK; k++)
            {{
                double _path = path[layer*{nlayers} + k];
                double _density = density[k+density_offset[layer]];
                dest[layer*{grid_size} + i] += sigma[(k{extra})*{grid_size} + i]*_path*_density*_density;
            }}
            
        }}
    
    }}
    
    """
    mod = SourceModule(code)
    return mod.get_function('contribute_cia')

def cuda_contribute_cia(startK, endK, density_offset, sigma, density, path, nlayers, ngrid,tau=None, with_sigma_offset=False):
    kernal = _contribute_cia_kernal(nlayers, ngrid, with_sigma_offset=with_sigma_offset)
    my_tau = tau
    if my_tau is None:
        my_tau = GPUArray(shape=(nlayers,ngrid),dtype=np.float64)
    
    THREAD_PER_BLOCK_X = 256
    NUM_BLOCK_X = int(math.ceil(ngrid/THREAD_PER_BLOCK_X))

    kernal(my_tau, sigma, density, path, startK, endK, density_offset, 
           block=(THREAD_PER_BLOCK_X, 1, 1),
           grid=(NUM_BLOCK_X, 1, 1))
    if tau is None:
        return my_tau



class CIACuda(CudaContribution):
    def __init__(self, cia_pairs=None):
        super().__init__('CIA')
        self._opacity_cache = CudaCiaCache()
        self._xsec_cache = {}
        self._memory_pool = DeviceMemoryPool()
        self._cia_pairs = cia_pairs
        if self._cia_pairs is None:
            self._cia_pairs = []

    def build(self, model):
        super().build(model)
        self._opacity_cache.set_native_grid(model.nativeWavenumberGrid)

    def contribute(self, model, start_layer, end_layer,
                   density_offset, layer, density, tau, path_length=None, with_sigma_offset=False):
        """
        Computes an integral for a single layer for the optical depth.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            A forward model

        start_layer: int
            Lowest layer limit for integration

        end_layer: int
            Upper layer limit of integration

        density_offset: int
            offset in density layer

        layer: int
            atmospheric layer being computed

        density: :obj:`array`
            density profile of atmosphere

        tau: :obj:`array`
            optical depth to store result

        path_length: :obj:`array`
            integration length

        """
        self.debug(' %s %s %s %s %s %s %s', start_layer, end_layer,
                   density_offset, layer, density, tau, self._ngrid)
        cuda_contribute_cia(start_layer, end_layer, density_offset,
                            self.sigma_xsec, density, path_length,
                            self._nlayers, self._ngrid, tau,with_sigma_offset)
        self.debug('DONE')



    @property
    def ciaPairs(self):
        """
        Returns list of molecular pairs involved

        Returns
        -------
        :obj:`list` of str
        """

        return self._cia_pairs

    @ciaPairs.setter
    def ciaPairs(self, value):
        self._cia_pairs = value


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
        sigma_xsec = GPUArray(shape=(model.nLayers, wngrid.shape[0]), dtype=np.float64, allocator=self._memory_pool.allocate)

        chemistry = model.chemistry
        # Loop through all active gases
        for pairName in self.ciaPairs:
            xsec = self._opacity_cache[pairName]
            cia = xsec._xsec
            cia_factor = chemistry.get_gas_mix_profile(cia.pairOne) * chemistry.get_gas_mix_profile(cia.pairTwo)

            # Get the cross section object relating to the gas
            
            xsec.opacity(model.temperatureProfile,
                         cia_factor, wngrid=wngrid, dest=sigma_xsec)

            # Temporarily assign to master cross-section
            self.sigma_xsec = sigma_xsec
            yield pairName, sigma_xsec

    def write(self, output):
        contrib = super().write(output)
        if len(self.ciaPairs) > 0:
            contrib.write_string_array('cia_pairs', self.ciaPairs)
        return contrib


    @classmethod
    def input_keywords(cls):
        return ['CIACUDA', ]