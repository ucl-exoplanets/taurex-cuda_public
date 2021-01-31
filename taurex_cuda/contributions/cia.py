from .cudacontribution import CudaContribution
import numpy as np
from taurex.cache import OpacityCache
from ..opacity.cudacache import CudaCiaCache
from pycuda.gpuarray import GPUArray
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceMemoryPool
from functools import lru_cache

import math
@lru_cache(maxsize=400)
def _contribute_cia_kernal(nlayers, grid_size,start_layer=0, with_sigma_offset=False):
    
    extra = '+layer'
    if with_sigma_offset:
        extra = ''


    code = f"""
    
    __global__ void contribute_cia(double* dest, const double* __restrict__ sigma, 
                                   const double* __restrict__ density, const double* __restrict__ path,
                                   const int* __restrict__ startK, const int* __restrict__ endK,
                                   const int* __restrict__ density_offset, const int total_layers)
    {{
        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
        
        if ( i >= {grid_size} )
            return;
        
        for (unsigned int layer={start_layer}; layer<{nlayers}; layer++)
        {{
            unsigned int _startK = startK[layer];
            unsigned int _endK = endK[layer];
            for (unsigned int k = _startK; k < _endK; k++)
            {{
                double _path = path[layer*total_layers + k];
                double _density = density[k+density_offset[layer]];
                dest[layer*{grid_size} + i] += sigma[(k{extra})*{grid_size} + i]*_path*_density*_density;
            }}
            
        }}
    
    }}
    
    """
    mod = SourceModule(code)
    func = mod.get_function('contribute_cia')
    func.prepare('PPPPPPPi')
    return func

import math
@lru_cache(maxsize=400)
def _contribute_cia_kernal_II(nlayers, grid_size,start_layer=0, with_sigma_offset=False):
    
    extra = '+layer'
    if with_sigma_offset:
        extra = ''


    code = f"""
    
    __global__ void contribute_cia(double* dest, const double* __restrict__ sigma, 
                                   const double* __restrict__ density, const double* __restrict__ path,
                                   const int* __restrict__ startK, const int* __restrict__ endK,
                                   const int* __restrict__ density_offset, const int total_layers)
    {{
        __shared__ double path_cache[{nlayers}];
        __shared__ double dens_cache[{nlayers}];

        const unsigned int grid = (blockIdx.x * blockDim.x) + threadIdx.x;
        const unsigned int layer = (blockIdx.y * blockDim.y) + threadIdx.y; + {start_layer};
        
        unsigned int _s_layer = threadIdx.y*blockDim.x + threadIdx.x;
                

        if (_s_layer < {nlayers})
        {{
            path_cache[_s_layer] = path[layer*{nlayers} + _s_layer];
            dens_cache[_s_layer] = density[_s_layer];
        }}

        __syncthreads();

        if ( grid >= {grid_size} )
            return;
        if (layer >= {nlayers})
            return;
        
        const unsigned int _startK = startK[layer];
        const unsigned int _endK = endK[layer];
        const unsigned int _offset = density_offset[layer];
        double _result = 0.0;
        for (unsigned int k = _startK; k < _endK; k++)
        {{
            double _path = path_cache[k];
            double _density = dens_cache[k+_offset];
            _result += sigma[(k{extra})*{grid_size} + grid]*_path*_density*_density;
        }}
        dest[layer*{grid_size} + grid] += _result;  
        
    
    }}
    
    """
    mod = SourceModule(code)
    func = mod.get_function('contribute_cia')
    func.prepare('PPPPPPPi')
    return func


def cuda_contribute_cia(startK, endK, density_offset, sigma, density, path, nlayers, ngrid, tau=None, with_sigma_offset=False,
                        start_layer=0,total_layers= None, stream=None):

    kernal = _contribute_cia_kernal_II(nlayers, ngrid, with_sigma_offset=with_sigma_offset, start_layer=start_layer)
    my_tau = tau
    if my_tau is None:
        my_tau = GPUArray(shape=(nlayers,ngrid),dtype=np.float64)
    if total_layers is None:
        total_layers = nlayers
    THREAD_PER_BLOCK_X = 32
    THREAD_PER_BLOCK_Y = 32

    #THREAD_PER_BLOCK_X = 128
    #THREAD_PER_BLOCK_Y = 1
    NUM_BLOCK_Y = int(math.ceil((total_layers)/THREAD_PER_BLOCK_Y))
    NUM_BLOCK_X = int(math.ceil((ngrid)/THREAD_PER_BLOCK_X))
    #NUM_BLOCK_Y = 1 

    kernal.prepared_call(
        (NUM_BLOCK_X, NUM_BLOCK_Y, 1),
        (THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, 1),
        my_tau.gpudata, sigma.gpudata, density.gpudata, path.gpudata, startK.gpudata, endK.gpudata, density_offset.gpudata,np.int32(total_layers))
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
                   density_offset, layer, density, tau, path_length=None, with_sigma_offset=False, streams=None):
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

        streams = None


        self.debug(' %s %s %s %s %s %s %s', start_layer, end_layer,
                   density_offset, layer, density, tau, self._ngrid)
        
        
        if streams is None:
            cuda_contribute_cia(start_layer, end_layer, density_offset,
                                self.sigma_xsec, density, path_length,
                                self._nlayers, self._ngrid, tau,with_sigma_offset)
        else:
            num_streams=len(streams)
            split = self._nlayers//num_streams
            stream_vals = [(x*split, (x+1)*split, streams[x]) for x in range(num_streams-1)] + [((num_streams-1)*split,self._nlayers,streams[-1])]

            for start, end, stream in stream_vals:
                cuda_contribute_cia(start_layer, end_layer, density_offset,
                                    self.sigma_xsec, density, path_length,
                                    end, self._ngrid, tau,with_sigma_offset,start_layer=start,total_layers=self._nlayers,stream=stream)           
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
        self._nlayers = model.nLayers
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