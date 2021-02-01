
from taurex.contributions import Contribution
from functools import lru_cache
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, zeros
import pycuda.driver as drv
import numpy as np
import math
@lru_cache(maxsize=400)
def _sum_kernal(nlayers, ngrid):

    code = f"""

    __global__ void sum_sigma(double* dest, const double* __restrict__ arr)
    {{
        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
        unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (i >= {ngrid})
            return;
        
        if (j >= {nlayers})
            return;
        
        dest[j*{ngrid} + i] += arr[j*{ngrid} + i];
    }}
    """
    mod = SourceModule(code)
    return mod.get_function('sum_sigma')

@lru_cache(maxsize=400)
def _contribute_tau_kernal(nlayers, grid_size, with_sigma_offset=False, start_layer=0):
    extra = '+layer'
    if with_sigma_offset:
        extra = ''


    code = f"""
    __global__ void contribute_tau(double* dest, const double* __restrict__ sigma, 
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
                double _path = path[layer*{nlayers} + k];
                double _density = density[k+density_offset[layer]];
                dest[layer*{grid_size} + i] += sigma[(k{extra})*{grid_size} + i]*_path*_density;
            }}
            
        }}
    
    }}
    
    """
    mod = SourceModule(code)
    func = mod.get_function('contribute_tau')
    func.prepare('PPPPPPPi')
    return func

@lru_cache(maxsize=400)
def _contribute_tau_kernal_II(nlayers, grid_size, with_sigma_offset=False, start_layer=0):
    extra = '+layer'
    if with_sigma_offset:
        extra = ''


    code = f"""
    __global__ void contribute_tau(double* dest, const double* __restrict__ sigma, 
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
            _result += sigma[(k{extra})*{grid_size} + grid]*_path*_density;
        }}
        dest[layer*{grid_size} + grid] += _result; 

    
    }}
    
    """
    mod = SourceModule(code,options=['-O3'])
    func = mod.get_function('contribute_tau')
    func.prepare('PPPPPPPi')
    return func


def cuda_contribute_tau_old(startK, endK, density_offset, sigma, density, path, 
                        nlayers, ngrid, tau=None, with_sigma_offset=False, start_layer=0,total_layers=None,
                        stream=None):

    kernal = _contribute_tau_kernal(nlayers, ngrid, with_sigma_offset=with_sigma_offset, start_layer=start_layer)
    my_tau = tau
    if total_layers is None:
        total_layers = nlayers
    if my_tau is None:
        my_tau = GPUArray(shape=(nlayers,ngrid),dtype=np.float64)
    


    #THREAD_PER_BLOCK_X = 32
    #THREAD_PER_BLOCK_Y = 32

    THREAD_PER_BLOCK_X = 128
    THREAD_PER_BLOCK_Y = 1
    #NUM_BLOCK_Y = int(math.ceil((total_layers)/THREAD_PER_BLOCK_Y))
    NUM_BLOCK_X = int(math.ceil((ngrid)/THREAD_PER_BLOCK_X))
    NUM_BLOCK_Y = 1 

    kernal.prepared_call(
        (NUM_BLOCK_X, NUM_BLOCK_Y, 1),
        (THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y, 1),
        my_tau.gpudata, sigma.gpudata, density.gpudata, path.gpudata, startK.gpudata, endK.gpudata, density_offset.gpudata,np.int32(total_layers))
    if tau is None:
        return my_tau


def cuda_contribute_tau(startK, endK, density_offset, sigma, density, path, 
                        nlayers, ngrid, tau=None, with_sigma_offset=False, start_layer=0,total_layers=None,
                        stream=None):

    kernal = _contribute_tau_kernal_II(nlayers, ngrid, with_sigma_offset=with_sigma_offset, start_layer=start_layer)
    my_tau = tau
    if total_layers is None:
        total_layers = nlayers
    if my_tau is None:
        my_tau = GPUArray(shape=(nlayers,ngrid),dtype=np.float64)
    


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


class CudaContribution(Contribution):


    def __init__(self, name):
        super().__init__(name)
        self._is_cuda_model = False

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
            cuda_contribute_tau(start_layer, end_layer, density_offset,
                                self.sigma_xsec, density, path_length,
                                self._nlayers, self._ngrid, tau,with_sigma_offset)
        else:
            num_streams=len(streams)
            split = self._nlayers//num_streams
            stream_vals = [(x*split, (x+1)*split, streams[x]) for x in range(num_streams-1)] + [((num_streams-1)*split,self._nlayers,streams[-1])]

            for start, end, stream in stream_vals:
                cuda_contribute_tau(start_layer, end_layer, density_offset,
                                    self.sigma_xsec, density, path_length,
                                    end, self._ngrid, tau,with_sigma_offset,start_layer=start,total_layers=self._nlayers,stream=stream)           
        self.debug('DONE')

    def prepare(self, model, wngrid):
        """

        Used to prepare the contribution for the calculation.
        Called before the forward model performs the main optical depth
        calculation. Default behaviour is to loop through :func:`prepare_each`
        and sum all results into a single cross-section.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid
        """

        self._ngrid = wngrid.shape[0]
        self._nlayers = model.nLayers

        sigma_xsec = zeros(shape=(self._nlayers, self._ngrid), dtype=np.float64)

        sum_kernal = _sum_kernal(self._nlayers, self._ngrid)
        THREAD_PER_BLOCK_X = 16
        THREAD_PER_BLOCK_Y = 16
        
        NUM_BLOCK_X = int(math.ceil(self._ngrid/THREAD_PER_BLOCK_X))
        NUM_BLOCK_Y = int(math.ceil(self._nlayers/THREAD_PER_BLOCK_Y))

        for gas, sigma in self.prepare_each(model, wngrid):
            self.debug('Gas %s', gas)
            self.debug('Sigma %s', sigma)
            sum_kernal(sigma_xsec, sigma, block=(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y,1), grid=(NUM_BLOCK_X, NUM_BLOCK_Y,1))

        self.sigma_xsec = sigma_xsec
        self.debug('Final sigma is %s', self.sigma_xsec)
        self.info('Done')





