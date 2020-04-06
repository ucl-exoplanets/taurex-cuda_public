from functools import lru_cache
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from taurex.log import Logger
from pycuda.gpuarray import GPUArray, to_gpu
from pycuda.compiler import SourceModule
import math
from taurex.cache import CIACache


@lru_cache(maxsize=400)
def kernal_func(nlayers, min_idx, stride_1, grid_length):
    
    
    
    code = f"""
    
    __global__ void interp_mix_layers(double* dest, const double* __restrict__ xsec_grid,const double* __restrict__ tgrid, 
                                        const double* __restrict__ temperature, const int * __restrict__ Tmin, const int * __restrict__ Tmax, 
                                        const double * __restrict__ mix_ratio)
    {{
        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
        unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
        
        if ( i >= {grid_length} )
            return;
        
        if ( j >= {nlayers} )
            return;
        
        
        int Tmin_idx = Tmin[j];
        int Tmax_idx = Tmax[j];
        double Tmin_val = tgrid[Tmin_idx];
        double Tmax_val = tgrid[Tmax_idx];
        double T = temperature[j];
        double mix = mix_ratio[j];
        double _x11 = xsec_grid[Tmin_idx*{stride_1} + i + {min_idx}];
        double _x12 = xsec_grid[Tmax_idx*{stride_1} + i + {min_idx}];
        double diff = (Tmax_val - Tmin_val+1.0);
        dest[j*{grid_length} + i] = (_x11 * diff - (T - Tmin_val)*(_x11-_x12) )*mix/diff;
    }}                    
    
    
    """

    module = SourceModule(code)
    
    return module

class CudaCIA(Logger):
    
    def __init__(self, pair_name, wngrid=None):
        super().__init__(self.__class__.__name__)
        self._xsec = CIACache()[pair_name]
        self._gpu_tgrid = to_gpu(self._xsec.temperatureGrid)
        self.transfer_xsec_grid(wngrid)
    def transfer_xsec_grid(self, wngrid):

        self._wngrid = self._xsec.wavenumberGrid
        xsecgrid = self._xsec._xsec_grid

        if wngrid is not None:
            from scipy.interpolate import interp1d
            self._wngrid = wngrid
            f = interp1d(self._xsec.wavenumberGrid, xsecgrid, copy=False,
                         bounds_error=False,fill_value=1e-60, assume_sorted=True)
            xsecgrid = f(wngrid).ravel().reshape(*xsecgrid.shape[0:-1],-1) # Force contiguous array

        self._strides = xsecgrid.strides
        self._gpu_grid = GPUArray(shape=xsecgrid.shape, dtype=xsecgrid.dtype )
        self._gpu_grid.set(xsecgrid)

    def find_closest_index(self,T):
        t_min=self._xsec.temperatureGrid.searchsorted(T,side='right')-1
        t_min = max(min(t_min, len(self._xsec.temperatureGrid)-1),0)
        t_max = t_min+1
        t_max = max(min(t_max, len(self._xsec.temperatureGrid)-1),0)

        return t_min, t_max
    def compile_temperature_mix(self, temperature, mix):
        T_min = []
        T_max = []
        M = []
        
        for t, m in zip(temperature, mix):
            tmn,tmax = self.find_closest_index(t)
            T_min.append(tmn)
            T_max.append(tmax)
            M.append(m)
        
        return np.array(T_min,dtype=np.int32),np.array(T_max,dtype=np.int32), np.array(M,dtype=np.float64)

    def opacity(self, temperature, mix, wngrid=None, dest=None):
        temperature = np.clip(temperature,self._xsec.temperatureGrid.min(),self._xsec.temperatureGrid.max())
        T_min, T_max, mix = \
            self.compile_temperature_mix(temperature,mix)
        
        nlayers = len(temperature)
        compute_kernal, grid_length = self.kernal(nlayers=nlayers, wngrid=wngrid)
        
        my_dest = dest
        if my_dest is None:
            my_dest = GPUArray(shape=(nlayers, grid_length),dtype=np.float64) 
        
        THREAD_PER_BLOCK_X = 16
        THREAD_PER_BLOCK_Y = 16
        
        NUM_BLOCK_X = int(math.ceil(grid_length/THREAD_PER_BLOCK_X))
        NUM_BLOCK_Y = int(math.ceil(nlayers/THREAD_PER_BLOCK_Y))
        
        compute_kernal(my_dest, self._gpu_grid,self._gpu_tgrid,drv.In(temperature.ravel()),
                       drv.In(T_min), drv.In(T_max), drv.In(mix),
                      block=(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y,1), grid=(NUM_BLOCK_X, NUM_BLOCK_Y,1) )
        if dest is None:
            return my_dest


    def _get_kernal_function(self, nlayers=100, min_wn=None, max_wn=None):
        min_grid_idx = 0
        max_grid_idx = len(self._wngrid)
        if min_wn is not None:
            min_grid_idx = max(np.argmax(min_wn<self._wngrid)-1,0)
        if max_wn is not None:
            max_grid_idx = np.argmax(self._wngrid>=max_wn)+1
        grid_length = self._wngrid[min_grid_idx:max_grid_idx].shape[0]
        return kernal_func(nlayers, min_grid_idx, self._strides[0]//8, grid_length).get_function("interp_mix_layers"), grid_length
    
    
    
    def kernal(self, nlayers=100,wngrid=None):
        if wngrid is None:
            return self._get_kernal_function()
        else:
            min_wngrid = wngrid.min()
            max_wngrid = wngrid.max()
            return self._get_kernal_function(nlayers, min_wngrid, max_wngrid)