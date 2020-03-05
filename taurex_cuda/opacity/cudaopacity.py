from functools import lru_cache
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from taurex.log import Logger
from pycuda.gpuarray import GPUArray, to_gpu
from pycuda.compiler import SourceModule
import math
from taurex.cache import OpacityCache


class CudaOpacity(Logger):
    
    def __init__(self, molecule_name, wngrid=None):
        super().__init__(self.__class__.__name__)
        self._xsec = OpacityCache()[molecule_name]

        self._lenP = len(self._xsec.pressureGrid)
        self._lenT = len(self._xsec.temperatureGrid)
        self.info('Transfering xsec grid to GPU')
        self._gpu_tgrid = to_gpu(self._xsec.temperatureGrid)
        self._gpu_pgrid = to_gpu(self._xsec.pressureGrid)
        self._memory_pool = drv.DeviceMemoryPool()
        self.transfer_xsec_grid(wngrid)

    def transfer_xsec_grid(self, wngrid):

        self._wngrid = self._xsec.wavenumberGrid
        xsecgrid = self._xsec.xsecGrid

        if wngrid is not None:
            from scipy.interpolate import interp1d
            self._wngrid = wngrid
            f = interp1d(self._xsec.wavenumberGrid, self._xsec.xsecGrid, copy=False,
                         bounds_error=False,fill_value=0.0, assume_sorted=True)
            xsecgrid = f(wngrid).ravel().reshape(*xsecgrid.shape[0:-1],-1) # Force contiguous array

        self._strides = xsecgrid.strides
        self._gpu_grid = GPUArray(shape=xsecgrid.shape,dtype=xsecgrid.dtype )
        self._gpu_grid.set(xsecgrid)
        self.kernal_func.cache_clear()

    def _get_kernal_function(self, nlayers=100, min_wn=None, max_wn=None):
        min_grid_idx = 0
        max_grid_idx = len(self._wngrid)
        if min_wn is not None:
            min_grid_idx = max(np.argmax(min_wn<self._wngrid)-1,0)
        if max_wn is not None:
            max_grid_idx = np.argmax(self._wngrid>=max_wn)+1
        grid_length = self._wngrid[min_grid_idx:max_grid_idx].shape[0]
        return self.kernal_func(nlayers, min_grid_idx, grid_length), grid_length
    
    @lru_cache(maxsize=4)
    def kernal_func(self, nlayers, min_idx, grid_length):
        
        
        
        code = f"""
        
        __global__ void interp_mix_layers(double* __restrict__ dest, const double* __restrict__ xsec_grid,const double* __restrict__ tgrid, 
                                         const double* __restrict__ pgrid, const double* __restrict__ temperature,
                                      const double* __restrict__ pressure, const int * __restrict__ Tmin, const int * __restrict__ Tmax, 
                                      const int* __restrict__ Pmin, const int* __restrict__ Pmax, const double * __restrict__ mix_ratio)
        {{
            unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
            unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
            
            if ( i >= {grid_length} )
                return;
            
            if ( j >= {nlayers} )
                return;
            
            
            int Tmin_idx = Tmin[j];
            int Tmax_idx = Tmax[j];
            int Pmin_idx = Pmin[j];
            int Pmax_idx = Pmax[j];
            double Tmin_val = tgrid[Tmin_idx];
            double Tmax_val = tgrid[Tmax_idx];
            double Pmin_val = pgrid[Pmin_idx];
            double Pmax_val = pgrid[Pmax_idx];
            double T = temperature[j];
            double P = pressure[j];
            double mix = mix_ratio[j];
            double _x11 = xsec_grid[Pmin_idx*{self._strides[0]//8} + Tmin_idx*{self._strides[1]//8} + i + {min_idx}];
            double _x12 = xsec_grid[Pmin_idx*{self._strides[0]//8} + Tmax_idx*{self._strides[1]//8} + i + {min_idx}];
            double _x21 = xsec_grid[Pmax_idx*{self._strides[0]//8} + Tmin_idx*{self._strides[1]//8} + i + {min_idx}];
            double _x22 = xsec_grid[Pmax_idx*{self._strides[0]//8} + Tmax_idx*{self._strides[1]//8} + i + {min_idx}];
            dest[j*{grid_length} + i] =((_x11*(Pmax_val - Pmin_val)*(Tmax_val - Tmin_val) - (P - Pmin_val)*(Tmax_val - Tmin_val)*(_x11 - _x21) - (T - Tmin_val)*(-(P - Pmin_val)*(_x11 - _x21) + 
                                        (P - Pmin_val)*(_x12 - _x22) + (Pmax_val - Pmin_val)*(_x11 - _x12)))/
                                        ((Pmax_val - Pmin_val)*(Tmax_val - Tmin_val)))*mix/10000.0;
        }}                    
        
        
        """
        
        self._module = SourceModule(code)
        interp_kernal = self._module.get_function("interp_mix_layers")
        
        return interp_kernal
    
    
    def kernal(self, nlayers=100,wngrid=None):
        if wngrid is None:
            return self._get_kernal_function()
        else:
            min_wngrid = wngrid.min()
            max_wngrid = wngrid.max()
            return self._get_kernal_function(nlayers, min_wngrid, max_wngrid)

    def compile_temperature_pressure_mix(self, temperature, pressure, mix):
        
        T_min = np.digitize(temperature, self._xsec.temperatureGrid).astype(np.int32)-1
        np.clip(T_min, 0, self._lenT-1,out=T_min)
        T_max = T_min+1
        np.clip(T_max, 0, self._lenT-1,out=T_max)
        P_min = np.digitize(pressure, self._xsec.pressureGrid).astype(np.int32)-1
        np.clip(P_min, 0, self._lenP-1,out=P_min)
        P_max = P_min+1
        np.clip(P_max, 0, self._lenP-1,out=P_max)
        
        
        return T_min, T_max, P_min, P_max, mix
        
        
        
        
    def find_closest_index(self,T,P):
        t_min=self._xsec.temperatureGrid.searchsorted(T,side='right')-1
        t_min = max(min(t_min, len(self._xsec.temperatureGrid)-1),0)
        t_max = t_min+1
        t_max = max(min(t_max, len(self._xsec.temperatureGrid)-1),0)

        p_min=self._xsec.pressureGrid.searchsorted(P,side='right')-1
        p_min = max(min(p_min, len(self._xsec.pressureGrid)-1),0)
        p_max = p_min+1
        p_max = max(min(p_max, len(self._xsec.pressureGrid)-1),0)
        return t_min,t_max,p_min,p_max
    
    
    def opacity(self, temperature, pressure, mix, wngrid=None, dest=None):
        minmaxT = self._xsec.temperatureGrid.min(),self._xsec.temperatureGrid.max()
        minmaxP = self._xsec.pressureGrid.min(),self._xsec.pressureGrid.max()
        temperature = np.clip(temperature,*minmaxT)
        pressure = np.clip(pressure,*minmaxP)
        T_min, T_max, P_min, P_max, mix = \
            self.compile_temperature_pressure_mix(temperature, pressure, mix)
        
        nlayers = len(temperature)
        compute_kernal, grid_length = self.kernal(nlayers=nlayers,wngrid=wngrid)
        
        my_dest = dest
        if my_dest is None:
            my_dest = GPUArray(shape=(nlayers, grid_length),dtype=np.float64) 
        
        THREAD_PER_BLOCK_X = 16
        THREAD_PER_BLOCK_Y = 16
        
        NUM_BLOCK_X = int(math.ceil(grid_length/THREAD_PER_BLOCK_X))
        NUM_BLOCK_Y = int(math.ceil(nlayers/THREAD_PER_BLOCK_Y))
        
        compute_kernal(my_dest, self._gpu_grid,self._gpu_tgrid,self._gpu_pgrid, drv.In(temperature.ravel()), drv.In(pressure.ravel()),
                       drv.In(T_min), drv.In(T_max), drv.In(P_min), drv.In(P_max), drv.In(mix),
                      block=(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y,1), grid=(NUM_BLOCK_X, NUM_BLOCK_Y,1) )
        if dest is None:
            return my_dest
        