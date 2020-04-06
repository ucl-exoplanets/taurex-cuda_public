import numpy as np
from taurex.data.fittable import fitparam
from .cudacontribution import CudaContribution
from pycuda.gpuarray import GPUArray, to_gpu, zeros
from pycuda.tools import DeviceMemoryPool
from pycuda.compiler import SourceModule
from functools import lru_cache
import math

@lru_cache(maxsize=400)
def _flat_kernal(grid_size):

    code = f"""

    __global__ void flat_cloud(double* dest,const double value, const int start_layer, const int end_layer)
    {{
        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
        unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (i >= {grid_size})
            return;
        j+= start_layer;
        if (j >= end_layer)
            return;
        
        dest[j*{grid_size} + i] = value;
    }}
    """
    mod = SourceModule(code)
    return mod.get_function('flat_cloud')


class FlatMieCuda(CudaContribution):
    """
    Computes a flat absorption contribution
    across all wavelengths to the optical depth

    Parameters
    ----------

    flat_mix_ratio: float
        Opacity value

    flat_bottomP: float
        Bottom of absorbing region in Pa

    flat_topP: float
        Top of absorbing region in Pa

    """

    def __init__(self,
                 flat_mix_ratio=1e-10, flat_bottomP=-1,
                 flat_topP=-1):
        super().__init__('Mie')

        self._mie_mix = flat_mix_ratio
        self._mie_bottom_pressure = flat_bottomP
        self._mie_top_pressure = flat_topP
        self._memory_pool = DeviceMemoryPool()

    @fitparam(param_name='flat_topP',
              param_latex='$P^{mie}_\mathrm{top}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[1e-20, 1])
    def mieTopPressure(self):
        """
        Pressure at top of absorbing region in Pa
        """
        return self._mie_top_pressure

    @mieTopPressure.setter
    def mieTopPressure(self, value):
        self._mie_top_pressure = value

    @fitparam(param_name='flat_bottomP',
              param_latex='$P^{mie}_\mathrm{bottom}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[1e-20, 1])
    def mieBottomPressure(self):
        """
        Pressure at bottom of absorbing region in Pa
        """
        return self._mie_bottom_pressure

    @mieBottomPressure.setter
    def mieBottomPressure(self, value):
        self._mie_bottom_pressure = value

    @fitparam(param_name='flat_mix_ratio',
              param_latex='$\chi_\mathrm{mie}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[1e-20, 1])
    def mieMixing(self):
        """
        Opacity of absorbing region in m2
        """
        return self._mie_mix

    @mieMixing.setter
    def mieMixing(self, value):
        self._mie_mix = value

    def prepare_each(self, model, wngrid):
        """
        Computes and flat absorbing opacity for
        the pressure regions given

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            ``Flat`` and the weighted mie opacity.


        """
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]
        sigma_xsec = GPUArray(shape=(model.nLayers, wngrid.shape[0]), dtype=np.float64, allocator=self._memory_pool.allocate)

        pressure_profile = model.pressureProfile

        bottom_pressure = self.mieBottomPressure
        if bottom_pressure < 0:
            bottom_pressure = pressure_profile[0]

        top_pressure = self.mieTopPressure
        if top_pressure < 0:
            top_pressure = pressure_profile[-1]
        
        start_layer = np.argmax(pressure_profile < bottom_pressure)
        end_layer = np.argmax(pressure_profile < top_pressure)

        sigma_xsec = zeros(shape=(self._nlayers, wngrid.shape[0]), dtype=np.float64, allocator=self._memory_pool.allocate)

        flat_kernel = _flat_kernal(self._ngrid) 

        THREAD_PER_BLOCK_X = 16
        THREAD_PER_BLOCK_Y = 16
        
        NUM_BLOCK_X = int(math.ceil(self._ngrid/THREAD_PER_BLOCK_X))
        NUM_BLOCK_Y = int(math.ceil(model.nLayers/THREAD_PER_BLOCK_Y))


        flat_kernel(sigma_xsec,np.float64(self.mieMixing), 
                    np.int32(start_layer), np.int32(end_layer), 
                    block=(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y,1), grid=(NUM_BLOCK_X, NUM_BLOCK_Y,1) )

        self.sigma_xsec = sigma_xsec

        yield 'Flat', sigma_xsec

    def write(self, output):
        contrib = super().write(output)
        contrib.write_scalar('flat_mix_ratio', self._mie_mix)
        contrib.write_scalar('flat_bottomP', self._mie_bottom_pressure)
        contrib.write_scalar('flat_topP', self._mie_top_pressure)
        return contrib

    @classmethod
    def input_keywords(self):
        return ['FlatMieCUDA', ]