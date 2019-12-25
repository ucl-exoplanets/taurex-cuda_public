import numpy as np
from taurex.model.simplemodel import SimpleForwardModel
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray, to_gpu, zeros
from functools import lru_cache
import math
from ..utils.emission import cuda_blackbody
import pycuda.tools as pytools

class EmissionCudaModel(SimpleForwardModel):
    """

    A forward model for transits

    Parameters
    ----------

    planet: :class:`~taurex.data.planet.Planet`, optional
        Planet model, default planet is Jupiter

    star: :class:`~taurex.data.stellar.star.Star`, optional
        Star model, default star is Sun-like

    pressure_profile: :class:`~taurex.data.profiles.pressure.pressureprofile.PressureProfile`, optional
        Pressure model, alternative is to set ``nlayers``, ``atm_min_pressure``
        and ``atm_max_pressure``

    temperature_profile: :class:`~taurex.data.profiles.temperature.tprofile.TemperatureProfile`, optional
        Temperature model, default is an :class:`~taurex.data.profiles.temperature.isothermal.Isothermal`
        profile at 1500 K

    chemistry: :class:`~taurex.data.profiles.chemistry.chemistry.Chemistry`, optional
        Chemistry model, default is
        :class:`~taurex.data.profiles.chemistry.taurexchemistry.TaurexChemistry` with
        ``H2O`` and ``CH4``

    nlayers: int, optional
        Number of layers. Used if ``pressure_profile`` is not defined.

    atm_min_pressure: float, optional
        Pressure at TOA. Used if ``pressure_profile`` is not defined.

    atm_max_pressure: float, optional
        Pressure at BOA. Used if ``pressure_profile`` is not defined.

    """

    def __init__(self,
                 planet=None,
                 star=None,
                 pressure_profile=None,
                 temperature_profile=None,
                 chemistry=None,
                 nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6,
                 ngauss=4):

        super().__init__(self.__class__.__name__, planet,
                         star,
                         pressure_profile,
                         temperature_profile,
                         chemistry,
                         nlayers,
                         atm_min_pressure,
                         atm_max_pressure)

        self.set_num_gauss(ngauss)

    def set_num_gauss(self, value):
        self._ngauss = int(value)



    def build(self):
        super().build()
        self._start_surface_K = to_gpu(np.array([0]).astype(np.int32))
        self._end_surface_K = to_gpu(np.array([self.nLayers]).astype(np.int32))

        self._start_layer = to_gpu(np.array([x+1 for x in range(self.nLayers)],dtype=np.int32))
        self._end_layer = to_gpu(np.array([self.nLayers for x in range(self.nLayers)],dtype=np.int32))

        self._start_dtau = to_gpu(np.array([x for x in range(self.nLayers)]).astype(np.int32))
        self._end_dtau = to_gpu(np.array([x+1 for x in range(self.nLayers)]).astype(np.int32))

        self._dz = zeros(shape=(self.nLayers,self.nLayers, ),dtype=np.float64)
        self._density_offset = zeros(shape=(self.nLayers,),dtype=np.int32)
        self._memory_pool = pytools.DeviceMemoryPool()

    @lru_cache(maxsize=4)
    def _gen_ngauss_kernal(self, ngauss, nlayers, grid_size):
        from taurex.constants import PI
        mu, weight = np.polynomial.legendre.leggauss(ngauss*2)
        mu_quads = mu[ngauss:]
        wi_quads = weight[ngauss:]

        code = f"""

        __global__ void quadrature_kernal(double* __restrict__ dest, 
                                          const double* __restrict__ layer_tau, 
                                          const double* __restrict__ dtau, 
                                          const double* __restrict__ BB)
        {{
            unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
            
        if ( i >= {grid_size} )
            return;


            double I = 0.0;
            for (int layer = 0; layer < {nlayers}; layer++)
            {{

                double _dtau = dtau[layer*{grid_size} + i];
                double _layer_tau = layer_tau[layer*{grid_size} + i];
                double _BB = BB[layer*{grid_size} + i]*{1.0/PI};
                
                _dtau += _layer_tau;
                if (layer == 0){{


        """
        for mu,weight in zip(mu_quads, wi_quads):
            code += f"""

                I += exp(-_dtau*{1.0/mu})*{mu*weight}*_BB;
            """
        code+=f"""
        }}
        """
        for mu,weight in zip(mu_quads, wi_quads):
            code += f"""

                I += (exp(-_layer_tau*{1.0/mu}) - exp(-_dtau*{1.0/mu}))*{mu*weight}*_BB;
            """
        
        
        
        code += f"""
                }}

                dest[i] = {2.0*PI}*I;

        }}
        """

        mod = SourceModule(code)
        return mod.get_function('quadrature_kernal')

    def path_integral(self, wngrid, return_contrib):

        dz = np.gradient(self.altitudeProfile)
        dz = np.array([dz for x in range(self.nLayers)])
        self._dz.set(dz)

        wngrid_size = wngrid.shape[0]
        total_layers = self.nLayers
        temperature = self.temperatureProfile
        density_profile = to_gpu(self.densityProfile, allocator=self._memory_pool.allocate)
        total_layers = self.nLayers

        layer_tau = zeros(shape=(total_layers, wngrid_size), dtype=np.float64, allocator=self._memory_pool.allocate)
        dtau = zeros(shape=(total_layers, wngrid_size), dtype=np.float64, allocator=self._memory_pool.allocate)
        BB = zeros(shape=(total_layers, wngrid_size), dtype=np.float64, allocator=self._memory_pool.allocate)
        I = zeros(shape=(wngrid_size), dtype=np.float64, allocator=self._memory_pool.allocate)
        cuda_blackbody(wngrid, temperature, out=BB)


        for contrib in self.contribution_list:
            contrib.contribute(self, self._start_layer, self._end_layer, self._density_offset, 0,
                                density_profile, layer_tau, path_length=self._dz, with_sigma_offset=True)
            contrib.contribute(self, self._start_dtau, self._end_dtau, self._density_offset, 0,
                               density_profile, dtau, path_length=self._dz, with_sigma_offset=True)

        integral_kernal = self._gen_ngauss_kernal(self._ngauss, self.nLayers, wngrid_size)

        THREAD_PER_BLOCK_X = 64
        
        NUM_BLOCK_X = int(math.ceil(wngrid_size/THREAD_PER_BLOCK_X))
        
        integral_kernal(I, layer_tau, dtau, BB,
                      block=(THREAD_PER_BLOCK_X, 1,1), grid=(NUM_BLOCK_X, 1, 1) )

        return self.compute_final_flux(I.get()), None


    def compute_final_flux(self, f_total):
        star_sed = self._star.spectralEmissionDensity

        self.debug('Star SED: %s', star_sed)
        # quit()
        star_radius = self._star.radius
        planet_radius = self._planet.fullRadius
        self.debug('star_radius %s', self._star.radius)
        self.debug('planet_radius %s', self._star.radius)
        last_flux = (f_total/star_sed) * (planet_radius/star_radius)**2

        self.debug('last_flux %s', last_flux)

        return last_flux

        # tau = np.exp(-tau.get())
        # ap = self.altitudeProfile[:, None]
        # pradius = self._planet.fullRadius
        # sradius = self._star.radius
        # _dz = dz[:, None]

        # integral = np.sum((pradius+ap)*(1.0-tau)*_dz*2.0, axis=0)
        # return ((pradius**2.0) + integral)/(sradius**2), tau

    @classmethod
    def input_keywords(cls):
        return ['transmission_cuda', 'transit_cuda', ]
