import numpy as np
from taurex.model.simplemodel import SimpleForwardModel
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray, to_gpu, zeros
from functools import lru_cache
import math
import pycuda.tools as pytools
from taurex_cuda.contributions.cudacontribution import CudaContribution


@lru_cache(maxsize=400)
def absorption_kernal(nlayers, ngrid):

    code = f"""

    __global__ void compute_absorption(double* __restrict__ dest, double* __restrict__ tau,
                                        const double* __restrict__ dz, 
                                        const double* __restrict__ altitude,
                                        const double pradius, const double sradius)
    {{
        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (i >= {ngrid})
            return;

        double integral = 0.0;
        for (int layer=0; layer < {nlayers}; layer++)
        {{
            double etau = exp(-tau[layer*{ngrid} + i]);
            double _dz = dz[layer];
            double ap = altitude[layer];
            integral += (pradius+ap)*(1.0-etau)*_dz*2.0;
            tau[layer*{ngrid} + i] = etau;
        }}
        dest[i] = ((pradius*pradius) + integral)/(sradius*sradius);

    }}
    """
    mod = SourceModule(code)
    return mod.get_function('compute_absorption')

class TransmissionCudaModel(SimpleForwardModel):
    """

    A forward model for transits using GPU acceleration

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

    num_streams: int, optional
        Non-functional for now. 

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
                 num_streams=1):

        super().__init__(self.__class__.__name__, planet,
                         star,
                         pressure_profile,
                         temperature_profile,
                         chemistry,
                         nlayers,
                         atm_min_pressure,
                         atm_max_pressure)

        self.set_num_streams(num_streams)
        self._memory_pool = pytools.DeviceMemoryPool()
        self._tau_memory_pool = pytools.PageLockedMemoryPool()
    def compute_path_length(self, dz):

        

        planet_radius = self._planet.fullRadius
        total_layers = self.nLayers

        dl = np.zeros(shape=(total_layers, total_layers),dtype=np.float64)
        cpu_dl = []
        z = self.altitudeProfile
        self.debug('Computing path_length: \n z=%s \n dz=%s', z, dz)

        for layer in range(0, total_layers):

            p = (planet_radius+dz[0]/2 + z[layer])**2
            k = np.zeros(shape=(self.nLayers-layer))
            k[0] = np.sqrt((planet_radius + dz[0]/2. + z[layer] +
                            dz[layer]/2.)**2 - p)

            k[1:] = np.sqrt((planet_radius + dz[0]/2 + z[layer+1:] +
                             dz[layer+1:]/2)**2 - p)

            k[1:] -= np.sqrt((planet_radius + dz[0]/2 +
                              z[layer:self.nLayers-1] +
                              dz[layer:self.nLayers-1]/2)**2 - p)

            final_k = k*2.0
            dl[layer, :k.shape[0]] = (final_k)[:]
            cpu_dl.append(final_k)

        self._path_length.set(dl)
        return cpu_dl


    def set_num_streams(self, num_streams):
        self._streams = [drv.Stream() for x in range(num_streams)]

    def build(self):
        
        super().build()
        self._startK = to_gpu(np.array([0 for x in range(self.nLayers)]).astype(np.int32))
        self._endK = to_gpu(np.array([self.nLayers-x for x in range(self.nLayers)]).astype(np.int32))
        self._density_offset = to_gpu(np.array(list(range(self.nLayers))).astype(np.int32))
        self._path_length = to_gpu(np.zeros(shape=(self.nLayers, self.nLayers)))
        
        #self._tau_buffer= drv.pagelocked_zeros(shape=(self.nativeWavenumberGrid.shape[-1], self.nLayers,),dtype=np.float64)


    def path_integral(self, wngrid, return_contrib):
        from taurex.util.util import compute_dz
        total_layers = self.nLayers

        dz = compute_dz(self.altitudeProfile)

        wngrid_size = wngrid.shape[0]
        self._ngrid = wngrid_size
        cpu_dl = self.compute_path_length(dz)

        density_profile = to_gpu(self.densityProfile,allocator=self._memory_pool.allocate)

        

        self._cuda_contribs = [c for c in self.contribution_list if isinstance(c, CudaContribution)]
        self._noncuda_contribs = [c for c in self.contribution_list if not isinstance(c, CudaContribution)]
        self._fully_cuda = len(self._noncuda_contribs) == 0    



        tau = zeros(shape=(total_layers, wngrid_size), dtype=np.float64,allocator=self._memory_pool.allocate)
        tau_host = self._tau_memory_pool.allocate(shape=(total_layers, wngrid_size), dtype=np.float64)
        if not self._fully_cuda:
            tau.set(self.fallback_noncuda(total_layers, cpu_dl, self.densityProfile,dz))



        
        for contrib in self._cuda_contribs:
            contrib.contribute(self, self._startK, self._endK, self._density_offset, 0,
                                   density_profile, tau, path_length=self._path_length)

        drv.Context.synchronize()

        final_tau = None
        final_rprs = None
        rprs = zeros(shape=(wngrid_size), dtype=np.float64,allocator=self._memory_pool.allocate)
        self.compute_absorption(rprs,tau, dz)
        tau.get(ary=tau_host,pagelocked=True)
        
        final_tau = np.copy(tau_host)
        final_rprs = rprs.get()

        return final_rprs, final_tau


    def fallback_noncuda(self, total_layers, path_length, density_profile,
                         dz):
        tau = np.zeros(shape=(total_layers, self._ngrid))
        for layer in range(total_layers):

            self.debug('Computing layer %s', layer)
            dl = path_length[layer]

            endK = total_layers-layer

            for contrib in self._noncuda_contribs:
                self.debug('Adding contribution from %s', contrib.name)
                contrib.contribute(self, 0, endK, layer, layer,
                                   density_profile, tau, path_length=dl)
        return tau

    # def compute_absorption_cpu(self, tau, dz):

    #     tau = np.exp(-tau)
    #     ap = self.altitudeProfile[:, None]
    #     pradius = self._planet.fullRadius
    #     sradius = self._star.radius
    #     _dz = dz[:, None]

    #     integral = np.sum((pradius+ap)*(1.0-tau)*_dz*2.0, axis=0)
    #     return ((pradius**2.0) + integral)/(sradius**2), tau

    def compute_absorption(self, rprs, tau, dz):

        grid_size = tau.shape[-1]
        rprs_kernal = absorption_kernal(self.nLayers, grid_size)

        THREAD_PER_BLOCK_X = 256
        NUM_BLOCK_X = int(math.ceil(grid_size/THREAD_PER_BLOCK_X))

        rprs_kernal(rprs, tau, drv.In(dz), drv.In(self.altitudeProfile),
                    np.float64(self._planet.fullRadius),
                    np.float64(self._star.radius),
                    block=(THREAD_PER_BLOCK_X, 1, 1),
                    grid=(NUM_BLOCK_X, 1, 1))

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
