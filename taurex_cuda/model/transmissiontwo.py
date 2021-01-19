import numpy as np
from taurex.model import ForwardModel
from taurex.model import SimpleForwardModel
from taurex.util.util import clip_native_to_wngrid
from scipy.interpolate import interp1d
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray, to_gpu, zeros
from functools import lru_cache
import math
import pycuda.tools as pytools
from taurex_cuda.contributions.cudacontribution import CudaContribution


@lru_cache(maxsize=400)
def kernal_func(grid_length):

    code = f"""
    
    __global__ void interp_tau(double* dest, const double* __restrict__ tau,const double* __restrict__ agrid, 
                                        const double* __restrict__ ap, const int * __restrict__ amin, 
                                        const int * __restrict__ amax, const int nlayers)
    {{
        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
        unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
        
        if ( i >= {grid_length} )
            return;
        
        if ( j >= nlayers )
            return;
        
        
        int amin_idx = amin[j];
        int amax_idx = amax[j];
        double amin_val = agrid[amin_idx];
        double amax_val = agrid[amax_idx];
        double a = ap[j];
        double _x11 = tau[amin_idx*{grid_length} + i];
        double _x12 = tau[amax_idx*{grid_length} + i];
        double diff = (amax_val - amin_val);
        if (diff == 0.0)
        {{
            dest[j*{grid_length} + i] = _x12;
        }}else{{
            dest[j*{grid_length} + i] += (_x11 * (amax_val - amin_val) - (a - amin_val)*(_x11-_x12) )/diff;
        }}

    }}                    
    
    
    """
    
    module = SourceModule(code)
    interp_kernal = module.get_function("interp_tau")
    
    return interp_kernal


@lru_cache(maxsize=400)
def absorption_kernal(ngrid):

    code = f"""

    __global__ void compute_absorption(double* __restrict__ dest, double* __restrict__ tau,
                                        const double* __restrict__ dz, 
                                        const double* __restrict__ altitude, const int nlayers,
                                        const double pradius, const double sradius)
    {{
        unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (i >= {ngrid})
            return;

        double integral = 0.0;
        for (int layer=0; layer < nlayers; layer++)
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


class TransmissionCudaModelTwo(ForwardModel):
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


    """



    def __init__(self,
                 planet=None,
                 star=None,
                 pressure_profile_one=None,
                 temperature_profile_one=None,
                 chemistry_one=None,
                 nlayers=100,
                 atm_min_pressure_one=1e-4,
                 atm_max_pressure_one=1e6,
                 pressure_profile_two=None,
                 temperature_profile_two=None,
                 chemistry_two=None,
                 atm_min_pressure_two=1e-4,
                 atm_max_pressure_two=1e6,
                 ):
        super().__init__(self.__class__.__name__)
        self._star = star
        self._planet = planet
        self._model_one = SimpleForwardModel('FakeModel',temperature_profile=temperature_profile_one, chemistry=chemistry_one, planet=planet, star=star,
                                         pressure_profile=pressure_profile_one)

        self._model_two = SimpleForwardModel('FakeModel2',temperature_profile=temperature_profile_two, chemistry=chemistry_two,
                                      planet=planet, star=star,
                                      pressure_profile=pressure_profile_two)

        self._star = self._model_one.star
        self._planet = self._model_two.planet
        self._memory_pool = pytools.DeviceMemoryPool()

    @property
    def star(self):
        """
        Star model
        """
        return self._star

    @property
    def planet(self):
        """
        Planet model
        """
        return self._planet

    @property
    def nativeWavenumberGrid(self):
        """

        Searches through active molecules to determine the
        native wavenumber grid

        Returns
        -------

        wngrid: :obj:`array`
            Native grid

        Raises
        ------
        InvalidModelException
            If no active molecules in atmosphere
        """

        return self._model_one.nativeWavenumberGrid

    def build(self):
        """
        Build the forward model. Must be called at least
        once before running :func:`model`
        """
        self._model_one.build()
        self._model_two.build()

        m1 = self._model_one
        m2 = self._model_two

        self._startK_one = to_gpu(np.array([0 for x in range(m1.nLayers)]).astype(np.int32))
        self._endK_one = to_gpu(np.array([m1.nLayers-x for x in range(m1.nLayers)]).astype(np.int32))
        self._density_offset_one = to_gpu(np.array(list(range(m1.nLayers))).astype(np.int32))
        self._path_length_one = to_gpu(np.zeros(shape=(m1.nLayers, m1.nLayers)))
        
        self._tau_buffer = drv.pagelocked_zeros(shape=(self.nativeWavenumberGrid.shape[-1]*(m1.nLayers + m2.nLayers)),dtype=np.float64)

        self._startK_two = to_gpu(np.array([0 for x in range(m2.nLayers)]).astype(np.int32))
        self._endK_two = to_gpu(np.array([m2.nLayers-x for x in range(m2.nLayers)]).astype(np.int32))
        self._density_offset_two = to_gpu(np.array(list(range(m2.nLayers))).astype(np.int32))
        self._path_length_two = to_gpu(np.zeros(shape=(m2.nLayers, m2.nLayers)))
        
        #self._tau_buffer_two = drv.pagelocked_zeros(shape=(self.nativeWavenumberGrid.shape[-1], m2.nLayers,),dtype=np.float64)



    def add_contribution(self, contribution):
        super().add_contribution(contribution)
        self._model_one.add_contribution(contribution)
        self._model_two.add_contribution(contribution)

    def add_contribution_one(self, contribution):
        self._model_one.add_contribution(contribution)
    
    def add_contribution_two(self, contribution):
        self._model_two.add_contribution(contribution)

    def model(self, wngrid=None, cutoff_grid=True):
        """
        Runs the forward model

        Parameters
        ----------

        wngrid: :obj:`array`, optional
            Wavenumber grid, default is to use native grid

        cutoff_grid: bool
            Run model only on ``wngrid`` given, default is ``True``

        Returns
        -------

        native_grid: :obj:`array`
            Native wavenumber grid, clipped if ``wngrid`` passed

        depth: :obj:`array`
            Resulting depth

        tau: :obj:`array`
            Optical depth.

        extra: ``None``
            Empty
        """


        # self._model_one.initialize_profiles()
        # self._model_two.initialize_profiles()
        try:
            model_one = self._model_one.model(wngrid,cutoff_grid)
        except NotImplementedError:
            pass
        try:
            model_two = self._model_two.model(wngrid,cutoff_grid)
        except NotImplementedError:
            pass

        # Clip grid if necessary
        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)


        # Compute path integral
        absorp, tau_one = self.path_integral(native_grid, False)

        return native_grid, absorp, tau_one,

    def compute_path_length(self, dz, ap, layers, gpu_path):

        planet_radius = self._planet.fullRadius
        total_layers = layers

        dl = np.zeros(shape=(total_layers, total_layers),dtype=np.float64)
        cpu_dl = []
        z = ap
        self.debug('Computing path_length: \n z=%s \n dz=%s', z, dz)

        for layer in range(0, total_layers):

            p = (planet_radius+dz[0]/2 + z[layer])**2
            k = np.zeros(shape=(total_layers-layer))
            k[0] = np.sqrt((planet_radius + dz[0]/2. + z[layer] +
                            dz[layer]/2.)**2 - p)

            k[1:] = np.sqrt((planet_radius + dz[0]/2 + z[layer+1:] +
                             dz[layer+1:]/2)**2 - p)

            k[1:] -= np.sqrt((planet_radius + dz[0]/2 +
                              z[layer:total_layers-1] +
                              dz[layer:total_layers-1]/2)**2 - p)

            final_k = k
            dl[layer, :k.shape[0]] = (final_k)[:]
            cpu_dl.append(final_k)

        gpu_path.set(dl)
        return cpu_dl

    def fallback_noncuda(self, total_layers, ngrid, path_length, density_profile,
                         dz, noncuda_contribs):
        tau = np.zeros(shape=(total_layers, ngrid))
        for layer in range(total_layers):

            self.debug('Computing layer %s', layer)
            dl = path_length[layer]

            endK = total_layers-layer

            for contrib in noncuda_contribs:
                self.debug('Adding contribution from %s', contrib.name)
                contrib.contribute(self, 0, endK, layer, layer,
                                   density_profile, tau, path_length=dl)
        return tau


    def path_integral(self, wngrid, return_contrib):
        from taurex.util.util import compute_dz

        dz_one = compute_dz(self._model_one.altitudeProfile)
        dz_two = compute_dz(self._model_two.altitudeProfile)

        wngrid_size = wngrid.shape[0]
        self._ngrid = wngrid_size
        m1 = self._model_one
        m2 = self._model_two

        ap_one = self._model_one.altitudeProfile
        ap_two = self._model_two.altitudeProfile
        layer_one = self._model_one.nLayers
        layer_two = self._model_two.nLayers

        self.debug('Altitude profile [0] %s',ap_one)
        self.debug('Altitude profile [1] %s',ap_two)

        self.debug('Nlayers [0] %s',layer_one)
        self.debug('Nlayers [1] %s',layer_two)

        path_length_one = self.compute_path_length(dz_one, ap_one, layer_one,self._path_length_one)
        path_length_two = self.compute_path_length(dz_two, ap_two, layer_two,self._path_length_two)

        self.debug('Computed path length [0] %s',path_length_one)
        self.debug('Computed path length [1] %s',path_length_two)

        density_profile_one = to_gpu(m1.densityProfile,allocator=self._memory_pool.allocate)
        density_profile_two = to_gpu(m2.densityProfile,allocator=self._memory_pool.allocate)

        total_layers_one = m1.nLayers
        total_layers_two = m2.nLayers

        # path_length_one = self.compute_path_length(dz_one)
        # # path_length_two = self.compute_path_length(dz_two)
        # self.path_length_one = path_length_one
        # self.path_length_one = path_length_two

        tau_one = zeros(shape=(total_layers_one, wngrid_size), dtype=np.float64)
        cuda_contribs_one = [c for c in m1.contribution_list if isinstance(c, CudaContribution)]
        noncuda_contribs_one = [c for c in m1.contribution_list if not isinstance(c, CudaContribution)]
        fully_cuda_one = len(noncuda_contribs_one) == 0 

        if not fully_cuda_one:
            tau_one.set(self.fallback_noncuda(total_layers_one, wngrid_size,
                                              path_length_one, m1.densityProfile, dz_one,noncuda_contribs_one)
                                              )

        for contrib in cuda_contribs_one:
            contrib.contribute(self, self._startK_one, self._endK_one, self._density_offset_one, 0,
                                   density_profile_one, tau_one, path_length=self._path_length_one)

        drv.Context.synchronize()

        tau_two = zeros(shape=(total_layers_two, wngrid_size), dtype=np.float64)
        cuda_contribs_two = [c for c in m1.contribution_list if isinstance(c, CudaContribution)]
        noncuda_contribs_two = [c for c in m1.contribution_list if not isinstance(c, CudaContribution)]
        fully_cuda_two = len(noncuda_contribs_two) == 0 

        if not fully_cuda_two:
            tau_two.set(self.fallback_noncuda(total_layers_two, wngrid_size,
                                              path_length_two, m1.densityProfile, dz_two,noncuda_contribs_two)
                                              )

        for contrib in cuda_contribs_two:
            contrib.contribute(self, self._startK_two, self._endK_two, self._density_offset_two, 0,
                                   density_profile_two, tau_two, path_length=self._path_length_two)

        drv.Context.synchronize()
        #drv.memcpy_dtoh(self._tau_buffer_one[:wngrid_size,:], tau_one.gpudata)
        #drv.memcpy_dtoh(self._tau_buffer_two[:wngrid_size,:], tau_two.gpudata)

        #t1 = self._tau_buffer_one[:wngrid_size,:].reshape(m1.nLayers,wngrid_size)
        #t2 = self._tau_buffer_two[:wngrid_size,:].reshape(m2.nLayers,wngrid_size)
        #self.debug('tau one %s %s', tau_one, tau_one.shape)
        #self.debug('tau two %s %s', tau_two, tau_two.shape)
        absorption, tau_one = self.compute_absorption(tau_one, dz_one, ap_one, tau_two, dz_two, ap_two)

        return absorption, tau_one


    def compile_altitude(self, altitude, new_altitude):
        
        lenA = len(altitude)
        a_min = np.digitize(new_altitude, altitude).astype(np.int32)-1
        np.clip(a_min, 0, lenA-1,out=a_min)
        a_max = a_min+1
        np.clip(a_max, 0, lenA-1,out=a_max)

        return a_min, a_max



    def compute_absorption(self, tau_one, dz_one, ap_one,  tau_two, dz_two, ap_two):
        pradius = self._planet.fullRadius
        sradius = self._star.radius

        interp_kernel = kernal_func(self._ngrid)

        new_alt = np.sort(np.unique(np.concatenate([ap_one, ap_two])))
        new_alt_gpu = to_gpu(new_alt, self._memory_pool.allocate)
        num_new_alt = new_alt.shape[0]
        new_dz = np.gradient(new_alt)

        THREAD_PER_BLOCK_X = 16
        THREAD_PER_BLOCK_Y = 16
        
        NUM_BLOCK_X = int(math.ceil(self._ngrid/THREAD_PER_BLOCK_X))
        NUM_BLOCK_Y = int(math.ceil(num_new_alt/THREAD_PER_BLOCK_Y))

        rprs = zeros(shape=(self._ngrid), dtype=np.float64,allocator=self._memory_pool.allocate)
        new_tau = zeros(shape=(num_new_alt, self._ngrid), 
                        dtype=np.float64, allocator=self._memory_pool.allocate)


        ap = to_gpu(ap_one,allocator=self._memory_pool.allocate)
        apmin, apmax = self.compile_altitude(ap_one, new_alt)
        self.info('apmin %s', apmin)
        self.info('apmax %s', apmax)
        
        interp_kernel(new_tau, tau_one,ap,new_alt_gpu,
                       drv.In(apmin), drv.In(apmax), np.int32(num_new_alt),
                      block=(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y,1), grid=(NUM_BLOCK_X, NUM_BLOCK_Y,1) )

        ap = to_gpu(ap_two,allocator=self._memory_pool.allocate) 
        apmin, apmax = self.compile_altitude(ap_two, new_alt)
        interp_kernel(new_tau, tau_two,ap,new_alt_gpu,
                       drv.In(apmin), drv.In(apmax), np.int32(num_new_alt),
                      block=(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y,1), grid=(NUM_BLOCK_X, NUM_BLOCK_Y,1) )

        grid_size = self._ngrid
        rprs_kernal = absorption_kernal( grid_size)

        THREAD_PER_BLOCK_X = 256
        NUM_BLOCK_X = int(math.ceil(grid_size/THREAD_PER_BLOCK_X))

        rprs_kernal(rprs, new_tau, drv.In(new_dz), new_alt_gpu, np.int32(num_new_alt),
                    np.float64(self._planet.fullRadius),
                    np.float64(self._star.radius),
                    block=(THREAD_PER_BLOCK_X, 1, 1),
                    grid=(NUM_BLOCK_X, 1, 1))

        drv.memcpy_dtoh(self._tau_buffer[:self._ngrid*num_new_alt], new_tau.gpudata)
        
        final_tau = self._tau_buffer[:self._ngrid*num_new_alt].reshape(num_new_alt,self._ngrid)

        return rprs.get(), final_tau#((pradius**2.0) + integral)/(sradius**2), tau_one
