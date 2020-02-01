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

class TransmissionCudaModelTwo(ForwardModel):


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
        
        self._tau_buffer_one = drv.pagelocked_zeros(shape=(self.nativeWavenumberGrid.shape[-1], m1.nLayers,),dtype=np.float64)

        self._startK_two = to_gpu(np.array([0 for x in range(m2.nLayers)]).astype(np.int32))
        self._endK_two = to_gpu(np.array([m2.nLayers-x for x in range(m2.nLayers)]).astype(np.int32))
        self._density_offset_two = to_gpu(np.array(list(range(m2.nLayers))).astype(np.int32))
        self._path_length_two = to_gpu(np.zeros(shape=(m2.nLayers, m2.nLayers)))
        
        self._tau_buffer_two = drv.pagelocked_zeros(shape=(self.nativeWavenumberGrid.shape[-1], m2.nLayers,),dtype=np.float64)



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
        absorp, tau_one, tau_two = self.path_integral(native_grid, False)

        return native_grid, absorp, tau_one, tau_two

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

        dz_one = np.gradient(self._model_one.altitudeProfile)
        dz_two = np.gradient(self._model_two.altitudeProfile)

        wngrid_size = wngrid.shape[0]
        
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

        #self.debug('tau one %s %s', tau_one, tau_one.shape)
        #self.debug('tau two %s %s', tau_two, tau_two.shape)
        absorption, tau_one, tau_two = self.compute_absorption(tau_one.get(), dz_one, ap_one, tau_two.get(), dz_two, ap_two)

        return absorption, tau_one, tau_two


    def compute_absorption(self, tau_one, dz_one, ap_one,  tau_two, dz_two, ap_two):
        pradius = self._planet.fullRadius
        sradius = self._star.radius




        #integral = self.compute_integral(tau, dz)
        # _dz_one = dz_one[:, None]
        # _ap_one = ap_one[:, None]
        # _dz_two = dz_two[:, None]
        # _ap_two = ap_two[:, None]

        # _ap_one_mapped = _ap_two
        # _dz_one_mapped = _dz_two

        # ### build tau at the model 2 altitude grid.
        # from scipy.interpolate import interp1d
        # f = interp1d(_ap_one, tau_one, axis=0)
        # tau_one_mapped = f(_ap_one_mapped)

        # _ap_one_mapped.append(_ap_one[_ap_one[:] > np.max(_ap_one_mapped)])
        # _dz_one_mapped.append(_dz_one[_ap_one[:] > np.max(_ap_one_mapped)])
        # tau_one_mapped.append(tau_one[_ap_one[:] > np.max(_ap_one_mapped),:])

        # supp = np.zeros(len(_ap_one_mapped)-len(_ap_two))
        # tau_two_mapped = tau_two
        # tau_one_mapped.append(supp)

        ###### we need a mapping for dz and ap here!!!
        ###### this assumes the structure is the same

        self.debug('Alt profile [0] %s', ap_one)
        self.debug('Alt profile [1] %s', ap_two)
        self.debug('Alt profile shape [0] %s', ap_one.shape)
        self.debug('Alt profile shape [1] %s', ap_two.shape)
        self.debug('tau shape [0] %s', tau_one.shape)
        self.debug('tau shape [1] %s', tau_two.shape)
        # integral = np.sum((pradius + ap_) * (1.0 - tau_one) * dz_one, axis=0)
        f1 = interp1d(ap_one,tau_one, copy=False,
                                    bounds_error=False,fill_value=0.0, axis=0,assume_sorted=True)

        self.debug('F1 done')
        f2 = interp1d(ap_two,tau_two, copy=False,
                                bounds_error=False,fill_value=0.0, axis=0,assume_sorted=True)

        self.debug('F2 done')

        self.debug('Alt profile [0] %s', ap_one)
        self.debug('Alt profile [1] %s', ap_two)
        new_alt =np.sort(np.unique(np.concatenate([ap_one, ap_two])))
        self.info('Number of altitude layers %s',new_alt.shape)
        self.debug('Altitude layers %s',new_alt)
        
        new_tau1 = f1(new_alt)
        self.info('New tau one %s',new_tau1)
        self.info('New tau one shape %s',new_tau1.shape)
        new_tau2 = f2(new_alt)
        self.info('New tau two %s', new_tau2)
        self.info('New tau two shape %s', new_tau2.shape)

        tau_one = np.exp(-new_tau1)
        tau_two = np.exp(-new_tau2)

        self.debug('tau [0] %s', tau_one)
        self.debug('tau shape [0] %s', tau_one.shape)
        self.debug('tau [1] %s', tau_two)
        self.debug('tau shape [1] %s', tau_two.shape)
        # tau_one = np.exp(-tau_one)
        # tau_two = np.exp(-tau_two)
        new_dz = np.gradient(new_alt)
        self.debug('new dz %s', new_dz)
        self.debug('new dz shape %s', new_dz.shape)
        #integral = np.sum((pradius + new_alt[:,None]) * (2.0 - tau_one) * new_dz[:,None]*2,axis=0)

        integral = np.sum((pradius + new_alt[:,None]) * (1.0 - tau_one*tau_two) * new_dz[:,None]*2.0,axis=0)
        # self.info('integral = %s',integral)
        # integral += np.sum((pradius + ap_two[:,None]) * (1.0 - tau_two) * dz_two[:,None],axis=0)
        # integral = np.sum((pradius + _ap_one_mapped) * (1.0 - tau_one) * _dz_one_mapped + (pradius + _ap_one_mapped) * (1.0 - tau_two) * _dz_one_mapped, axis=0)

        return ((pradius**2.0) + integral)/(sradius**2), tau_one, tau_two
