from .cudacontribution import CudaContribution
import numpy as np
from ..opacity.cudacache import CudaOpacityCache
from pycuda.gpuarray import GPUArray
from pycuda.tools import DeviceMemoryPool

class AbsorptionCuda(CudaContribution):
    """
    Computes the contribution to the optical depth
    occuring from molecular absorption.
    """

    def __init__(self):
        super().__init__('Absorption')
        self._opacity_cache = CudaOpacityCache()
        self._xsec_cache = {}
        self._memory_pool = DeviceMemoryPool()

    def build(self, model):
        super().build(model)
        self._opacity_cache.set_native_grid(model.nativeWavenumberGrid)

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
        # Loop through all active gases
        for gas in model.chemistry.activeGases:


            # Get the mix ratio of the gas
            gas_mix = model.chemistry.get_gas_mix_profile(gas)
            self.info('Recomputing active gas %s opacity', gas)

            # Get the cross section object relating to the gas
            xsec = self._opacity_cache[gas]
            xsec.opacity(model.temperatureProfile, model.pressureProfile,
                         gas_mix, wngrid=wngrid, dest=sigma_xsec)

            # Temporarily assign to master cross-section
            self.sigma_xsec = sigma_xsec
            yield gas, sigma_xsec
        sigma_xsec = None
    @classmethod
    def input_keywords(cls):
        return ['AbsorptionCUDA', ]