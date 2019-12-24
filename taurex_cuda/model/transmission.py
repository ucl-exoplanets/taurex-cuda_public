import numpy as np
from taurex.model.simplemodel import SimpleForwardModel
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu, zeros

class TransmissionCudaModel(SimpleForwardModel):
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
                 atm_max_pressure=1e6):

        super().__init__(self.__class__.__name__, planet,
                         star,
                         pressure_profile,
                         temperature_profile,
                         chemistry,
                         nlayers,
                         atm_min_pressure,
                         atm_max_pressure)

    def compute_path_length(self, dz):

        

        planet_radius = self._planet.fullRadius
        total_layers = self.nLayers

        dl = np.zeros(shape=(total_layers, total_layers),dtype=np.float64)

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

            dl[layer, :k.shape[0]] = (k*2.0)[:]

        return to_gpu(dl)

    def path_integral(self, wngrid, return_contrib):

        dz = np.gradient(self.altitudeProfile)

        wngrid_size = wngrid.shape[0]

        path_length = self.compute_path_length(dz)

        density_profile = to_gpu(self.densityProfile)

        total_layers = self.nLayers

        startK = to_gpu(np.array([0 for x in range(total_layers)]))
        endK = to_gpu(np.array([100-x for x in range(total_layers)]))
        density_offset = to_gpu(np.array(list(range(total_layers))))
        tau = zeros(shape=(total_layers, wngrid_size), dtype=np.float64)
        for contrib in self.contribution_list:
            contrib.contribute(self, startK, endK, density_offset, 0,
                                   density_profile, tau, path_length=path_length)


        #self.debug('tau %s %s', tau, tau.shape)

        #absorption, tau = self.compute_absorption(tau, dz)
        return None,tau #absorption, tau

    def compute_absorption(self, tau, dz):

        tau = np.exp(-tau)
        ap = self.altitudeProfile[:, None]
        pradius = self._planet.fullRadius
        sradius = self._star.radius
        _dz = dz[:, None]

        integral = np.sum((pradius+ap)*(1.0-tau)*_dz*2.0, axis=0)
        return ((pradius**2.0) + integral)/(sradius**2), tau