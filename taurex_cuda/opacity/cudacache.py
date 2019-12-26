from .cudaopacity import CudaOpacity
from ..cia.cudacia import CudaCIA
from taurex.cache.singleton import Singleton
from taurex.log import Logger
import numpy as np


class CudaCache(Singleton):
    def init(self):
        self.opacity_dict = {}
        self.log = Logger(self.__class__.__name__) 
        self._wngrid = None
    
    def set_native_grid(self, native_grid):
        if self._wngrid is None or \
            not np.array_equal(native_grid, self._wngrid):

            self.log.info('Re-homogenizing native grids!')
            self._wngrid = native_grid

            for opac in self.opacity_dict.values():
                opac.transfer_xsec_grid(self._wngrid)

    def create_object(self, key, wngrid):
        raise NotImplementedError


    def __getitem__(self,key):
        """
        For a molecule return the relevant :class:`~taurex.opacity.opacity.Opacity` object.


        Parameter
        ---------
        key : str
            molecule name

        Returns
        -------
        :class:`~taurex.opacity.pickleopacity.PickleOpacity`
            Cross-section object desired
        
        Raise
        -----
        Exception
            If molecule could not be loaded/found

        """
        if key in self.opacity_dict:
            return self.opacity_dict[key]
        else:
            #Try a load of the opacity
            self.opacity_dict[key] = self.create_object(key, wngrid=self._wngrid)
            return self.opacity_dict[key]
    def clear_cache(self):
        """
        Clears all currently loaded cross-sections
        """
        self.opacity_dict = {}


class CudaOpacityCache(CudaCache):

    def create_object(self, key, wngrid):
        return CudaOpacity(key, wngrid=self._wngrid)

class CudaCiaCache(CudaCache):

    def create_object(self, key, wngrid):
        return CudaCIA(key, wngrid=self._wngrid)