from .cudaopacity import CudaOpacity
from taurex.cache.singleton import Singleton
from taurex.log import Logger


class CudaOpacityCache(Singleton):
    def init(self):
        self.opacity_dict = {}
        self.log = Logger('CudaCache') 


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
            self.opacity_dict[key] = CudaOpacity(key)
            return self.opacity_dict[key]
    def clear_cache(self):
        """
        Clears all currently loaded cross-sections
        """
        self.opacity_dict = {}