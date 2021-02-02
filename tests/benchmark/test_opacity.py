import pytest
import numpy as np

NLAYERS=300

@pytest.fixture
def opac():
    from taurex.opacity.fakeopacity import FakeOpacity
    fo = FakeOpacity('H2O',wn_res=100000)
    from taurex.cache import OpacityCache
    oc = OpacityCache()
    oc.clear_cache()
    oc.add_opacity(fo)
    from taurex_cuda.opacity.cudaopacity import CudaOpacity
    co = CudaOpacity('H2O')
    yield fo, co
    oc.clear_cache() 
    del co
    del fo


def test_cpu_opacity(benchmark, opac):
    
    def layer_function():
        for n in range(NLAYERS):
            opac[0].opacity(1000,1e1)


    benchmark(layer_function)


def test_cuda_opacity(benchmark, opac):
    
    T = np.ones(NLAYERS)*1000
    P = np.ones(NLAYERS)*1e1
    mix = np.ones(NLAYERS)
    benchmark(opac[1].opacity,T,P,mix)


def test_cuda_opacity_nocopy(benchmark, opac):
    from pycuda.gpuarray import GPUArray
    T = np.ones(NLAYERS)*1000
    P = np.ones(NLAYERS)*1e1
    mix = np.ones(NLAYERS)
    
    res = GPUArray(shape=(NLAYERS,opac[0].wavenumberGrid.shape[0]), dtype=np.float64)

    benchmark(opac[1].opacity,T,P,mix, dest=res)
