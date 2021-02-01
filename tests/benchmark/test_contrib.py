from taurex.contributions import contribute_tau
import numpy as np
import pytest
import pycuda.autoinit

NLAYERS = 300
WNGRID_SIZE = 100000


@pytest.fixture
def setup():
    from pycuda.gpuarray import to_gpu
    sigma = np.random.rand(NLAYERS, WNGRID_SIZE)
    density = np.random.rand(NLAYERS)
    path = np.random.rand(NLAYERS,NLAYERS)
    
    gpu_sigma = to_gpu(sigma)
    gpu_density = to_gpu(density)
    gpu_path = to_gpu(path)
    
    yield (sigma, density, path[0]), (gpu_sigma, gpu_density, gpu_path)



def test_contribute_tau_cpu(benchmark, setup):
    sigma, density, path = setup[0]
        # # startK, endK, density_offset, sigma, density, path, nlayers,
            #                ngrid, layer, tau
    tau = np.zeros(shape=(NLAYERS, WNGRID_SIZE))

    benchmark(contribute_tau, 0, NLAYERS, 0, sigma, density, path, NLAYERS, 
              WNGRID_SIZE, 0, tau)



def test_contribute_tau_cuda(benchmark, setup):
    from pycuda.gpuarray import zeros, to_gpu
    from taurex_cuda.contributions.cudacontribution import cuda_contribute_tau
    gpu_sigma, gpu_density, gpu_path = setup[1]

    startK = to_gpu(np.array([0 for x in range(NLAYERS)]).astype(np.int32))
    endK = to_gpu(np.array([NLAYERS-x for x in range(NLAYERS)]).astype(np.int32))
    offset = to_gpu(np.array(list(range(NLAYERS))).astype(np.int32))
    tau = zeros(shape=(NLAYERS, WNGRID_SIZE), dtype=np.float64)
    


    benchmark(cuda_contribute_tau,startK,endK,offset,gpu_sigma, gpu_density, gpu_path,
              NLAYERS, WNGRID_SIZE,tau=tau,start_layer=0)


def test_contribute_tau_cuda_old_kernel(benchmark, setup):
    from pycuda.gpuarray import zeros, to_gpu
    from taurex_cuda.contributions.cudacontribution import cuda_contribute_tau_old
    gpu_sigma, gpu_density, gpu_path = setup[1]

    startK = to_gpu(np.array([0 for x in range(NLAYERS)]).astype(np.int32))
    endK = to_gpu(np.array([NLAYERS-x for x in range(NLAYERS)]).astype(np.int32))
    offset = to_gpu(np.array(list(range(NLAYERS))).astype(np.int32))
    tau = zeros(shape=(NLAYERS, WNGRID_SIZE), dtype=np.float64)
    


    benchmark(cuda_contribute_tau_old,startK,endK,offset,gpu_sigma, gpu_density, gpu_path,
              NLAYERS, WNGRID_SIZE,tau=tau,start_layer=0)

