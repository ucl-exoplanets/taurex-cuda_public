from taurex.mpi import shared_comm, shared_rank
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import pycuda.gpuarray as gpuarray

def gpu_single_allocate(arr, logger=None, force_shared=False):
    """

    Converts a numpy array into an MPI shared memory.
    This allow for things like opacities to be loaded only
    once per node when using MPI. Only activates if mpi4py 
    installed and when enabled via the ``mpi_use_shared`` input::

        [Global]
        mpi_use_shared = True

    or ``force_shared=True`` otherwise does nothing and
    returns the same array back

    Parameters
    ----------

    arr: numpy array
        Array to convert

    logger: :class:`~taurex.log.logger.Logger`
        Logger object to print outputs

    force_shared: bool
        Force conversion to shared memory


    Returns
    -------
    array:
        If enabled and MPI present, shared memory version of array
        otherwise the original array

    """
    import os


    try:
        from mpi4py import MPI
    except ImportError:
        return gpuarray.to_gpu(arr)
    from taurex.cache import GlobalCache
    if GlobalCache()['gpu_allocate_single'] or force_shared:
        if logger is not None:
            logger.info('Moving to GPU once')
        comm = shared_comm()

        cuda_device = int(os.environ.get('CUDA_VISIBLE_DEVICES',0))
        myrank = shared_rank()
        
        cuda_comm = comm.Split(cuda_device, myrank)

        cuda_rank = cuda_comm.Get_rank()

        my_array = None
        shape, dtype = arr.shape, arr.dtype
        h = None
        if cuda_rank == 0:
            my_array = gpuarray.to_gpu(arr)
            h = drv.mem_get_ipc_handle(my_array.ptr)

        h,shape, dtype = cuda_comm.bcast((h, shape, dtype))

        if cuda_rank != 0:
            x_ptr = drv.IPCMemoryHandle(h)
            my_array = gpuarray.GPUArray(shape, dtype, gpudata=x_ptr)

        return my_array
    else:
        return gpuarray.to_gpu(arr)


