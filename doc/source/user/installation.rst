.. _installation:

============
Installation
============

TauREx-CUDA requires the latest nVidia CUDA toolkit_. In particular, you must have access to the 
``nvcc`` compiler. This can be tested by running on the terminal or command prompt::

    > nvcc

This will either be available as a module in your HPC cluster::

    > module load cuda
    > nvcc

Downloaded and installed from the offical nVidia toolkit_ page. Or
downloaded for Mac and Linux from conda-forge::

    > conda install -c conda-forge cudatoolkit-dev
    > nvcc

Once installed the plugin can be simply installed using::

    > pip install taurex_cuda

Or for the latest development version, directly from git::

    > pip install git+https://github.com/ucl-exoplanets/taurex-cuda.git

You can verify if the plugin is functioning by seeing if TauREx successfully detects
``cuda``::

    > taurex --plugins

    Successfully loaded plugins
    ---------------------------
    venotdiseq
    ggchem
    bhmie
    cuda








.. _toolkit: https://developer.nvidia.com/cuda-toolkit
