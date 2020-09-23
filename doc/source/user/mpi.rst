

==============
Using with MPI
==============

TauREx-CUDA can be used with MPI to perform retrievals
using multiple GPUs with samplers such as MultiNest_.

Assuming an we have our ``[Model]``, ``[Optimizer]`` and
``[Fitting]`` sections setup like so::
    [Model]
    model_type = transmission_cuda
        [[AbsorptionCUDA]]

        [[CIACUDA]]
        cia_pairs = H2-H2, H2-He

        [[RayleighCUDA]]
    
    [Optimizer]
    optimizer = multinest
    multi_nest_path = ./multinest
    num_live_points = 1000

    [Fitting]
    T:fit = True
    T:priors = "Uniform(bounds=[1000,2000])"

We can run the retrieval with 2 cores under MPI::

    > mpirun -n 2 taurex -i science.par --retrieval

This will run 2 tasks on a single GPU. Increasing the n number will force all tasks onto this GPU.
For multiple GPU systems we can control which GPU is used through the ``CUDA_VISIBLE_DEVICES``
environment variable. For example to switch to a second GPU we set the variable::

    > export CUDA_VISIBLE_DEVICES=1
    > mpirun -n 2 taurex -i science.par --retrieval

However again, all tasks will use this GPU.

Multi-GPU
---------

To make use of multiple GPUs in an MPI run we will need to *bind* the 
TauREx instances to a specific GPU. We can accomplish this with a binding
script that sets th ``CUDA_VISIBLE_DEVICES`` environment at run-time. 
This will vary slightly with different MPI implementations but for 
OpenMPI we can use a modified version_ of ``gpu_bind.sh``::

    #!/usr/bin/bash

    export TOTAL_GPUS=2
    export PROC_ID=$OMPI_COMM_WORLD_LOCAL_RANK


    export CUDA_VISIBLE_DEVICES=$((PROC_ID % TOTAL_GPUS))

    $@

Here ``OMPI_COMM_WORLD_LOCAL_RANK`` is an environment variable set by OpenMPI that gives the task rank
and ``TOTAL_GPUS`` are number of GPUs available (per node).
We can therefore use this script with taurex like so::

    > mpirun -n 2 ./gpu_bind.sh taurex -i science.par --retrieval

Each task will run on a seperate GPU. Depending on the GPU and retrieval, we can run 2 tasks on each
GPU by doubling the number of tasks::

    > mpirun -n 4 ./gpu_bind.sh taurex -i science.par --retrieval

This can potentially double performance for GPUs with large memory and number of cores. 
For the V100 it is possible to run 3 tasks without affecting performance too much::

    > mpirun -n 6 ./gpu_bind.sh taurex -i science.par --retrieval

For different clusters the ``PROC_ID`` and ``TOTAL_GPUS`` may need to be modified.
For Wilkes ``TOTAL_GPUS=4``, for SLURM schedulers you can set ``$PROC_ID=$SLURM_PROCID``.
Check example scripts for your HPC centre to determine the best variables to use.




.. _MultiNest: https://github.com/JohannesBuchner/MultiNest
.. _version: https://medium.com/@jeffrey_91423/binding-to-the-right-gpu-in-mpi-cuda-programs-263ac753d232