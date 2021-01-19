.. _overview:

========
Overview
========

The TauREx-CUDA plugin supplies drop-in models and contribution functions
that allow for the exploitation of nVidia CUDA hardware.

.. warning::
    TauREx-CUDA is only compatible with the cross-section opacity method.

To make use of the plugin, you can simply replace existing models and contributions
with the CUDA-enabled ones. For example if we have a transit spectrum input file::

    [Model]
    model_type = transmission
        [[Absorption]]

        [[CIA]]
        cia_pairs = H2-H2, H2-He

        [[Rayleigh]]

The CUDA enabled version is simply::

    [Model]
    model_type = transmission_cuda
        [[AbsorptionCUDA]]

        [[CIACUDA]]
        cia_pairs = H2-H2, H2-He

        [[RayleighCUDA]]

The CUDA forward models can function with non-CUDA enabled contribution functions.
We can mix the C++ Mie scattering code for the eclipse version::

    [Model]
    model_type = emission_cuda
        [[AbsorptionCUDA]]

        [[CIACUDA]]
        cia_pairs = H2-H2, H2-He

        [[RayleighCUDA]]
        
        [[BHMie]]

When non-CUDA contributions are included, they are first computed on the CPU before their
results are transferred and the model is continued on the GPU.

.. warning::

    The oppisite is not true. CUDA contributions are only compatible with CUDA forward models.
    Attempting to use them in non-CUDA forward models will result in an error.


