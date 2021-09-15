#!/usr/bin/env python
import setuptools
from setuptools import find_packages
from distutils.core import setup
from distutils.core import Extension
from distutils import log
import re, os

packages = find_packages(exclude=('tests', 'doc'))
provides = ['taurex_cuda', ]

requires = []

install_requires = ['taurex', 'pycuda',]

entry_points = {'taurex.plugins': 'cuda = taurex_cuda'}

version="1.0.0"

classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries',
     'Environment :: GPU',
    'Environment :: GPU :: NVIDIA CUDA'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='taurex_cuda',
      author="Ahmed Faris Al-Refaie",
      author_email="ahmed.al-refaie.12@ucl.ac.uk",
      license="BSD",
      version=version,
      description='TauREx 3 accelerator plugin using nVidia CUDA',
      classifiers=classifiers,
      packages=packages,
      long_description=long_description,
      url='https://github.com/ucl-exoplanets/taurex-cuda_public/',
      long_description_content_type="text/markdown",
      keywords=['exoplanet',
                'retrieval',
                'taurex',
                'taurex3',
                'cuda',
                'gpu',
                'atmosphere',
                'atmospheric'],
      entry_points=entry_points,
      provides=provides,
      requires=requires,
      install_requires=install_requires)