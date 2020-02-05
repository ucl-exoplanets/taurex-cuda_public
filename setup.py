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

setup(name='taurex_cuda',
      author="Ahmed Faris Al-Refaie",
      author_email="ahmed.al-refaie.12@ucl.ac.uk",
      license="BSD",
      description='Plugin to compute forward models using nVidia gpus ',
      packages=packages,
      
      entry_points=entry_points,
      provides=provides,
      requires=requires,
      install_requires=install_requires)