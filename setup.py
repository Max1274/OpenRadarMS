import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

#module = [Extension ('oscfar', sources=['oscfar.pyx'], include_dirs=[numpy.get_include()]),
 #         Extension ('music', sources = ['music.pyx'], include_dirs=[numpy.get_include()])]

setup (
    name = 'MyProject',
    ext_modules = cythonize(["*.pyx"]),
    include_dirs= numpy.get_include()
)