from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="OpenMP.OpenMP_2D_FFT",
        sources=["OpenMP/OpenMP_2D_FFT.pyx"],
        include_dirs=[np.get_include()], 
        extra_compile_args=['/openmp'],  
        extra_link_args=[]  
    )
]

setup(
    name="OpenMP FFT 2D",
    ext_modules=cythonize(extensions),
)
