from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Enable OpenMP support
extra_compile_args = ["-fopenmp"]
extra_link_args = ["-fopenmp"]

# Define extensions for 1D and 2D FFT
extensions = [
    Extension(
        "OpenMP_1D_FFT",
        ["OpenMP/OpenMP_1D_FFT.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()],
    ),
    Extension(
        "OpenMP_2D_FFT",
        ["OpenMP/OpenMP_2D_FFT.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions),
)
