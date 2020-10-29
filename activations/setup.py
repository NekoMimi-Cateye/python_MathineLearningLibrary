from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

sourcefiles = ['sigmoid.pyx']
setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = [Extension('Sigmoid', sourcefiles)],
    include_dirs=[np.get_include()]
)