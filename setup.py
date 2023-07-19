#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils.extension import Extension
import os


def build_ext(*args,**kwargs):
    from Cython.Distutils import build_ext as _build_ext
    return _build_ext(*args,**kwargs)
    
def cythonize(*args,**kwargs):
    from Cython.Build import cythonize as _cythonize
    return _cythonize(*args,**kwargs)
    
class CustomInstall(install):
    def run(self):
        command = "git clone https://github.com/boostorg/boost.git"
        #process = subprocess.Popen(command, shell=True, cwd="soapcw/line_aware_statistic")
        #process.wait()

        command1 = "git clone https://gitlab.com/GNU/GSL.git"
        #process1 = subprocess.Popen(command1, shell=True, cwd="soapcw/line_aware_statistic")
        #process1.wait()

        install.run(self)
    
    
cwd = os.getcwd()
homedir = os.path.expanduser("~")

soapcw_include_dirs = ["{}/soapcw/glibc_fix.h".format(cwd),
                     "{}/repositories/gsl/include/".format(homedir),
                     "{}/repositories/boost_1_71_0/boost/math/distributions/".format(homedir)]

int_include_dirs = ["/{}/repositories/boost_1_71_0/boost/math/distributions/".format(homedir),
                    "/{}/repositories/boost_1_71_0/".format(homedir),
                    "/{}/repositories/".format(homedir),
                    "{}/soapcw/glibc_fix.h".format(cwd),
                    "{}/soapcw/line_aware_statistic/".format(homedir),
                    "/{}/repositories/gsl/include/".format(homedir)]

int_libraries = ["gsl","gslcblas"]

int_library_dirs = ["/{}/repositories/gsl/lib/".format(homedir)]

# will include this later
lookup_module = Extension("line_aware_stat.gen_lookup",
                              ["src/soapcw/line_aware_stat/gen_lookup.pyx", "src/soapcw/line_aware_stat/integrals.cpp"],
                              language='c++',
                              include_dirs=int_include_dirs,
                              library_dirs=int_library_dirs,
                              libraries=int_libraries)

# decide whether to build lookup tables using the C++ code (is faster but requires external libraries)
cpp=False

if cpp:
    ext_modules = [
        Extension("soapcw.soap",
                  [ "src/soapcw/soap.pyx" ]),
        lookup_module,]
else:
    ext_modules = [
    Extension("soapcw.soap",
              [ "src/soapcw/soap.pyx" ]),]


cmdclass = { 'build_ext': build_ext , }

setup(
    zip_safe=False,
    ext_modules = cythonize(ext_modules),
    cmdclass=cmdclass,
)
