#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
from distutils.extension import Extension
import os


def build_ext(*args,**kwargs):
    from Cython.Distutils import build_ext as _build_ext
    return _build_ext(*args,**kwargs)
    
def cythonize(*args,**kwargs):
    from Cython.Build import cythonize as _cythonize
    return _cythonize(*args,**kwargs)


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()
    
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




dep_links = []

# will include this later
lookup_module =     Extension("soapcw.line_aware_stat.gen_lookup",
                              ["soapcw/line_aware_stat/gen_lookup.pyx", "soapcw/line_aware_stat/integrals.cpp"],
                              language='c++',
                              include_dirs=int_include_dirs,
                              library_dirs=int_library_dirs,
                              libraries=int_libraries)

# decide whether to build lookup tables using the C++ code (is faster but requires external libraries)
cpp=False

if cpp:
    ext_modules = [
        Extension("soapcw.soapcw",
                  [ "soapcw/soapcw.pyx" ]),
        lookup_module,]
else:
    ext_modules = [
    Extension("soapcw.soapcw",
              [ "soapcw/soapcw.pyx" ]),]


cmdclass = { 'build_ext': build_ext , }

setup_requirements = ["cython", "matplotlib", "numpy", "torch", "torch-summary", "scipy", "lalsuite", "regex"]

test_requirements = ["cython"]

setup(
    author="Joe Bayley",
    author_email='joseph.bayley@glasgow.ac.uk',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="SOAP is a rapid algorithm to search for continuous sources of gravitational waves, with a wider application to long duration narrowband signals.",
    install_requires=setup_requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    #package_data = {"":['cnn/pytorch/trained_models/O3/*.ckpt','cnn/pytorch/trained_models/O3/*.ini']},
    keywords=['soapcw','soap','gravitational waves','pulsars','neutron stars','continuous waves'],
    name='soapcw',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://git.ligo.org/joseph.bayley/soapcw',
    download_url='https://git.ligo.org/joseph.bayley/soapcw/-/archive/0.1.2/soapcw-0.1.2.tar.gz',
    zip_safe=False,
    ext_modules = cythonize(ext_modules),
    cmdclass=cmdclass,
)
