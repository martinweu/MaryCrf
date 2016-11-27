#from distutils.core import setup

from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import numpy
import io
import codecs
import os
import sys

from Cython.Build import cythonize

# run form commandline: py.test --cov=.
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
  name = 'MaryCrf',
  version='1.0.0',
  author='Martin Weber',
  tests_require=['pytest'],
  install_requires=['cythonize',
                     'numpy',
                     ],
  cmdclass={'test': PyTest},
  author_email='napster2202@gmail.com',
  description='A conditional random field written in python/cython',
  ext_modules = cythonize('./marycrf/*.pyx'),
  include_dirs=[numpy.get_include()]
)
