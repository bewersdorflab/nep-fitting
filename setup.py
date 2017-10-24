#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='nep_fitting',
      version='1.3',
      description='Nested Ensemble Resolution Estimation',
      author='Andrew Barentine, Michael Graff, David Baddeley',
      author_email='andrew.barentine@yale.edu',
      packages=find_packages(),
     )
