#!/usr/bin/env python

from distutils.core import setup
setup(
   name='dorothea',
   version='1.0',
   description='Dorothea is a python package to compute TF activity from RNA-seq data using SCIRA method',
   author='Pau Badia i Mompel',
   url='https://github.com/saezlab/dorothea-py',
   packages=['dorothea'], 
   license='LICENSE.txt',
   package_data={'dorothea': ['data/dorothea_hs.pkl', 'data/dorothea_mm.pkl']},
   install_requires=[
        'anndata',
        'scanpy',
        'numpy']
)
