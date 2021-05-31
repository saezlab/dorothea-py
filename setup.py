#!/usr/bin/env python

from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='dorothea-py',
    version='1.0.4',
    author='Pau Badia i Mompel',
    author_email="pau.badia@uni-heidelberg.de",
    description='dorothea-py is a python package to compute TF activity \
    from RNA-seq data using DoRothEA as regulon resource',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/saezlab/dorothea-py',
    project_urls={
        "Bug Tracker": "https://github.com/saezlab/dorothea-py/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=['dorothea'], 
    license='LICENSE.txt',
    package_data={'dorothea': ['data/dorothea_hs.pkl', 
                              'data/dorothea_mm.pkl', 
                              'data/c_dorothea_hs.pkl', 
                              'data/c_dorothea_mm.pkl']
                },
    install_requires=[
        'anndata',
        'scanpy',
        'numpy',
        'pandas',
        'tqdm',
        'seaborn'
    ]
)
