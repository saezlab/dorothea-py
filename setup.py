from distutils.core import setup
setup(
   name='dorothea',
   version='0.0.1',
   description='Dorothea is a python package to compute TF activity using SCIRA\' linear models',
   long_description=open('README.md').read(),
   author='Pau Badia i Mompel',
   url='',
   packages=['dorothea'],  
   install_requires=['numpy >= 1.16.1'], 
   license='LICENSE.txt'
)