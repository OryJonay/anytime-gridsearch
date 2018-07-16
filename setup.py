'''
Created on Aug 9, 2017

@author: Ory Jonay
'''

import os
from setuptools import setup, find_packages

from AnyTimeGridSearchCV import grids

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt'), 'r') as req_file:
    install_reqs = req_file.read().strip().split('\n')

setup(
    name="AnyTimeGridSearchCV",
    version=grids.__version__,
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.log', '*.json', '*.md', '*.html', '*.js', '*.png'], }
)
