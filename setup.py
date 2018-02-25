'''
Created on Aug 9, 2017

@author: Ory Jonay
'''

import os
import uuid

from pip.req import parse_requirements
from setuptools import setup, find_packages

from AnyTimeGridSearchCV import grids

install_reqs = parse_requirements(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                               'requirements.txt'), session=uuid.uuid1())

setup(
    name = "AnyTimeGridSearchCV",
    version = grids.__version__,
    packages = find_packages(),
    include_package_data = True,
    package_data={'': ['*.log', '*.json', '*.md', '*.html', '*.js', '*.png'],}
    )
