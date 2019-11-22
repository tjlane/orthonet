# -*- coding: utf8 -*-

import os
from setuptools import setup, find_packages

# Meta information
version = open('VERSION').read().strip()
dirname = os.path.dirname(__file__)

setup(
    name='orthonet',
    version=version,
    author='Poitevin, Moilane, Lane',
    author_email='thomas.joseph.lane@gmail.com',
    url='https://github.com/tjlane/orthonet',

    # Packages and depencies
    packages=['orthonet', 'orthonet.temsim'],
    package_dir={'orthonet': 'orthonet',
                 'orthonet.temsim' : 'orthonet/temsim'},
    install_requires=[
        'numpy',
        'torch'
    ],

    # Data files
    #package_data={
    #    'orthonet': [
    #        'test/data/*.*',
    #    ]
    #},

    # Scripts
    #entry_points={
    #    'console_scripts': [
    #        'python-boilerplate = python_boilerplate.__main__:main'],
    #},

    # Other configurations
    zip_safe=False,
    platforms='any',
)

