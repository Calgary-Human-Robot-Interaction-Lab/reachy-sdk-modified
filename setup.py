#!/usr/bin/env python

from os import path

from setuptools import find_packages, setup


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='reachy-sdk',
    version='0.1.0',
    packages=find_packages(exclude=['tests']),

    install_requires=[
        'reachy-sdk-api',
        'grpcio',
        'protobuf>3',
    ],

    author='Pollen Robotics',
    author_email='contact@pollen-robotics.com',
    url='https://github.com/pollen-robotics/reachy-sdk',

    description='Python Reachy SDK.',
    long_description=long_description,
    long_description_content_type='text/markdown',
)