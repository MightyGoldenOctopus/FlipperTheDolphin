#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup
from setuptools_rust import RustExtension

setup_requires = ['setuptools-rust>=0.10.2']
install_requires = ['numpy']
test_requires = install_requires + ['pytest']

setup(
    name='pso',
    version='0.1.0',
    description='PSO bindings',
    rust_extensions=[RustExtension(
        'pso.pso',
        './Cargo.toml',
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    packages=find_packages(),
    zip_safe=False,
)


