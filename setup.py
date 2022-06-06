#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pytcherplants',
    version='0.0.1',
    description='pitcher geometry & color analysis for top-down images of Sarracenia ',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Wes Bonelli',
    author_email='wbonelli@uga.edu',
    license='BSD-3-Clause',
    url='https://github.com/w-bonelli/pytcherplants',
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pypl = pytcherplants.cli:cli'
        ]
    },
    python_requires='>=3.6.8',
    install_requires=['requests', 'httpx', 'click', 'tenacity', 'tqdm', 'pytest', 'pytest-dotenv', 'pytest-asyncio'],
    setup_requires=['wheel'],
    tests_require=['pytest', 'coveralls', 'dotenv', 'pytest-asyncio'])