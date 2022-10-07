#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Olivier COUDRAY",
    author_email='olivier.coudray.15@polytechnique.org',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="PU learning under SAR assumption with unknown propensity. Implementation of the general SAR-EM algorithm.",
    entry_points={
        'console_scripts': [
            'pysarpu=pysarpu.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pysarpu',
    name='pysarpu',
    packages=find_packages(include=['pysarpu', 'pysarpu.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ocoudray/pysarpu',
    version='0.1.0',
    zip_safe=False,
)
