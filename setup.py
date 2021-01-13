import os
from setuptools import setup, find_packages

with open(os.path.join(os.getcwd(), 'configs/__VERSION__')) as fp:
    __VERSION__ = str(fp.read())

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

with open('requirements.txt') as f:
    required = f.read().splitlines()
setup(
    name="pipelineservice",
    version=__VERSION__,
    description="Auto machine learning, deep learning library in Python.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/toandaominh1997/pipelineservice",
    author="toandaominh1997",
    author_email="toandaominh1997@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    install_requires=required
)
