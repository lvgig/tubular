import re

import setuptools

with open("README.md") as fh:
    long_description = fh.read()

# get version from _version.py file, from below
# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
VERSION_FILE = "tubular/_version.py"
VERSION_STR_RE = r"^__version__ = ['\"]([^'\"]*)['\"]"
with open(VERSION_FILE) as version_file:
    version_file_str = version_file.read()
    mo = re.search(VERSION_STR_RE, version_file_str, re.M)
    if mo:
        version = mo.group(1)
    else:
        msg = f"Unable to find version string in {VERSION_FILE}."
        raise RuntimeError(msg)


def list_reqs(fname: str = "requirements.txt") -> list:
    with open(fname) as fd:
        return fd.read().splitlines()


setuptools.setup(
    name="tubular",
    version=version,
    author="LV GI Data Science Team",
    author_email="#DataSciencePackages@lv.co.uk",
    description="Package to perform pre processing steps for machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=list_reqs(),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
)
