import setuptools
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

# get version from _version.py file, from below
# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
VERSION_FILE = "tubular/_version.py"
version_file_str = open(VERSION_FILE, "rt").read()
VERSION_STR_RE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERSION_STR_RE, version_file_str, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSION_FILE,))


def list_reqs(fname="requirements.txt"):
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
