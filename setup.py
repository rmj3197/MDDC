from pathlib import Path

import pybind11
import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "1.1.dev0"

AUTHOR = "Raktim Mukhopadhyay, Anran Liu, Marianthi Markatou"
AUTHOR_EMAIL = "raktimmu@buffalo.edu, anranliu@buffalo.edu, markatou@buffalo.edu"
URL = "https://github.com/rmj3197/MDDC"

REQUIRED_PACKAGES = [
    "joblib>=1.2.0",
    "matplotlib>=3.7.1,<3.9.0",
    "numpy>=1.26.2",
    "pandas>=2.1.3",
    "peigen>=0.0.9",
    "pybind11>=2.9.0",
    "scipy>=1.11.0",
]

DESCRIPTION = """Methods for detecting signals related to adverse event and medical product (e.g., drugs, vaccines) pairs. This includes a data generation function for simulating pharmacovigilance datasets, along with various utility functions. For more details, please see - Liu A, Mukhopadhyay R, Markatou M (2024). “MDDC: An R and Python Package for Adverse Event Identification in Pharmacovigilance Data.” arXiv preprint. https://doi.org/10.48550/arXiv.2410.01168."""

this_directory = Path(__file__).parent
LONG_DESCRIPTION = (this_directory / "README.md").read_text()

class get_eigen_include:
    """Helper class to determine the Eigen include path
    The purpose of this class is to postpone importing Eigen
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import peigen
        return peigen.header_path


ext_modules = [
    Pybind11Extension(
        "mddc_cpp_helper",
        ["src/main.cpp"],
        include_dirs=[str(pybind11.get_include()), str(get_eigen_include())],
        define_macros=[("VERSION_INFO", __version__)],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="MDDC",
    version=__version__,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    description=DESCRIPTION,
    description_content_type='text/markdown',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    ext_modules=ext_modules,
    include_package_data=True,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10, <3.13",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython"
    ],
)
