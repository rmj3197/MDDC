from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import setuptools

__version__ = "1.0.1"


class get_eigen_include(object):
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
        include_dirs=[str(get_eigen_include())],
        define_macros=[("VERSION_INFO", __version__)],
        language="c++",
    ),
]

setup(
    name="MDDC",
    version=__version__,
    author="Raktim Mukhopadhyay, Anran Liu, Marianthi Markatou",
    author_email="raktimmu@buffalo.edu, anranliu@buffalo.edu, markatou@buffalo.edu",
    url="https://github.com/pybind/mddc_cpp_helper",
    description="MDDC provides methods for detecting (adverse event, drug) signals, a data generating function for simulating pharmacovigilance data, and a few functions for pre-processing and computing statistics such as expectations or residuals.",
    long_description="",
    packages=setuptools.find_packages(),
    install_requires=["pybind11", "peigen"],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10, <3.13",
)
