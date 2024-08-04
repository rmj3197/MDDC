from importlib import metadata, import_module

__version__ = "1.0.0"

submodules = ["MDDC", "utils", "datasets"]
__all__ = submodules + [__version__]


def __dir__():
    return __all__


# taken from scipy
def __getattr__(name):
    if name in submodules:
        return import_module(f"MDDC.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'MDDC' has no attribute '{name}'")
