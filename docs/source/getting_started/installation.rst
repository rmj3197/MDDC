Installation
=============

.. title:: Getting Started : contents
.. _installation:


Which Python?
--------------

You’ll need **Python 3.10 or greater** (>=3.10, <3.13).

Installation using ``pip``
----------------------------
MDDC is availble on PyPI and can be installed using ``pip install MDDC``.

Dependencies
-------------
MDDC requires the following (arranged alphabetically):

- `matplotlib <https://matplotlib.org/>`_ (>=3.8.2)
- `numpy <https://numpy.org/>`_  (>= 1.26.2)
- `pandas <https://pandas.pydata.org/docs/index.html>`_ (>= 2.1.3)
- `peigen <https://github.com/fred3m/peigen>`_ (>=0.0.9)
- `pybind11 <https://pybind11.readthedocs.io/en/stable/>`_ (2.9.0)
- `scipy <https://docs.scipy.org/doc/scipy/reference/>`_ (>= 1.11.0)

Testing
--------
``MDDC`` uses the Python ``pytest`` package.  
To install ``pytest``, please go `here <https://docs.pytest.org/en/latest/getting-started.html#>`_.
To run the tests using ``pytest``, please follow these `instructions <https://docs.pytest.org/en/latest/how-to/usage.html>`_.
Navigate to the tests folder to run the tests. 