.. Sit2StandPy installation file

Installation, Requirements, and Testing
=======================================

Installation
------------

Sit2StandPy can be installed by running any of the following in the terminal:

::

    pip install sit2standpy  # install checking for dependencies with pip
    pip install sit2standpy --no-deps  # install without checking for dependencies
    pip install git+https://github.com/PfizerRD/Sit2StandPy  # alternative to pull from the master branch


Requirements
------------
These requirements will be collected if not already installed, and should require no input from the user.

- Python >= 3.7
- NumPy
- SciPy
- pandas
- pywavelets
- udatetime

These are the versions developed on, and some backwards compatibility may be possible.

To run the tests, additionally the following are needed:

- pytest
- h5py

Testing
-------

Automated tests can be run with ``pytest`` through the terminal:

::

    pytest --pyargs sit2standpy.tests -v

