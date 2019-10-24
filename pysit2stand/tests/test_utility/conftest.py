import pytest
from pandas import to_datetime, date_range
from numpy import loadtxt, random
from importlib import resources


@pytest.fixture
def start_t1():
    return to_datetime(1567616049649, unit='ms')


@pytest.fixture
def end_t1():
    return to_datetime(1567616049649 + 1e3, unit='ms')