import pytest
from tempfile import TemporaryFile

from sit2standpy.v2.day_window import WindowDays
from sit2standpy.v2.filters import AccelerationFilter

from sit2standpy.v2.tests.conftest import BaseProcessTester


class TestWindowDays(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.process = WindowDays(hours=[8, 20])
        cls.test_keys = ['Processed/Sit2Stand/Day 1/Indices']


class TestAccelerationFilterMovingAverage(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.process = AccelerationFilter('gaus1', power_band=[0, 0.5], power_peak_kw=None, power_std_height=True,
                                         power_std_trim=0.5, reconstruction_method='moving average',
                                         lowpass_order=4, lowpass_cutoff=5, window=0.25)
        cls.test_keys = [
            'Processed/Sit2Stand/Day 1/Filtered Acceleration',
            'Processed/Sit2Stand/Day 1/Reconstructed Acceleration',
            'Processed/Sit2Stand/Day 1/Power',
            'Processed/Sit2Stand/Day 1/Power Peaks'
        ]
        # TODO add tests for processing with indices, errors?
