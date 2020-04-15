import pytest
from tempfile import TemporaryFile

from sit2standpy.v2.day_window import WindowDays
from sit2standpy.v2.filters import AccelerationFilter
from sit2standpy.v2.detectors import Detector
from sit2standpy.v2.pipeline import Sequential

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
        cls.process = AccelerationFilter('gaus1', power_band=[0, 0.5], power_peak_kw={'distance': 128},
                                         power_std_height=True, power_std_trim=0,
                                         reconstruction_method='moving average', lowpass_order=4, lowpass_cutoff=5,
                                         window=0.25)
        cls.test_keys = [
            'Processed/Sit2Stand/Day 1/Filtered Acceleration',
            'Processed/Sit2Stand/Day 1/Reconstructed Acceleration',
            'Processed/Sit2Stand/Day 1/Power',
            'Processed/Sit2Stand/Day 1/Power Peaks'
        ]
        # TODO add tests for processing with indices, errors?


class TestDetectorStillness(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.process = Detector(stillness_constraint=True, gravity=9.81, thresholds=None,
                               gravity_pass_order=4, gravity_pass_cutoff=0.8, long_still=0.5, moving_window=0.3,
                               duration_factor=10, displacement_factor=0.75)
        cls.processed_keys = [
            'Processed/Sit2Stand/Day 1/Filtered Acceleration',
            'Processed/Sit2Stand/Day 1/Reconstructed Acceleration',
            'Processed/Sit2Stand/Day 1/Power Peaks'
        ]
        cls.test_keys = [
            'Processed/Sit2Stand/Day 1/Stillness Method/STS Times',
            'Processed/Sit2Stand/Day 1/Stillness Method/Duration',
            'Processed/Sit2Stand/Day 1/Stillness Method/Max. Accel.',
            'Processed/Sit2Stand/Day 1/Stillness Method/Min. Accel.',
            'Processed/Sit2Stand/Day 1/Stillness Method/SPARC'
        ]


class TestDetectorDisplacement(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.process = Detector(stillness_constraint=False, gravity=9.81, thresholds=None,
                               gravity_pass_order=4, gravity_pass_cutoff=0.8, long_still=0.5, moving_window=0.3,
                               duration_factor=10, displacement_factor=0.75)
        cls.processed_keys = [
            'Processed/Sit2Stand/Day 1/Filtered Acceleration',
            'Processed/Sit2Stand/Day 1/Reconstructed Acceleration',
            'Processed/Sit2Stand/Day 1/Power Peaks'
        ]
        cls.test_keys = [
            'Processed/Sit2Stand/Day 1/Displacement Method/STS Times',
            'Processed/Sit2Stand/Day 1/Displacement Method/Duration',
            'Processed/Sit2Stand/Day 1/Displacement Method/Max. Accel.',
            'Processed/Sit2Stand/Day 1/Displacement Method/Min. Accel.',
            'Processed/Sit2Stand/Day 1/Displacement Method/SPARC'
        ]


class TestStillnessSequentialPipeline(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        seq = Sequential()
        seq.add(WindowDays(hours=[8, 20]))
        seq.add(AccelerationFilter('gaus1', power_band=[0, 0.5], power_peak_kw={'distance': 128}, power_std_height=True,
                                   power_std_trim=0, reconstruction_method='moving average', lowpass_order=4,
                                   lowpass_cutoff=5, window=0.25))
        seq.add(Detector(stillness_constraint=True, gravity=9.81, thresholds=None, gravity_pass_order=4,
                         gravity_pass_cutoff=0.8, long_still=0.5, moving_window=0.3, duration_factor=10,
                         displacement_factor=0.75))

        cls.process = seq

        cls.test_keys = [
            'Processed/Sit2Stand/Day 1/Filtered Acceleration',
            'Processed/Sit2Stand/Day 1/Reconstructed Acceleration',
            'Processed/Sit2Stand/Day 1/Power',
            'Processed/Sit2Stand/Day 1/Power Peaks',
            'Processed/Sit2Stand/Day 1/Stillness Method/STS Times',
            'Processed/Sit2Stand/Day 1/Stillness Method/Duration',
            'Processed/Sit2Stand/Day 1/Stillness Method/Max. Accel.',
            'Processed/Sit2Stand/Day 1/Stillness Method/Min. Accel.',
            'Processed/Sit2Stand/Day 1/Stillness Method/SPARC'
        ]


class TestDisplacementSequentialPipeline(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        seq = Sequential()
        seq.add(WindowDays(hours=[8, 20]))
        seq.add(AccelerationFilter('gaus1', power_band=[0, 0.5], power_peak_kw={'distance': 128}, power_std_height=True,
                                   power_std_trim=0, reconstruction_method='moving average', lowpass_order=4,
                                   lowpass_cutoff=5, window=0.25))
        seq.add(Detector(stillness_constraint=False, gravity=9.81, thresholds=None, gravity_pass_order=4,
                         gravity_pass_cutoff=0.8, long_still=0.5, moving_window=0.3, duration_factor=10,
                         displacement_factor=0.75))

        cls.process = seq

        cls.test_keys = [
            'Processed/Sit2Stand/Day 1/Filtered Acceleration',
            'Processed/Sit2Stand/Day 1/Reconstructed Acceleration',
            'Processed/Sit2Stand/Day 1/Power',
            'Processed/Sit2Stand/Day 1/Power Peaks',
            'Processed/Sit2Stand/Day 1/Displacement Method/STS Times',
            'Processed/Sit2Stand/Day 1/Displacement Method/Duration',
            'Processed/Sit2Stand/Day 1/Displacement Method/Max. Accel.',
            'Processed/Sit2Stand/Day 1/Displacement Method/Min. Accel.',
            'Processed/Sit2Stand/Day 1/Displacement Method/SPARC'
        ]