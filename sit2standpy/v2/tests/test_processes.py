import pytest
from tempfile import TemporaryFile

from sit2standpy.v2.day_window import WindowDays

from sit2standpy.v2.tests.conftest import BaseProcessTester


class TestWindowDays(BaseProcessTester):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.process = WindowDays(hours=[8, 20])
        cls.test_keys = ['Processed/Sit2Stand/Day 1/Indices']
