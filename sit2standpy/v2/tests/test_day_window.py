import pytest
from tempfile import TemporaryFile

from sit2standpy.v2.day_window import WindowDays


class TestWindowDays:
    def test_h5(self, sample_h5):
        wd = WindowDays(hours=[8, 20])

        wd.predict(sample_h5)

        assert True
