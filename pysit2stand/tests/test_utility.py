import pytest
import pandas as pd
from pysit2stand.utility import Transition, mov_stats


@pytest.mark.parametrize(('start_time', 'stop_time'), (
        ("to_datetime(1567616049649, unit='ms')", "to_datetime(1567616049649 - 1e3, unit='ms')"),
        ("to_datetime(1567616049649, unit='ms')", "to_datetime(1567616049649 + 20e3, unit='ms')")))
def test_transition_errors(start_time, stop_time):
    with pytest.raises(ValueError) as e_info:
        Transition((start_time, stop_time))
