import pytest
from pandas import to_datetime
from numpy import isclose, allclose, array, ones
from sit2standpy.utility import Transition, mov_stats


@pytest.mark.parametrize(('start_time', 'end_time', 'ttype', 'long_type', 'duration'), (
        (to_datetime(1.5e12, unit='ms'), to_datetime(1.5e12 + 1e3, unit='ms'), 'SiSt', 'Sit to Stand', 1.),
        (to_datetime(1.5e12, unit='ms'), to_datetime(1.5e12 + 1e3, unit='ms'), 'StSi', 'Stand to Sit', 1.)))
def test_transition_representations(start_time, end_time, ttype, long_type, duration):
    trans = Transition((start_time, end_time), t_type=ttype)

    assert str(trans) == 'Postural Transition'
    assert repr(trans) == f'{long_type} (Duration: {duration:.2f})'


def test_transition_input_errors(start_t1, end_t1):
    with pytest.raises(ValueError) as e_info:
        Transition({'start': start_t1, 'end': end_t1})
    with pytest.raises(ValueError) as e_info:
        Transition((start_t1, end_t1), t_type='Sit to Stand')


@pytest.mark.parametrize(('start_time', 'stop_time'), (
        (to_datetime(1.5e12, unit='ms'), to_datetime(1.5e12 - 1e3, unit='ms')),
        (to_datetime(1.5e12, unit='ms'), to_datetime(1.5e12 + 20e3, unit='ms'))))
def test_transition_time_errors(start_time, stop_time):
    with pytest.raises(ValueError) as e_info:
        Transition((start_time, stop_time))


@pytest.mark.parametrize(('start_time', 'stop_time', 'duration'), (
        (to_datetime(1.5e12, unit='ms'), to_datetime(1.5e12 + 1e3, unit='ms'), 1.0),
        (to_datetime(1.5e12, unit='ms'), to_datetime(1.5e12 + 3437.132, unit='ms'), 3.437132)))
def test_transition_duration(start_time, stop_time, duration):
    trans = Transition((start_time, stop_time))

    assert isclose(trans.duration, duration)


@pytest.mark.parametrize(('x', 'win', 'mn', 'sd', 'pad'), (
        (array([1, 2, 3, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 5, 4, 6, 7, 8, 9]), 5,
         array([2.2, 2.2, 2.2, 2.2, 2.8, 3.4, 4.0, 5.0, 5.6, 5.8, 5.6, 5.4, 5.0, 4.6, 4.8, 5.4, 6.0, 6.8, 5.4]),
         array([0.83666003, 0.83666003, 0.83666003, 0.83666003, 0.83666003, 1.14017543, 1.58113883, 1.58113883,
                1.14017543, 0.83666003, 1.14017543, 1.14017543, 0.70710678, 0.54772256, 0.83666003, 1.14017543,
                1.58113883, 1.92353841, 1.14017543]), 3),
        (array([1, 2, 3, 4, 5]), 1, array([1.5, 1.5, 2.5, 3.5, 4.5]), 0.70710678 * ones(5), 1)))
def test_mov_stats(x, win, mn, sd, pad):
    mean, st_dev, n_pad = mov_stats(x, win)

    assert allclose(mean, mn)
    assert allclose(st_dev, sd)
    assert isclose(n_pad, pad)
