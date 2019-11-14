import pytest
from numpy import isclose

from sit2standpy.quantify import TransitionQuantifier


def test_transition_quantifier_no_data(gen_raw_data):
    tq = TransitionQuantifier()

    transition = tq.quantify((gen_raw_data[-2], gen_raw_data[-1]), gen_raw_data[-3])

    assert isclose(transition.duration, 2.5)
    assert transition.v_displacement is None
    assert transition.max_acceleration is None
    assert transition.min_acceleration is None
    assert transition.max_v_velocity is None
    assert transition.min_v_velocity is None
    assert transition.sparc is None


def test_transition_quantifier(gen_raw_data):
    tq = TransitionQuantifier()
    transition = tq.quantify((gen_raw_data[-2], gen_raw_data[-1]), gen_raw_data[-3], raw_acc=gen_raw_data[0],
                             mag_acc_f=gen_raw_data[1], mag_acc_r=gen_raw_data[1], v_vel=gen_raw_data[2],
                             v_pos=gen_raw_data[3])

    assert isclose(transition.duration, 2.5)
    assert isclose(transition.v_displacement, 2.30953222)
    assert isclose(transition.max_acceleration, 11.53293795)
    assert isclose(transition.min_acceleration, 5.089386237)
    assert isclose(transition.max_v_velocity, 2.13211138)
    assert isclose(transition.min_v_velocity, -0.00814809789)
    assert isclose(transition.sparc, -2.062909)
