from pytest import fixture
from numpy import arange, sin, pi, exp, random, sign, zeros
from scipy.integrate import cumtrapz
from pandas import to_datetime


@fixture
def gen_raw_data():
    fs = 100
    x = arange(-0.5, 2.0, 1 / fs)
    y = sin(pi * x) * exp(x) + 9.8

    random.seed(10)
    noise = random.randn(y.size)
    raw_y = zeros((y.size, 3))
    raw_y[:, 2] = y + 0.35 * (noise**2 * sign(noise))

    v_vel = cumtrapz(y, dx=1/fs, initial=0)
    v_vel -= ((v_vel[-1] - v_vel[0]) / (x[-1] - x[0])) * (x + 0.5)

    v_pos = cumtrapz(v_vel, dx=1/fs, initial=0)

    t0 = to_datetime(1.5e9, unit='s')
    t1 = to_datetime(1.5e9 + 2.5, unit='s')

    return raw_y, y, v_vel, v_pos, fs, t0, t1
