from pytest import fixture
from numpy import sin, cos, arange, pi


@fixture
def integrate_data():
    dt = 0.01
    x = arange(0, 2 * pi, dt)

    f = cos(x)
    F = sin(x)
    FF = -cos(x) + 1  # need to add one due to intial value of the integration being 0

    return f, F, FF, dt
