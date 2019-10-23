from numpy import array, loadtxt, allclose
import pysit2stand as s2s
from pandas import to_datetime
from sys import version_info
if version_info < (3, 7):
    import pkg_resources as pkr
else:
    from importlib import resources


# UTILITY
def _test_mov_stats():
    x = array([0, 1, 2, 3, 4, 5])

    mn, sd, pad = s2s.utility.mov_stats(x, 3)

    assert (mn == array([1., 1., 1., 2., 3., 4.])).all()
    assert (sd == array([1., 1., 1., 1., 1., 1.])).all()
    assert pad == 2


def _test_transition():
    t1 = to_datetime(1567616049649, unit='ms')
    t2 = to_datetime(1567616049649 + 1e3, unit='ms')

    trans = s2s.Transition(times=(t1, t2))

    assert trans.duration == 1.


def _test_stillness():
    if version_info < (3, 7):
        file_path = pkr.resource_filename('pysit2stand.data', 'sample.csv')
        data = loadtxt(file_path, delimiter=',')
    else:
        with resources.path('pysit2stand.data', 'sample.csv') as file_path:
            data = loadtxt(file_path, delimiter=',')
    time = data[:, 0]
    acc = data[:, 1:]
    a_time = to_datetime(time, unit='us', utc=True).tz_convert('EST').tz_localize(None)

    a_filt = s2s.AccFilter(lowpass_order=4, lowpass_cutoff=5, discrete_wavelet='dmey',
                           reconstruction_level=1, reconstruction_method='moving average')

    ths = {'stand displacement': 0.125, 'transition velocity': 0.2, 'accel moving avg': 0.2, 'accel moving std': 0.1,
           'jerk moving avg': 2.5, 'jerk moving std': 3}
    still = s2s.detectors.Stillness(thresholds=ths, long_still=0.5, moving_window=0.3, duration_factor=10,
                                    lmax_kwargs=None, lmin_kwargs=dict(height=-9.5))

    cwave = 'gaus1'
    pk_pwr = [0, 0.5]
    pk_pwr_kw = {'height': 90, 'distance': 128}
    wavelet_ptd = s2s.Sit2Stand(continuous_wavelet=cwave, peak_pwr_band=pk_pwr, peak_pwr_par=pk_pwr_kw)

    SiSt = wavelet_ptd.fit(acc, a_time, still, a_filt, fs=128)

    assert len(SiSt) == 3
    assert allclose(array([SiSt[i].duration for i in SiSt]), array([1.413972, 1.64052, 1.67958]), atol=1e-3)


def _test_displacement():
    if version_info < (3, 7):
        file_path = pkr.resource_filename('pysit2stand.data', 'sample.csv')
        data = loadtxt(file_path, delimiter=',')
    else:
        with resources.path('pysit2stand.data', 'sample.csv') as file_path:
            data = loadtxt(file_path, delimiter=',')
    time = data[:, 0]
    acc = data[:, 1:]
    a_time = to_datetime(time, unit='us', utc=True).tz_convert('EST').tz_localize(None)

    a_filt = s2s.AccFilter(lowpass_order=4, lowpass_cutoff=5, discrete_wavelet='dmey',
                           reconstruction_level=1, reconstruction_method='moving average')

    ths = {'stand displacement': 0.125, 'transition velocity': 0.2, 'accel moving avg': 0.2,
           'accel moving std': 0.1, 'jerk moving avg': 2.5, 'jerk moving std': 3}
    disp = s2s.detectors.Displacement(gravity=9.81, thresholds=ths, long_still=0.5, moving_window=0.3,
                                      duration_factor=10, displacement_factor=0.75,
                                      lmax_kwargs=None, lmin_kwargs={'height': -9.5})
    cwave = 'gaus1'
    pk_pwr = [0, 0.5]
    pk_pwr_kw = {'height': 90, 'distance': 128}
    wavelet_ptd = s2s.Sit2Stand(continuous_wavelet=cwave, peak_pwr_band=pk_pwr, peak_pwr_par=pk_pwr_kw)

    SiSt = wavelet_ptd.fit(acc, a_time, disp, a_filt, fs=128)

    assert len(SiSt) == 5
    assert allclose(array([SiSt[i].duration for i in SiSt]),
                    array([1.413972, 1.64052, 1.67958, 3.296664, 1.812384]), atol=1e-3)
