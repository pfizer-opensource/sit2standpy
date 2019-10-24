import pytest
from numpy import isclose, allclose, array
from pysit2stand.processing import AccelerationFilter, process_timestamps


def test_accelerationfilter_none_inputs():
    af = AccelerationFilter()

    assert isclose(af.power_start_f, 0.0)
    assert isclose(af.power_end_f, 0.5)
    assert af.power_peak_kw == {'height': 90}


def test_accelerationfilter_single_power_band_input():
    af = AccelerationFilter(power_band=0.8)

    assert isclose(af.power_start_f, 0.0)
    assert isclose(af.power_end_f, 0.8)


def test_accelerationfilter(raw_accel, filt_accel, rm_accel, power, power_peaks):
    af = AccelerationFilter(continuous_wavelet='gaus1', power_band=[0, 0.5], power_peak_kw={'distance': 128},
                            power_std_height=True, reconstruction_method='moving average', lowpass_order=4,
                            lowpass_cutoff=5, window=0.25)

    f_acc, rm_acc, pwr, pwr_pk = af.apply(raw_accel, 128)

    assert allclose(filt_accel, f_acc)
    assert allclose(rm_accel, rm_acc)
    assert allclose(power, pwr)
    assert allclose(power_peaks, pwr_pk)


def test_process_timestamps_error(time, raw_accel):
    with pytest.raises(ValueError) as e_info:
        process_timestamps(time, raw_accel)


def test_process_timestamps_window(overnight_time_accel, windowed_timestamps):
    timestamps, dt, accel = process_timestamps(overnight_time_accel[0], overnight_time_accel[1], time_units='ns',
                                               conv_kw={}, window=True, hours=('08:00', '20:00'))

    assert timestamps.size == 10
    assert all(timestamps == windowed_timestamps)
    assert accel.size == 10
    assert isclose(dt, 3600.0)


def test_process_timestamps(timestamps_time_accel):
    timestamps, dt = process_timestamps(timestamps_time_accel[1], timestamps_time_accel[2], time_units='ns',
                                        window=False)

    assert all(timestamps == timestamps_time_accel[0])
    assert isclose(dt, 0.05)

