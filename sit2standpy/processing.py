"""
Methods for preprocessing of inertial data.

Lukas Adamowicz
Pfizer
2019
"""
from numpy import around, mean, diff, timedelta64, arange, logical_and, sum, std, argwhere, append, insert, \
    ascontiguousarray
from numpy.linalg import norm
from scipy.signal import butter, filtfilt, find_peaks
import pywt
from pandas import to_datetime
from udatetime import utcfromtimestamp
from warnings import warn

from sit2standpy.utility import mov_stats


__all__ = ['AccelerationFilter', 'process_timestamps']


class AccelerationFilter:
    def __init__(self, continuous_wavelet='gaus1', power_band=None, power_peak_kw=None, power_std_height=True,
                 power_std_trim=0, reconstruction_method='moving average', lowpass_order=4, lowpass_cutoff=5, 
                 window=0.25, discrete_wavelet='dmey', extension_mode='constant', reconstruction_level=1):
        """
        Object for filtering and reconstructing raw acceleration data

        Parameters
        ----------
        continuous_wavelet : str, optional
            Continuous wavelet to use for signal deconstruction. Default is 'gaus1'. CWT coefficients will be summed
            in frequency bands that will be used for detecting approximate STS locations.
        power_band : {array_like, int, float}, optional
            Frequency band in which to sum the CWT coefficients. Either an array_like of length 2, with the lower and
            upper limits, or a number, which will be taken as the upper limit, and the lower limit will be set to 0.
            Default is [0, 0.5].
        power_peak_kw : {None, dict}, optional
            Extra key-word arguments to pass to scipy.signal.find_peaks when finding peaks in the
            summed CWT coefficient power band data. Default is None, which will use the default parameters except
            setting minimum height to 90, unless std_height is True.
        power_std_height : bool, optional
            Use the standard deviation of the power for peak finding. Default is True. If True, the standard deviation
            height will overwrite the setting in `power_peak_kw`.
        power_std_trim : int, optional
            Number of seconds to trim off the start and end of the power signal before computing the standard deviation
            for `power_std_height`. Default is 0s, which will not trim anything. Recommended value if trimming is 1s.
        reconstruction_method : {'moving average', 'dwt'}, optional
            Method for computing the reconstructed acceleration. Default is 'moving average', which takes the moving
            average over the specified window. Other option is 'dwt', which uses the discrete wavelet transform to
            deconstruct and reconstruct the signal while filtering noise out.
        lowpass_order : int, optional
            Initial low-pass filtering order. Default is 4.
        lowpass_cutoff : float, optional
            Initial low-pass filtering cuttoff, in Hz. Default is 5Hz.
        window : float, optional
            Window to use for moving average, in seconds. Default is 0.25s. Ignored if reconstruction_method is 'dwt'.
        discrete_wavelet : str, optional
            Discrete wavelet to use if reconstruction_method is 'dwt'. Default is 'dmey'. See
            pywt.wavelist(kind='discrete') for a complete list of options. Ignored if reconstruction_method is
            'moving average'.
        extension_mode : str, optional
            Signal extension mode to use in the DWT de- and re-construction of the signal. Default is 'constant', see
            pywt.Modes.modes for a list of options. Ignored if reconstruction_method is 'moving average'.
        reconstruction_level : int, optional
            Reconstruction level of the DWT processed signal. Default is 1. Ignored if reconstruction_method is
            'moving average'.
        """
        if power_band is None:
            power_band = [0, 0.5]
        self.cwave = continuous_wavelet
        if isinstance(power_band, (int, float)):
            self.power_start_f = 0
            self.power_end_f = power_band
        else:
            self.power_start_f = power_band[0]
            self.power_end_f = power_band[1]

        self.std_height = power_std_height
        self.std_trim = power_std_trim
        if power_peak_kw is None:  # if not set, set the default values
            # default height is 90. this will be reset later if necessary if using the stdev height
            self.power_peak_kw = {'height': 90}
        else:
            self.power_peak_kw = power_peak_kw  # if passed as a dictionary

        self.method = reconstruction_method

        self.lp_ord = lowpass_order
        self.lp_cut = lowpass_cutoff

        self.window = window

        self.dwave = discrete_wavelet
        self.ext_mode = extension_mode
        self.recon_level = reconstruction_level

    def apply(self, accel, fs):
        """
        Apply the desired filtering to the provided signal.

        Parameters
        ----------
        accel : numpy.ndarray
            (N, 3) array of raw acceleration values.
        fs : float, optional
            Sampling frequency for the acceleration data.

        Returns
        -------
        mag_acc_f : numpy.ndarray
            (N, ) array of the filtered (low-pass only) acceleration magnitude.
        mag_acc_r : numpy.ndarray
            (N, ) array of the reconstructed acceleration magnitude. This is either filtered and then moving averaged,
            or filtered, and then passed through the DWT and inverse DWT with more filtering, depending on the
            reconstruction_method specified.
        power : numpy.ndarray
            (N, ) array of the CWT power approximation in the band specified by `power_band`.
        power_peaks : numpy.ndarray
            Indices of the peaks detected in the power signal.
        """
        # compute the acceleration magnitude
        macc = norm(accel, axis=1)

        # setup the filter, and filter the acceleration magnitude
        fc = butter(self.lp_ord, 2 * self.lp_cut / fs, btype='low')
        macc_f = ascontiguousarray(filtfilt(fc[0], fc[1], macc))

        if self.method == 'dwt':
            # deconstruct the filtered acceleration magnitude
            coefs = pywt.wavedec(macc_f, self.dwave, mode=self.ext_mode)

            # set all but the desired level of coefficients to be 0s
            if (len(coefs) - self.recon_level) < 1:
                warn(UserWarning(f'Chosen reconstruction level is too high, '
                                 f'setting reconstruction level to {len(coefs) - 1}'))
                ind = 1
            else:
                ind = len(coefs) - self.recon_level

            for i in range(1, len(coefs)):
                if i != ind:
                    coefs[i][:] = 0

            macc_r = pywt.waverec(coefs, self.dwave, mode=self.ext_mode)
        elif self.method == 'moving average':
            n_window = int(around(fs * self.window))  # compute the length in samples of the moving average
            macc_r, _, _ = mov_stats(macc_f, n_window)  # compute the moving average

        # ---------------------------------------------------
        # CWT power peak detection

        # compute the cwt
        coefs, freqs = pywt.cwt(macc_r, arange(1, 65), self.cwave, sampling_period=1/fs)

        # sum the coefficients over the frequencies in the power band
        f_mask = logical_and(freqs <= self.power_end_f, freqs >= self.power_start_f)
        power = sum(coefs[f_mask, :], axis=0)

        # find the peaks in the power data
        if self.std_height:
            if self.std_trim != 0:
                trim = int(fs * self.std_trim)
                self.power_peak_kw['height'] = std(power[trim:-trim], ddof=1)
            else:
                self.power_peak_kw['height'] = std(power, ddof=1)

        power_peaks, _ = find_peaks(power, **self.power_peak_kw)

        return macc_f, macc_r[:macc_f.size], power, power_peaks


def process_timestamps(times, accel, time_units=None, conv_kw=None, window=False, hours=('08:00', '20:00')):
    """
    Convert timestamps into pandas datetime64 objects, and window as appropriate.

    Parameters
    ----------
    times : array_like
        N-length array of timestamps to convert.
    accel : {numpy.ndarray, pd.Series}
        (N, 3) array of acceleration values. They will be windowed the same way as the timestamps if `window` is set
        to True.
    time_units : {None, str}, optional
        Time units. Useful if conversion is from unix timestamps in seconds (s), milliseconds (ms), microseconds (us),
        or nanoseconds (ns). If not None, will override the value in conv_kw, though one or the other must be provided.
        Default is None.
    conv_kw : {None, dict}, optional
        Additional key-word arguments for the pandas.to_datetime function. If time_units is not None, that value
        will be used and overwrite the value in conv_kw. If the timestamps are in unix time, it is unlikely this
        argument will be necessary. Default is None.
    window : bool, optional
        Window the timestamps into the selected hours per day.
    hours : array_like, optional
        Length two array_like of hours (24-hour format) as strings, defining the start (inclusive) and end (exclusive)
        times to include in the processing. Default is ('08:00', '20:00').

    Returns
    -------
    timestamps : {pandas.DatetimeIndex, pandas.Series, dict}
        Array_like of timestamps. DatetimeIndex if times was a numpy.ndarray, or list. pandas.Series with a dtype of
        'datetime64' if times was a pandas.Series. If `window` is set to True, then a dictionary of timestamps for
        each day is returned.
    dt : float
        Sampling time in seconds.
    accel : {numpy.ndarray, pd.Series, dict}, optional
        Acceleration windowed the same way as the timestamps (dictionary of acceleration for each day), if `window` is
        True. If `window` is False, then the acceleration is not returned.
    """
    if conv_kw is not None:
        if time_units is not None:
            conv_kw['unit'] = time_units
    else:
        if time_units is not None:
            conv_kw = {'unit': time_units}
        else:
            raise ValueError('Either (time_units) must be defined, or "unit" must be a key of (conv_kw).')

    # convert using pandas
    if 'unit' in conv_kw:
        if conv_kw['unit'] == 'ms':
            timestamps = to_datetime([utcfromtimestamp(t).replace(tzinfo=None) for t in times / 1e3])
        elif conv_kw['unit'] == 'us':
            timestamps = to_datetime([utcfromtimestamp(t).replace(tzinfo=None) for t in times / 1e6])
        elif conv_kw['unit'] == 'ns':
            timestamps = to_datetime([utcfromtimestamp(t).replace(tzinfo=None) for t in times / 1e9])
        elif conv_kw['unit'] == 's':
            timestamps = to_datetime([utcfromtimestamp(t).replace(tzinfo=None) for t in times])
    else:
        timestamps = to_datetime(times, **conv_kw)

    # find the sampling time
    dt = mean(diff(timestamps[:100])) / timedelta64(1, 's')  # convert to seconds

    # windowing
    if window:
        hour_inds = timestamps.indexer_between_time(hours[0], hours[1])

        day_splits = argwhere(diff(hour_inds) > 1) + 1
        day_splits = append(insert(day_splits, 0, 0), hour_inds.size)

        timestamps_ = {}
        accel_ = {}

        for i in range(len(day_splits) - 1):
            timestamps_[f'Day {i + 1}'] = timestamps[hour_inds][day_splits[i]: day_splits[i + 1]]
            accel_[f'Day {i + 1}'] = accel[hour_inds][day_splits[i]: day_splits[i + 1]]

        return timestamps_, dt, accel_
    else:
        return timestamps, dt



