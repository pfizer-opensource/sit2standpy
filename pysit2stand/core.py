"""
Wavelet based methods of detecting postural transitions

Lukas Adamowicz
June 2019
"""
from numpy import mean, diff, arange, logical_and, sum as npsum, std, timedelta64
from scipy.signal import find_peaks
import pywt


class Sit2Stand:
    """
    Wavelet based detection of sit-to-stand transitions

    Parameters
    ----------
    continuous_wavelet : str, optional
        Continuous wavelet to use for signal deconstruction. Default is 'gaus1'. CWT coefficients will be summed
        in frequency bands that will be used for detecting approximate STS locations.
    peak_pwr_band : {array_like, int, float}, optional
        Frequency band in which to sum the CWT coefficients. Either an array_like of length 2, with the lower and
        upper limits, or a number, which will be taken as the upper limit, and the lower limit will be set to 0.
        Default is [0, 0.5].
    peak_pwr_par : {None, dict}, optional
        Extra parameters (key-word arguments) to pass to scipy.signal.find_peaks when finding peaks in the
        summed CWT coefficient power band data. Default is None, which will use the default parameters, unless
        std_height is True.
    std_height : bool, optional
        Use the standard deviation of the power for peak finding. Default is True.
    """

    def __init__(self, continuous_wavelet='gaus1', peak_pwr_band=[0, 0.5], peak_pwr_par=None, std_height=True):
        self.cwave = continuous_wavelet  # TODO add checks this is a valid wavelet

        if isinstance(peak_pwr_band, (int, float)):
            self.pk_pwr_start = 0
            self.pk_pwr_stop = peak_pwr_band
        else:
            self.pk_pwr_start = peak_pwr_band[0]
            self.pk_pwr_stop = peak_pwr_band[1]

        if peak_pwr_par is None:
            self.pk_pwr_par = {'height': 90}
        else:
            self.pk_pwr_par = peak_pwr_par

        self.std_height = std_height
        if self.std_height:
            if 'height' in self.pk_pwr_par:
                del self.pk_pwr_par['height']

    def fit(self, accel, time, detector, acc_filter, fs=None):
        """
        Fit the data and determine sit-to-stand transitions start and stop times.

        First, the data is filtered using the acc_filter.apply(). After which, the Continuous Wavelet Transform is
        taken of the reconstructed data. The CWT coefficients are summed in the specified power band, and peaks are
        found. Filtered data, power peaks, and CWT data are passed to the detector.apply() method, which detects the
        sit-to-stand transitions.

        Parameters
        ----------
        accel : numpy.ndarray
            (N, 3) array of raw accelerations measured by a lumbar sensor.
        time : pandas.DatetimeIndex
            (N, ) array of pandas.DatetimeIndex corresponding with the acceleration data.
        detector : pysit2stand.detectors
            Initialized detector objects for detecting the sit-to-stand transisions. Must have an apply method. If
            creating a new object for this detection, see pysit2stand.detector.Displacement for the required arguments.
        acc_filter : pysit2stand.AccFilter
            Acceleration filter object, used to filter and reconstruct the magnitude of the acceleration. Must have
            an apply() method (eg acc_filter.apply()) that takes the raw acceleration, and sampling frequency only
            as arguments.
        fs : {None, float}, optional
            Sampling frequency. If none, calculated from the time. Default is None.

        Returns
        -------
        sts : dict
            Dictionary of pysit2stand.Transition objects containing information about a individual sit-to-stand
            transition. Keys for the dictionary are string timestamps of the start of the transition.
        """
        # calculate the sampling time and frequency
        if fs is None:
            dt = mean(diff(time)) / timedelta64(1, 's')  # convert to seconds
            fs = 1 / dt
        else:
            dt = 1 / fs

        # filter the raw acceleration using the AccFilter object
        self.macc_f, self.macc_r = acc_filter.apply(accel, fs)

        # compute the continuous wavelet transform on the reconstructed acceleration data
        self.coefs, self.freqs = pywt.cwt(self.macc_r, arange(1, 65), self.cwave, sampling_period=dt)

        # sum the CWT coefficients over the set of frequencies specified in the peak power band
        f_mask = logical_and(self.freqs <= self.pk_pwr_stop, self.freqs >= self.pk_pwr_start)
        self.power = npsum(self.coefs[f_mask, :], axis=0)

        # find the peaks in the power data
        if self.std_height:
            self.pwr_pks, _ = find_peaks(self.power, height=std(self.power, ddof=1), **self.pk_pwr_par)
        else:
            self.pwr_pks, _ = find_peaks(self.power, **self.pk_pwr_par)

        # use the detector object to fully detect the sit-to-stand transitions
        sts = detector.apply(accel, self.macc_f, self.macc_r, time, dt, self.pwr_pks, self.coefs, self.freqs)

        return sts


