"""
Wavelet based methods of detecting postural transitions

Lukas Adamowicz
June 2019
"""
from numpy import mean, diff, arange, logical_and, sum as npsum, abs as npabs, gradient, where, around, isclose, \
    append, sign, array, median, std, timedelta64
from numpy.linalg import norm
from scipy.signal import find_peaks, butter, filtfilt, detrend
from scipy.integrate import cumtrapz
import pywt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pysit2stand import utility as u_
from pysit2stand.common import Transition

plt.style.use(['ggplot', 'presentation'])


class Wavelet:
    def __init__(self, continuous_wavelet='gaus1', peak_pwr_band=[0, 0.5], peak_pwr_par=None, std_height=True):
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
        self.cwave = continuous_wavelet  # TODO add checks this is a valid wavelet

        if isinstance(peak_pwr_band, (int, float)):
            self.pk_pwr_start = 0
            self.pk_pwr_stop = peak_pwr_band
        else:
            self.pk_pwr_start = peak_pwr_band[0]
            self.pk_pwr_stop = peak_pwr_band[1]

        if peak_pwr_par is None:
            self.pk_pwr_par = {}
        else:
            self.pk_pwr_par = peak_pwr_par

        self.std_height = std_height
        if self.std_height:
            if 'height' in self.pk_pwr_par:
                del self.pk_pwr_par['height']

    def fit(self, accel, time, detector, acc_filter):
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
        detector: {pysit2stand.acc.StillnessDetector, pysit2stand.acc.SimilarityDetector}
            Initialized detector objects for detecting the sit-to-stand transisions. Must have an apply method. If
            creating a new object for this detection, see StillnessDetector.apply() for the required arguments.
        acc_filter : pysit2stand.AccFilter
            Acceleration filter object, used to filter and reconstruct the magnitude of the acceleration. Must have
            an apply() method (eg acc_filter.apply()) that takes the raw acceleration, and sampling frequency only
            as arguments.

        Returns
        -------
        sts : list
            List of tuples of (STS start, STS end) for all of the detected STS transitions in the acceleration data.
        """
        # calculate the sampling time and frequency
        dt = mean(diff(time)) / timedelta64(1, 's')  # convert to seconds
        fs = 1 / dt

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
        sts, self.ext = detector.apply(accel, self.macc_f, self.macc_r, time, dt, self.pwr_pks, self.coefs, self.freqs)

        return sts


class DisplacementDetector:
    def __init__(self, strict_stillness=False, gravity=9.81, thresholds=None, gravity_pass_ord=4,
                 gravity_pass_cut=0.8, long_still=0.5, moving_window=0.3, duration_factor=3, lmax_kwargs=None,
                 lmin_kwargs=None):
        """
        Method for detecting sit-to-stand transitions based on requiring stillness before a transition, and the
        vertical position of a lumbar accelerometer.

        Parameters
        ----------
        strict_stillness : bool, optional
            Whether or not to require stillness for a sit-to-stand transition, or to use vertical position data instead.
            True requires that stillness precede a transition, and this is recommended for situations where transitions
            are not expected to occur rapidly, or with much motion beforehand. Setting this to False allows the
            vertical position to be used without requiring a still period before the transition. Default is True.
        gravity : float, optional
            Value of gravitational acceleration measured by the accelerometer when still. Default is 9.81 m/s^2.
        thresholds : {None, dict}, optional
            Either None, for the default, or a dictionary of thresholds to change. See
            PosiStillDetector.default_thresholds for a dictionary of the thresholds and their default values. Default
            is None, which uses the default values.
        gravity_pass_ord : int, optional
            Low-pass filter order for estimating the direction of gravity by low-pass filtering the raw acceleration
            data. Default is 4.
        gravity_pass_cut : float, optional
            Low-pass filter frequency cutoff for estimating thd direction of gravity. Default is 0.8Hz.
        long_still : float, optional
            Length of time of stillness for it to be qualified as a long period of stillness. Used to determing
            integration limits when available. Default is 0.5s.
        moving_window : float, optional
            Length of the moving window for calculating the moving statistics for determining stillness.
            Default is 0.3s.
        duration_factor : float, optional
            The factor for the maximum difference between the duration before and after the generalized location of
            the sit to stand. Lower factors result in more equal time before and after the detection. Default
            is 3.
        lmax_kwargs : {None, dict}, optional
            Additional key-word arguments for finding local maxima in the acceleration signal. Default is None,
            for no specified arguments. See scipy.signal.find_peaks for possible arguments.
        lmin_kwargs : {None, dict}, optional
            Additional key-word arguments for finding local minima in the acceleration signal. Default is None,
            for no specified arguments. See scipy.signal.find_peaks for the possible arguments.
        """
        # set the default thresholds
        self.default_thresholds = {'stand displacement': 0.15,
                                   'still velocity': 0.05,
                                   'accel moving avg': 0.25,
                                   'accel moving std': 0.5,
                                   'jerk moving avg': 3,
                                   'jerk moving std': 5}
        # assign attributes
        self.strict = strict_stillness
        self.grav = gravity

        self.thresh = {i: self.default_thresholds[i] for i in self.default_thresholds.keys()}
        if thresholds is not None:
            for key in thresholds.keys():
                if key in self.thresh:
                    self.thresh[key] = thresholds[key]

        self.grav_ord = gravity_pass_ord
        self.grav_cut = gravity_pass_cut

        self.long_still = long_still
        self.mov_window = moving_window

        self.dur_factor = duration_factor

        if lmin_kwargs is None:
            self.lmin_kw = {}
        else:
            self.lmin_kw = lmin_kwargs
        if lmax_kwargs is None:
            self.lmax_kw = {}
        else:
            self.lmax_kw = lmax_kwargs

    def apply(self, raw_acc, mag_acc, mag_acc_r, time, dt, power_peaks, cwt_coefs, cwt_freqs):
        # find stillness
        acc_still, still_starts, still_stops = DisplacementDetector._stillness(mag_acc, dt, self.mov_window, self.grav,
                                                                               self.thresh)
        # starts and stops of long still periods
        still_dt = (still_stops - still_starts) * dt  # durations of stillness, in seconds
        lstill_starts = still_starts[still_dt > self.long_still]
        lstill_stops = still_starts[still_dt > self.long_still]

        # find the local minima and maxima in the acceleration signals. Use the reconstructed acceleration for
        # local minima, as this avoids some possible artefacts in the signal
        acc_lmax, _ = find_peaks(mag_acc, **self.lmax_kw)
        acc_lmin, _ = find_peaks(-mag_acc_r, **self.lmin_kw)

        # compute an estimate of the direction of gravity, assumed to be the vertical direction
        gfc = butter(self.grav_ord, 2 * self.grav_cut * dt, btype='low')
        vert = filtfilt(gfc[0], gfc[1], raw_acc, axis=1, padlen=None)
        vert /= norm(vert, axis=1, keepdims=True)  # make into unit vectors

        # get an estimate of the vertical acceleration
        v_acc = npsum(vert * raw_acc, axis=1)

        # iterate over the power peaks
        sts = {}
        pos_lines = []

        prev_int_start = -1
        prev_int_stop = -1

        for ppk in power_peaks:
            # look for the preceding end of stillness
            try:
                end_still = still_stops[still_stops < ppk][-1] if self.strict else lstill_stops[lstill_stops < ppk][-1]

                # make sure physically possible transition duration, or that integration isn't too long
                if (time[ppk] - time[end_still]).total_seconds() > (2 if self.strict else 15):
                    raise IndexError
            except IndexError:
                continue
            # look for the next local minima then maxima
            try:
                n_lmin = acc_lmin[acc_lmin > ppk][0]
                n_lmax = acc_lmax[acc_lmax > n_lmin][0]
                if (time[n_lmax] - time[ppk]).total_seconds() > 2:
                    raise IndexError
            except IndexError:
                continue
            # look for the following start of a long stillness, or if that fails, a short stillness
            try:
                start_still = lstill_starts[lstill_starts > ppk][0]
                if (time[start_still] - time[ppk]).total_seconds() < 30:
                    still_at_end = True
                else:
                    raise IndexError
            except IndexError:
                start_still = n_lmax if self.strict else int(ppk + (5 / dt))
                still_at_end = self.strict  # strict designation matches what this should get set to

            # integrate the signal between the start and stop points
            if end_still < prev_int_start or start_still > prev_int_stop:
                v_vel, v_pos = DisplacementDetector._get_position(v_acc[end_still:start_still] - self.grav, dt,
                                                                  still_at_end)



    @staticmethod
    def _get_position(acc, dt, still_at_end):
        """
        Double integrate acceleration along 1 axis (ie 1D) to get velocity and position

        Parameters
        ----------
        acc : numpy.ndarray
            (N, ) array of acceleration values to integrate
        still_at_end : bool
            Whether or not the acceleration ends with a still period. Determines how drift is mitigated.

        Returns
        -------
        vel : numpy.ndarray
            (N, ) array of velocities
        pos : numpy.ndarray
            (N, ) array of positions
        """
        x = arange(acc.size)

        # integrate and drift mitigate
        if not still_at_end:
            # fc = butter(1, [2 * 0.1 * dt, 2 * 5 * dt], btype='band')
            # vel = cumtrapz(filtfilt(fc[0], fc[1], acc), dx=dt, initial=0)
            vel = detrend(cumtrapz(acc, dx=dt, initial=0))
            if npabs(vel[0]) > 0.05:  # if too far away from zero
                vel -= vel[0]  # reset the beginning back to 0, the integration always starts with stillness
        else:
            vel_dr = cumtrapz(acc, dx=dt, initial=0)
            vel = vel_dr - (((vel_dr[-1] - vel_dr[0]) / (x[-1] - x[0])) * x)  # no intercept

        # integrate the velocity to get position
        pos = cumtrapz(vel, dx=dt, initial=0)

        return vel, pos

    @staticmethod
    def _stillness(mag_acc_f, dt, window, gravity, thresholds):
        """
        Stillness determination of acceleration magnitude data

        Parameters
        ----------
        mag_acc_f : numpy.ndarray
            (N, 3) array of filtered acceleration data.
        dt : float
            Sampling time difference, in seconds.
        window : float
            Moving statistics window length, in seconds.
        gravity : float
            Gravitational acceleration, as measured by the sensor during static sitting or standing.
        thresholds : dict
            Dictionary of the 4 thresholds to be used - acceleration and jerk moving averages and standard deviations.

        Returns
        -------
        acc_still : numpy.ndarray
            (N, ) boolean array indicating stillness
        starts : numpy.ndarray
            (Q, ) array of where stillness ends, where by necessity has to follow: Q < N / 2
        stops : numpy.ndarray
            (P, ) array of where stillness ends, where by necessity has to follow: P < N / 2
        """
        # calculate the sample window from the time window
        n_window = int(around(window / dt))
        # compute the acceleration moving standard deviation
        am_avg, am_std, _ = u_.mov_stats(mag_acc_f, n_window)
        # compute the jerk
        jerk = gradient(mag_acc_f, dt, edge_order=2)
        # compute the jerk moving average and standard deviation
        jm_avg, jm_std, _ = u_.mov_stats(jerk, n_window)

        # create masks from the moving statistics of acceleration and jerk
        am_avg_mask = npabs(am_avg - gravity) < thresholds['accel moving avg']
        am_std_mask = am_std < thresholds['accel moving std']
        jm_avg_mask = npabs(jm_avg) < thresholds['jerk moving avg']
        jm_std_mask = jm_std < thresholds['jerk moving std']

        acc_still = am_avg_mask & am_std_mask & jm_avg_mask & jm_std_mask
        starts = where(diff(acc_still.astype(int)) == 1)[0]
        stops = where(diff(acc_still.astype(int)) == -1)[0]

        if acc_still[0]:
            still_starts = append(0, starts)
        if acc_still[-1]:
            still_stops = append(stops, len(acc_still) - 1)

        # TODO Could consider adding all the masks together and filtering, then taking values above a threshold

        return acc_still, starts, stops


class PosiStillDetector:
    def __str__(self):
        return f'Position and Stillness Sit-to-Stand Detector'

    def __repr__(self):
        return f'PosiStillDetector({self.grav}, {self.thresholds}, {self.grav_ord}, {self.grav_cut}, ' \
            f'{self.long_still}, {self.mov_window}'

    def __init__(self, strict_stillness=True, gravity=9.81, thresholds=None, gravity_pass_ord=4,
                 gravity_pass_cut=0.8, long_still=0.5, moving_window=0.3, duration_factor=3, lmax_kwargs=None,
                 lmin_kwargs=None):
        """
        Method for detecting sit-to-stand transitions based on requiring stillness before a transition, and the
        vertical position of a lumbar accelerometer.

        Parameters
        ----------
        strict_stillness : bool, optional
            Whether or not to require stillness for a sit-to-stand transition, or to use vertical position data instead.
            True requires that stillness precede a transition, and this is recommended for situations where transitions
            are not expected to occur rapidly, or with much motion beforehand. Setting this to False allows the
            vertical position to be used without requiring a still period before the transition. Default is True.
        gravity : float, optional
            Value of gravitational acceleration measured by the accelerometer when still. Default is 9.81 m/s^2.
        thresholds : {None, dict}, optional
            Either None, for the default, or a dictionary of thresholds to change. See
            PosiStillDetector.default_thresholds for a dictionary of the thresholds and their default values. Default
            is None, which uses the default values.
        gravity_pass_ord : int, optional
            Low-pass filter order for estimating the direction of gravity by low-pass filtering the raw acceleration
            data. Default is 4.
        gravity_pass_cut : float, optional
            Low-pass filter frequency cutoff for estimating thd direction of gravity. Default is 0.8Hz.
        long_still : float, optional
            Length of time of stillness for it to be qualified as a long period of stillness. Used to determing
            integration limits when available. Default is 0.5s.
        moving_window : float, optional
            Length of the moving window for calculating the moving statistics for determining stillness.
            Default is 0.3s.
        duration_factor : float, optional
            The factor for the maximum difference between the duration before and after the generalized location of
            the sit to stand. Lower factors result in more equal time before and after the detection. Default
            is 3.
        lmax_kwargs : {None, dict}, optional
            Additional key-word arguments for finding local maxima in the acceleration signal. Default is None,
            for no specified arguments. See scipy.signal.find_peaks for possible arguments.
        lmin_kwargs : {None, dict}, optional
            Additional key-word arguments for finding local minima in the acceleration signal. Default is None,
            for no specified arguments. See scipy.signal.find_peaks for the possible arguments.
        """
        # set the default thresholds
        self.default_thresholds = {'stand displacement': 0.15,
                                   'still velocity': 0.05,
                                   'accel moving avg': 0.25,
                                   'accel moving std': 0.5,
                                   'jerk moving avg': 3,
                                   'jerk moving std': 5}
        # assign attributes
        self.strict = strict_stillness
        self.grav = gravity

        self.thresh = {i: self.default_thresholds[i] for i in self.default_thresholds.keys()}
        if thresholds is not None:
            for key in thresholds.keys():
                if key in self.thresh:
                    self.thresh[key] = thresholds[key]

        self.grav_ord = gravity_pass_ord
        self.grav_cut = gravity_pass_cut

        self.long_still = long_still
        self.mov_window = moving_window

        self.dur_factor = duration_factor

        if lmin_kwargs is None:
            self.lmin_kw = {}
        else:
            self.lmin_kw = lmin_kwargs
        if lmax_kwargs is None:
            self.lmax_kw = {}
        else:
            self.lmax_kw = lmax_kwargs

    def apply(self, raw_acc, mag_acc, mag_acc_r, time, dt, power_peaks, cwt_coefs, cwt_freqs):
        # find where the accelerometer is still
        acc_still, still_stops = PosiStillDetector._stillness(mag_acc, dt, self.mov_window, self.grav,
                                                              self.thresh['accel moving avg'],
                                                              self.thresh['accel moving std'],
                                                              self.thresh['jerk moving avg'],
                                                              self.thresh['jerk moving std'])
        # find starts of stillness, and add beginning and end of trials if necessary
        still_starts = where(diff(acc_still.astype(int)) == 1)[0]
        if acc_still[0]:
            still_starts = append(0, still_starts)
        if acc_still[-1]:
            still_stops = append(still_stops, len(acc_still) - 1)

        still_dt = still_stops - still_starts  # durations of stillness, in samples
        # starts and stops of long still periods
        lstill_starts = still_starts[still_dt > (self.long_still / dt)]
        lstill_stops = still_stops[still_dt > (self.long_still / dt)]

        # find the local minima and maxima in the acceleration signals. Use the reconstructed acceleration for
        # local minima, as this avoids some possible artefacts in the signal
        acc_lmax, _ = find_peaks(mag_acc, **self.lmax_kw)
        acc_lmin, _ = find_peaks(-mag_acc_r, **self.lmin_kw)

        # compute an estimate of the direction of gravity, assumed to be vertical direction
        gfc = butter(self.grav_ord, 2 * self.grav_cut * dt, btype='low')
        vertical = filtfilt(gfc[0], gfc[1], raw_acc, axis=0, padlen=None)
        vertical /= norm(vertical, axis=1, keepdims=True)  # make into a unit vector

        # get an estimate of the vertical acceleration
        v_acc = npsum(vertical * raw_acc, axis=1)

        # iterate over the peaks
        sts = {}
        pos_lines = []

        prev_int_start = -1
        prev_int_stop = -1
        if self.strict:
            for ppk in power_peaks:
                # look for the preceding end of any stillness
                try:
                    end_still = still_stops[still_stops < ppk][-1]
                    # TODO make this a parameter?
                    if (time[ppk] - time[end_still]) > 2:  # check to make sure its not too long of a time
                        raise IndexError
                except IndexError:
                    continue
                # look for the following local min -> local max pattern
                try:
                    n_lmin = acc_lmin[acc_lmin > ppk][0]
                    n_lmax = acc_lmax[acc_lmax > n_lmin][0]
                    # TODO make this a parameter
                    if (time[n_lmax] - time[ppk]) > 2:  # check to make sure not too long
                        raise IndexError
                except IndexError:
                    continue
                # look for a still period for integration
                try:
                    start_still = still_starts[still_starts > ppk][0]
                    if start_still < n_lmax:
                        raise IndexError
                    elif (time[start_still] - time[ppk]) < 30:  # can integrate for a little while if necessary
                        still_at_end = True
                    else:
                        raise IndexError
                except IndexError:
                    start_still = n_lmax
                    still_at_end = True

                # integrate the signal between the start and stop points
                if end_still < prev_int_start or start_still > prev_int_stop:
                    v_vel, v_pos = PosiStillDetector._get_position(v_acc[end_still:start_still] - self.grav, dt,
                                                                   still_at_end)
                    pos_lines.append(Line2D(time[end_still:start_still], v_pos, color='C5', linewidth=1.5))

                    # find the zero-crossings
                    pos_zc = where(diff(sign(v_vel)) > 0)[0]
                    neg_zc = where(diff(sign(v_vel)) < 0)[0]

                    if neg_zc.size == 0:
                        if v_vel[-1] < 1e-2:
                            neg_zc = array([v_pos.size - 1])
                # ensure that the vertical velocity indicates that it is a peak as well
                if v_vel[ppk - end_still] < 0.2:  # TODO make parameter
                    continue
                # previous and next zc
                try:
                    p_pzc = pos_zc[pos_zc + end_still < ppk][-1]
                except IndexError:
                    continue
                try:
                    n_nzc = neg_zc[neg_zc + end_still > ppk][0]
                except IndexError:
                    continue

                if (time[ppk] - time[end_still]) > self.dur_factor * (time[n_lmax] - time[ppk]):
                    continue
                if npabs(time[p_pzc + end_still] - time[end_still]) > 0.35:  # TODO make parameter
                    continue
                if (v_pos[n_nzc] - v_pos[p_pzc]) > self.thresh['stand displacement']:
                    if len(sts) > 0:
                        if (time[end_still] - sts[list(sts.keys())[-1]].end_time) > 0.5:  # prevent overlap
                            # sts.append((time[end_still], time[n_lmax]))
                            sts[f'{time[end_still]}'] = Transition(times=(time[end_still], time[n_lmax]),
                                                                   v_displacement=v_pos[n_nzc] - v_pos[p_pzc])
                    else:
                        sts[f'{time[end_still]}'] = Transition(times=(time[end_still], time[n_lmax]),
                                                               v_displacement=v_pos[n_nzc] - v_pos[p_pzc])

                # save so don't have to integrate again when not necessary
                prev_int_start = end_still
                prev_int_stop = start_still

        else:
            for ppk in power_peaks:
                # look for the preceding end of long stillness
                try:
                    end_still = lstill_stops[lstill_stops < ppk][-1]
                    if (time[ppk] - time[end_still]) > 30:  # don't want to integrate for too long
                        raise IndexError
                except IndexError:
                    end_still = int(ppk - (2.5 / dt))  # will try to use set time beforehand

                # look for the next start of long stillness, and if not look for any short stillness ends
                try:
                    start_still = lstill_starts[lstill_starts > ppk][0]
                    if (time[start_still] - time[ppk]) < 30:
                        still_at_end = True
                    else:
                        raise IndexError
                except IndexError:
                    start_still = int(ppk + (5 / dt))  # will try to use a set time afterwards
                    still_at_end = False

                # integrate
                if end_still < prev_int_start or start_still > prev_int_stop:
                    v_vel, v_pos = PosiStillDetector._get_position(v_acc[end_still:start_still] - self.grav, dt,
                                                                   still_at_end)

                    pos_lines.append(Line2D(time[end_still:start_still], v_pos, color='C5', linewidth=1.5))

                    # find the zero-crossings
                    pos_zc = where(diff(sign(v_vel)) > 0)[0] + end_still
                    pos_zc = append(end_still, pos_zc)
                    # neg_zc = where(diff(sign(v_vel)) < 0)[0] + end_still
                # make sure that the velocity is high enough to indicate a peak
                if v_vel[ppk - end_still] < 0.2:  # TODO make parameter
                    continue
                # find the previous positive zero crossing
                try:
                    p_pzc = pos_zc[pos_zc < ppk][-1]
                    p_still = still_stops[still_stops < ppk][-1]
                    if (-0.5 / dt) < (p_still - p_pzc) < (0.7 / dt):
                        p_pzc = p_still
                    if (time[ppk] - time[p_pzc]) > 2:  # TODO make this a parameter?
                        raise IndexError
                except IndexError:
                    continue
                # find the next negative zero crossing
                try:
                    n_lmin = acc_lmin[acc_lmin > ppk][0]
                    n_lmax = acc_lmax[acc_lmax > n_lmin][0]
                    if (time[n_lmax] - time[ppk]) > 2:  # TODO make this a parameter?
                        raise IndexError
                except IndexError:
                    continue

                if (time[ppk] - time[p_pzc]) > self.dur_factor * (time[n_lmax] - time[ppk]):
                    continue
                test_ind = n_lmax - end_still if (n_lmax - end_still) < v_pos.size else -1
                if (v_pos[test_ind] - v_pos[p_pzc - end_still]) > self.thresh['stand displacement']:
                    if len(sts) > 0:
                        if (time[p_pzc] - sts[list(sts.keys())[-1]].end_time) > 0.4:  # prevent overlap TODO make cooldown a parameter
                            # sts.append((time[p_pzc], time[n_lmax]))
                            sts[f'{time[p_pzc]}'] = Transition(times=(time[p_pzc], time[n_lmax]),
                                                               v_displacement=v_pos[test_ind] - v_pos[p_pzc - end_still])
                    else:
                        # sts.append((time[p_pzc], time[n_lmax]))
                        sts[f'{time[p_pzc]}'] = Transition(times=(time[p_pzc], time[n_lmax]),
                                                           v_displacement=v_pos[test_ind] - v_pos[p_pzc - end_still])

                # save so don't have to integrate again when not necessary
                prev_int_start = end_still
                prev_int_stop = start_still

        # check to ensure no partial transitions
        vd = [sts[i].v_displacement for i in sts]
        vd_high_diff = array(vd) < 0.5 * median(vd)  # TODO should probably make a parameter
        for elem in array(list(sts.keys()))[vd_high_diff]:
            del sts[elem]

        # some stuff for plotting
        l1 = Line2D(time[acc_still], mag_acc[acc_still], color='k', marker='.', ls='')

        return sts, {'pos lines': pos_lines, 'lines': [l1]}

    @staticmethod
    def _get_position(acc, dt, still_at_end):
        """
        Double integrate acceleration along 1 axis (ie 1D) to get velocity and position

        Parameters
        ----------
        acc : numpy.ndarray
            (N, ) array of acceleration values to integrate
        still_at_end : bool
            Whether or not the acceleration ends with a still period. Determines how drift is mitigated.

        Returns
        -------
        vel : numpy.ndarray
            (N, ) array of velocities
        pos : numpy.ndarray
            (N, ) array of positions
        """
        x = arange(acc.size)

        # integrate and drift mitigate
        if not still_at_end:
            # fc = butter(1, [2 * 0.1 * dt, 2 * 5 * dt], btype='band')
            # vel = cumtrapz(filtfilt(fc[0], fc[1], acc), dx=dt, initial=0)
            vel = detrend(cumtrapz(acc, dx=dt, initial=0))
            if npabs(vel[0]) > 0.05:  # if too far away from zero
                vel -= vel[0]  # reset the beginning back to 0, the integration always starts with stillness
        else:
            vel_dr = cumtrapz(acc, dx=dt, initial=0)
            vel = vel_dr - (((vel_dr[-1] - vel_dr[0]) / (x[-1] - x[0])) * x)  # no intercept

        # integrate the velocity to get position
        pos = cumtrapz(vel, dx=dt, initial=0)

        return vel, pos

    @staticmethod
    def _stillness(mag_acc_f, dt, window, gravity, acc_mov_avg_thresh, acc_mov_std_thresh, jerk_mov_avg_thresh,
                   jerk_mov_std_thresh):
        """
        Stillness determination of acceleration magnitude data

        Parameters
        ----------
        mag_acc_f : numpy.ndarray
            (N, 3) array of filtered acceleration data.
        dt : float
            Sampling time difference, in seconds.
        window : float
            Moving statistics window length, in seconds.
        gravity : float
            Gravitational acceleration, as measured by the sensor during static sitting or standing.
        acc_mov_avg_thresh : float
            Acceleration moving average threshold, used for determining stillness.
        acc_mov_std_thresh : float
            Acceleration moving standard deviation threshold, used for determining stillness.
        jerk_mov_avg_thresh : float
            Jerk moving average threshold, used for determining stillness.
        jerk_mov_std_thresh : float
            Jerk moving standard deviation threshold, used for determining stillness.

        Returns
        -------
        acc_still : numpy.ndarray
            (N, ) boolean array indicating stillness
        stops : numpy.ndarray
            (P, ) array of where stillness ends, where by necessity has to follow: P < N / 2
        """
        # calculate the sample window from the time window
        n_window = int(around(window / dt))
        # compute the acceleration moving standard deviation
        am_avg, am_std, _ = u_.mov_stats(mag_acc_f, n_window)
        # compute the jerk
        jerk = gradient(mag_acc_f, dt, edge_order=2)
        # compute the jerk moving average and standard deviation
        jm_avg, jm_std, _ = u_.mov_stats(jerk, n_window)

        # create masks from the moving statistics of acceleration and jerk
        am_avg_mask = npabs(am_avg - gravity) < acc_mov_avg_thresh
        am_std_mask = am_std < acc_mov_std_thresh
        jm_avg_mask = npabs(jm_avg) < jerk_mov_avg_thresh
        jm_std_mask = jm_std < jerk_mov_std_thresh

        acc_still = am_avg_mask & am_std_mask & jm_avg_mask & jm_std_mask
        stops = where(diff(acc_still.astype(int)) == -1)[0]

        # TODO Could consider adding all the masks together and filtering, then taking values above a threshold

        return acc_still, stops