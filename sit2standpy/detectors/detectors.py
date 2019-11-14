"""
Methods for sit to stand transition detection, using previously processed data

Lukas Adamowicz
June 2019
Pfizer
"""
from numpy import around, gradient, abs as npabs, where, diff, sum as npsum, isclose, append, arange, array, sign, \
    median
from numpy.linalg import norm
from scipy.integrate import cumtrapz
from scipy.signal import find_peaks, butter, filtfilt, detrend

from sit2standpy.utility import Transition, mov_stats
from sit2standpy.quantify import TransitionQuantifier as TQ


__all__ = ['Stillness', 'Displacement']


# some common methods
def _get_still(mag_acc_f, dt, window, gravity, thresholds):
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
    am_avg, am_std, _ = mov_stats(mag_acc_f, n_window)
    # compute the jerk
    jerk = gradient(mag_acc_f, dt, edge_order=2)
    # compute the jerk moving average and standard deviation
    jm_avg, jm_std, _ = mov_stats(jerk, n_window)

    # create masks from the moving statistics of acceleration and jerk
    am_avg_mask = npabs(am_avg - gravity) < thresholds['accel moving avg']
    am_std_mask = am_std < thresholds['accel moving std']
    jm_avg_mask = npabs(jm_avg) < thresholds['jerk moving avg']
    jm_std_mask = jm_std < thresholds['jerk moving std']

    acc_still = am_avg_mask & am_std_mask & jm_avg_mask & jm_std_mask
    starts = where(diff(acc_still.astype(int)) == 1)[0]
    stops = where(diff(acc_still.astype(int)) == -1)[0]

    if acc_still[0]:
        starts = append(0, starts)
    if acc_still[-1]:
        stops = append(stops, len(acc_still) - 1)

    return acc_still, starts, stops


def _integrate_acc(acc, dt, still_at_end):
    """
    Double integrate acceleration along 1 axis (ie 1D) to get velocity and position

    Parameters
    ----------
    acc : numpy.ndarray
        (N, ) array of acceleration values to integrate
    dt : float
        Time difference between samples of acceleration in seconds.
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


class Stillness:
    """
    Method for detecting sit-to-stand transitions based on requiring stillness before a transition, and the
    vertical displacement of a lumbar accelerometer for ensuring a transition.

    Parameters
    ----------
    gravity : float, optional
        Value of gravitational acceleration measured by the accelerometer when still. Default is 9.81 m/s^2.
    thresholds : {None, dict}, optional
        Either None, for the default, or a dictionary of thresholds to change. See *Notes*. Default
        is None, which uses the default values.
    gravity_pass_ord : int, optional
        Low-pass filter order for estimating the direction of gravity by low-pass filtering the raw acceleration
        data. Default is 4.
    gravity_pass_cut : float, optional
        Low-pass filter frequency cutoff for estimating thd direction of gravity. Default is 0.8Hz.
    long_still : float, optional
        Length of time of stillness for it to be qualified as a long period of stillness. Used to determining
        integration limits when available. Default is 0.5s.
    moving_window : float, optional
        Length of the moving window for calculating the moving statistics for determining stillness.
        Default is 0.3s.
    duration_factor : float, optional
        The factor for the maximum difference between the duration before and after the generalized location of
        the sit to stand. Lower factors result in more equal time before and after the detection. Default
        is 10, which effectively removes this constraint.
    displacement_factor : float, optional
        Factor multiplied by the median of the vertical displacements to determine the threshold for checking if a
        transition is a partial transition or a full transition. Default is 0.75
    lmax_kwargs : {None, dict}, optional
        Additional key-word arguments for finding local maxima in the acceleration signal. Default is None,
        for no specified arguments. See `scipy.signal.find_peaks` for possible arguments.
    lmin_kwargs : {None, dict}, optional
        Additional key-word arguments for finding local minima in the acceleration signal. Default is None,
        which specifies a maximum value of 9.5m/s^2 for local minima. See `scipy.signal.find_peaks` for the
        possible arguments.
    trans_quant : TransitionQuantifier
        TransitionQuantifier object, which contains a `quantify` method, which accepts the following arguments:
        `times`, `mag_acc_f`, `mag_acc_r`, `v_vel`, `v_pos`. Only times is required. See
        `sit2standpy.TransitionQuantifier`.

    Notes
    -----
    Default thresholds:
        - stand displacement: 0.125
        - transition velocity: 0.2
        - accel moving avg: 0.2
        - accel moving std: 0.1
        - jerk moving avg: 2.5
        - jerk moving std: 3
    """
    def __init__(self, gravity=9.81, thresholds=None, gravity_pass_ord=4, gravity_pass_cut=0.8, long_still=0.5,
                 moving_window=0.3, duration_factor=10, displacement_factor=0.75, lmax_kwargs=None,
                 lmin_kwargs=None, trans_quant=TQ()):

        # set the default thresholds
        self.default_thresholds = {'stand displacement': 0.125,
                                   'transition velocity': 0.2,
                                   'accel moving avg': 0.2,
                                   'accel moving std': 0.1,
                                   'jerk moving avg': 2.5,
                                   'jerk moving std': 3}
        # assign attributes
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
        self.disp_factor = displacement_factor

        if lmin_kwargs is None:
            self.lmin_kw = {'height': -9.5}
        else:
            self.lmin_kw = lmin_kwargs
        if lmax_kwargs is None:
            self.lmax_kw = {}
        else:
            self.lmax_kw = lmax_kwargs

        self.tq = trans_quant

    def apply(self, raw_acc, mag_acc, mag_acc_r, time, dt, power_peaks):
        """
        Apply the stillness-based STS detection to the given data

        Parameters
        ----------
        raw_acc : numpy.ndarray
            (N, 3) array of the raw acceleration signal
        mag_acc : numpy.ndarray
            (N, ) array of filtered acceleration magnitude.
        mag_acc_r : numpy.ndarray
            (N, ) array of reconstructed acceleration magnitude.
        time : numpy.ndarray
            (N, ) array of time-stamps (in seconds) corresponding with the acceleration
        dt : float
            Sampling time difference
        power_peaks : numpy.ndarray
            Locations of the peaks in the CWT power data.

        Returns
        -------
        sts : dict
            Dictionary of *sit2standpy.Transition* objects containing information about a individual sit-to-stand
            transition. Keys for the dictionary are string timestamps of the start of the transition.
        """
        # find stillness
        acc_still, still_starts, still_stops = _get_still(mag_acc, dt, self.mov_window, self.grav, self.thresh)
        # starts and stops of long still periods
        still_dt = (still_stops - still_starts) * dt  # durations of stillness, in seconds
        lstill_starts = still_starts[still_dt > self.long_still]
        lstill_stops = still_stops[still_dt > self.long_still]

        # find the local minima and maxima in the acceleration signals. Use the reconstructed acceleration for
        # local minima, as this avoids some possible artefacts in the signal
        # acc_lmax, _ = find_peaks(mag_acc, **self.lmax_kw)
        # acc_lmin, _ = find_peaks(-mag_acc_r, **self.lmin_kw)

        # compute an estimate of the direction of gravity, assumed to be the vertical direction
        gfc = butter(self.grav_ord, 2 * self.grav_cut * dt, btype='low')
        vert = filtfilt(gfc[0], gfc[1], raw_acc, axis=0, padlen=None)
        vert /= norm(vert, axis=1, keepdims=True)  # make into unit vectors

        # get an estimate of the vertical acceleration
        v_acc = npsum(vert * raw_acc, axis=1)

        # iterate over the power peaks
        sts = {}

        prev_int_start = -1
        prev_int_stop = -1

        for ppk in power_peaks:
            try:  # look for the preceding end of long stillness
                end_still = still_stops[still_stops < ppk][-1]
                if (time[ppk] - time[end_still]).total_seconds() > 2:  # transition shouldn't be too long
                    raise IndexError
            except IndexError:
                continue
            # try:  # look for the following local min -> local max pattern
            #     n_lmin = acc_lmin[acc_lmin > ppk][0]
            #     # n_lmax = acc_lmax[acc_lmax > n_lmin][0]
            #     # if (time[n_lmax] - time[ppk]).total_seconds() > 2:  # ensure not too far ahead
            #     #     raise IndexError
            # except IndexError:
            #     continue
            try:  # look for the next start of long stillness
                start_still = lstill_starts[lstill_starts > ppk][0]
                if (time[start_still] - time[ppk]).total_seconds() < 30:
                    still_at_end = True
                else:
                    raise IndexError
            except IndexError:
                start_still = int(ppk + (5 / dt))  # try to use a set time after the peak
                still_at_end = False

            # INTEGRATE between the determined indices
            if end_still < prev_int_start or start_still > prev_int_stop:
                v_vel, v_pos = _integrate_acc(v_acc[end_still:start_still] - self.grav, dt, still_at_end)

                # set used limits
                prev_int_start = end_still
                prev_int_stop = start_still

                # zero crossings
                pos_zc = append(0, where(diff(sign(v_vel)) > 0)[0]) + end_still
                neg_zc = append(where(diff(sign(v_vel)) < 0)[0], v_vel.size - 1) + end_still

            # make sure the velocity is high enough to indicate a peak
            if v_vel[ppk - prev_int_start] < self.thresh['transition velocity']:
                continue

            sts_start = end_still

            try:  # find the end of the transition
                sts_end = neg_zc[neg_zc > ppk][0]
            except IndexError:
                continue

            # quatity checks
            if (time[sts_end] - time[sts_start]).total_seconds() > 4.5:
                continue
            if (time[ppk] - time[sts_start]).total_seconds() > self.dur_factor * (time[sts_end]
                                                                                  - time[ppk]).total_seconds():
                continue
            t_start_i = sts_start - prev_int_start  # integrated value start index
            t_end_i = sts_end - prev_int_start  # integrated value end index
            if t_start_i == t_end_i:
                continue
            if (v_pos[t_end_i] - v_pos[t_start_i]) < self.thresh['stand displacement']:
                continue

            # sts assignment
            if len(sts) > 0:
                if (time[sts_start] - sts[list(sts.keys())[-1]].end_time).total_seconds() > 0.4:  # no overlap
                    sts[f'{time[sts_start]}'] = self.tq.quantify((time[sts_start], time[sts_end]), 1 / dt,
                                                                 raw_acc[sts_start:sts_end],
                                                                 mag_acc[sts_start:sts_end],
                                                                 mag_acc_r[sts_start:sts_end],
                                                                 v_vel[t_start_i:t_end_i], v_pos[t_start_i:t_end_i])
            else:
                sts[f'{time[sts_start]}'] = self.tq.quantify((time[sts_start], time[sts_end]), 1 / dt,
                                                             raw_acc[sts_start:sts_end],
                                                             mag_acc[sts_start:sts_end],
                                                             mag_acc_r[sts_start:sts_end],
                                                             v_vel[t_start_i:t_end_i], v_pos[t_start_i:t_end_i])

        # check to ensure no partial transitions
        vd = [sts[i].v_displacement for i in sts]
        vd_high_diff = array(vd) < self.disp_factor * median(vd)
        for elem in array(list(sts.keys()))[vd_high_diff]:
            del sts[elem]

        return sts


class Displacement:
    """
    Method for detecting sit-to-stand transitions based on requiring stillness before a transition, and the
    vertical displacement of a lumbar accelerometer for ensuring a transition.

    Parameters
    ----------
    gravity : float, optional
        Value of gravitational acceleration measured by the accelerometer when still. Default is 9.81 m/s^2.
    thresholds : {None, dict}, optional
        Either None, for the default, or a dictionary of thresholds to change. See *Notes*. Default
        is None, which uses the default values.
    gravity_pass_ord : int, optional
        Low-pass filter order for estimating the direction of gravity by low-pass filtering the raw acceleration
        data. Default is 4.
    gravity_pass_cut : float, optional
        Low-pass filter frequency cutoff for estimating thd direction of gravity. Default is 0.8Hz.
    long_still : float, optional
        Length of time of stillness for it to be qualified as a long period of stillness. Used to determining
        integration limits when available. Default is 0.5s.
    moving_window : float, optional
        Length of the moving window for calculating the moving statistics for determining stillness.
        Default is 0.3s.
    duration_factor : float, optional
        The factor for the maximum difference between the duration before and after the generalized location of
        the sit to stand. Lower factors result in more equal time before and after the detection. Default
        is 10, which effectively removes this constraint.
    displacement_factor : float, optional
        Factor multiplied by the median of the vertical displacements to determine the threshold for checking if a
        transition is a partial transition or a full transition. Default is 0.75
    lmax_kwargs : {None, dict}, optional
        Additional key-word arguments for finding local maxima in the acceleration signal. Default is None,
        for no specified arguments. See `scipy.signal.find_peaks` for possible arguments.
    lmin_kwargs : {None, dict}, optional
        Additional key-word arguments for finding local minima in the acceleration signal. Default is None,
        which specifies a maximum value of 9.5m/s^2 for local minima. See `scipy.signal.find_peaks` for the
        possible arguments.
    trans_quant : TransitionQuantifier
        TransitionQuantifier object, which contains a `quantify` method, which accepts the following arguments:
        `times`, `mag_acc_f`, `mag_acc_r`, `v_vel`, `v_pos`. Only times is required. See
        `sit2standpy.TransitionQuantifier`.

    Notes
    -----
    Default thresholds:
        - stand displacement: 0.125
        - transition velocity: 0.2
        - accel moving avg: 0.2
        - accel moving std: 0.1
        - jerk moving avg: 2.5
        - jerk moving std: 3
    """

    def __init__(self, gravity=9.81, thresholds=None, gravity_pass_ord=4, gravity_pass_cut=0.8, long_still=0.5,
                 moving_window=0.3, duration_factor=10, displacement_factor=0.75, lmax_kwargs=None,
                 lmin_kwargs=None, trans_quant=TQ()):

        # set the default thresholds
        self.default_thresholds = {'stand displacement': 0.125,
                                   'transition velocity': 0.2,
                                   'accel moving avg': 0.2,
                                   'accel moving std': 0.1,
                                   'jerk moving avg': 2.5,
                                   'jerk moving std': 3}
        # assign attributes
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
        self.disp_factor = displacement_factor

        if lmin_kwargs is None:
            self.lmin_kw = {'height': -9.5}
        else:
            self.lmin_kw = lmin_kwargs
        if lmax_kwargs is None:
            self.lmax_kw = {}
        else:
            self.lmax_kw = lmax_kwargs

        self.tq = trans_quant

    def apply(self, raw_acc, mag_acc, mag_acc_r, time, dt, power_peaks):
        """
        Apply the displacement-based STS detection to the given data

        Parameters
        ----------
        raw_acc : numpy.ndarray
            (N, 3) array of the raw acceleration signal
        mag_acc : numpy.ndarray
            (N, ) array of filtered acceleration magnitude.
        mag_acc_r : numpy.ndarray
            (N, ) array of reconstructed acceleration magnitude.
        time : numpy.ndarray
            (N, ) array of time-stamps (in seconds) corresponding with the acceleration
        dt : float
            Sampling time difference
        power_peaks : numpy.ndarray
            Locations of the peaks in the CWT power data.

        Returns
        -------
        sts : dict
            Dictionary of *sit2standpy.Transition* objects containing information about a individual sit-to-stand
            transition. Keys for the dictionary are string timestamps of the start of the transition.
        """
        # find stillness
        acc_still, still_starts, still_stops = _get_still(mag_acc, dt, self.mov_window, self.grav, self.thresh)
        # starts and stops of long still periods
        still_dt = (still_stops - still_starts) * dt  # durations of stillness, in seconds
        lstill_starts = still_starts[still_dt > self.long_still]
        lstill_stops = still_stops[still_dt > self.long_still]

        # find the local minima and maxima in the acceleration signals. Use the reconstructed acceleration for
        # local minima, as this avoids some possible artefacts in the signal
        # acc_lmax, _ = find_peaks(mag_acc, **self.lmax_kw)
        # acc_lmin, _ = find_peaks(-mag_acc_r, **self.lmin_kw)

        # compute an estimate of the direction of gravity, assumed to be the vertical direction
        gfc = butter(self.grav_ord, 2 * self.grav_cut * dt, btype='low')
        vert = filtfilt(gfc[0], gfc[1], raw_acc, axis=0, padlen=None)
        vert /= norm(vert, axis=1, keepdims=True)  # make into unit vectors

        # get an estimate of the vertical acceleration
        v_acc = npsum(vert * raw_acc, axis=1)

        # iterate over the power peaks
        sts = {}

        prev_int_start = -1
        prev_int_stop = -1

        for ppk in power_peaks:
            try:  # look for the preceding end of long stillness
                end_still = lstill_stops[lstill_stops < ppk][-1]
                if (time[ppk] - time[end_still]).total_seconds() > 30:  # don't want to integrate for too long
                    raise IndexError
            except IndexError:
                # end_still = int(ppk - (2.5 / dt))  # try to use a set time before the peak
                continue
            try:  # look for the next start of long stillness
                start_still = lstill_starts[lstill_starts > ppk][0]
                if (time[start_still] - time[ppk]).total_seconds() < 30:
                    still_at_end = True
                else:
                    raise IndexError
            except IndexError:
                start_still = int(ppk + (5 / dt))  # try to use a set time after the peak
                still_at_end = False

            # INTEGRATE between the determined indices
            if end_still < prev_int_start or start_still > prev_int_stop:
                v_vel, v_pos = _integrate_acc(v_acc[end_still:start_still] - self.grav, dt, still_at_end)

                # set used limits
                prev_int_start = end_still
                prev_int_stop = start_still

                # zero crossings
                pos_zc = append(0, where(diff(sign(v_vel)) > 0)[0]) + end_still
                neg_zc = append(where(diff(sign(v_vel)) < 0)[0], v_vel.size - 1) + end_still

            # make sure the velocity is high enough to indicate a peak
            if v_vel[ppk - prev_int_start] < self.thresh['transition velocity']:
                continue
            try:  # look for the previous positive zero crossing as the start of the transition
                sts_start = pos_zc[pos_zc < ppk][-1]
                p_still = still_stops[still_stops < ppk][-1]
                # possibly use the end of stillness if it is close enough to the ZC
                if -0.5 < (dt * (p_still - sts_start)) < 0.7:
                    sts_start = p_still
            except IndexError:
                continue
            try:  # find the end of the transition
                sts_end = neg_zc[neg_zc > ppk][0]
            except IndexError:
                continue

            # quatity checks
            if (time[sts_end] - time[sts_start]).total_seconds() > 4.5:
                continue
            if (time[ppk] - time[sts_start]).total_seconds() > self.dur_factor * (time[sts_end]
                                                                                  - time[ppk]).total_seconds():
                continue
            t_start_i = sts_start - prev_int_start  # integrated value start index
            t_end_i = sts_end - prev_int_start  # integrated value end index
            if t_start_i == t_end_i:
                continue
            if (v_pos[t_end_i] - v_pos[t_start_i]) < self.thresh['stand displacement']:
                continue

            # sts assignment
            if len(sts) > 0:
                if (time[sts_start] - sts[list(sts.keys())[-1]].end_time).total_seconds() > 0.4:  # no overlap
                    sts[f'{time[sts_start]}'] = self.tq.quantify((time[sts_start], time[sts_end]), 1 / dt,
                                                                 raw_acc[sts_start:sts_end],
                                                                 mag_acc[sts_start:sts_end],
                                                                 mag_acc_r[sts_start:sts_end],
                                                                 v_vel[t_start_i:t_end_i], v_pos[t_start_i:t_end_i])
            else:
                sts[f'{time[sts_start]}'] = self.tq.quantify((time[sts_start], time[sts_end]), 1 / dt,
                                                             raw_acc[sts_start:sts_end],
                                                             mag_acc[sts_start:sts_end],
                                                             mag_acc_r[sts_start:sts_end],
                                                             v_vel[t_start_i:t_end_i], v_pos[t_start_i:t_end_i])

        # check to ensure no partial transitions
        vd = [sts[i].v_displacement for i in sts]
        vd_high_diff = array(vd) < self.disp_factor * median(vd)
        for elem in array(list(sts.keys()))[vd_high_diff]:
            del sts[elem]

        return sts


'''
class __Similarity:
    """
    Sit-to-stand (STS) detection based on similarity of summed coefficients of the Continuous Wavelet Transform
    in different power bands

    Parameters
    ----------
    gravity_value : float, optional
        Value of gravitational acceleration of the sensor during still standing or sitting. Default is 9.81m/s^2.
    low_f_band : {array_like, float, int}, optional
        Low frequency limits for the low freq. power band, obtained by summing CWT coefficients in this band of
        frequencies. Can either be a length 2 array_like (min, max), or a number, which will be interpreted as the
        maximum value, with 0 Hz as the minimum. Default is [0, 0.5].
    high_f_band : {array_like, float, int}, optional
        High frequency limits for the high freq. power band, obtained by summing CWT coefficients in this band of
        frequencies. Can either be a length 2 array_like (min, max), or a number, which will be interpreted as the
        maximum value, with 0 Hz as the minimum. Default is [0, 3].
    similarity_atol : {float, int}, optional
        Absolute tolerance for determining similarity between the high and low frequency power bands. Default is 0
    similarity_rtol : {float, int}, optional
        Relative tolerance for determining similarity between the high and low frequency power bands. Default is 0.15
    tr_pk_diff : float, optional
        Minimum difference in acceleration magnitude between troughs and peaks that is used in determining the end
        time for the STS transitions. Default is 0.5 m/s^2.
    start_pos : {'fixed', 'variable'}, optional
        How the start of STS transitions is determined. Either a fixed location, or can be variable among several
        possible locations, and the best location is chosen. Default is 'fixed'
    acc_peak_params : {None, dict}, optional
        Additional parameters (key-word arguments) to be passed to scipy.signal.find_peaks for finding peaks in the
        acceleration magnitude. Default is None, for which the find_peaks defaults will be used.
    acc_trough_params : {None, dict}, optional
        Additional parameters (key-word arguments) to be passed to scipy.signal.find_peaks for finding troughs
        (local minima) in the acceleration magnitude. Default is None, for which the find_peaks defaults will be
        used.
    """

    def __init__(self, gravity_value=9.81, low_f_band=[0, 0.5], high_f_band=[0, 3], similarity_atol=0,
                 similarity_rtol=0.15, tr_pk_diff=0.5, start_pos='fixed', acc_peak_params=None, acc_trough_params=None):

        self.gravity = gravity_value

        if isinstance(low_f_band, (float, int)):
            self.low_f = [0, low_f_band]
        else:
            self.low_f = low_f_band  # TODO add check for length
        if isinstance(high_f_band, (float, int)):
            self.high_f = [0, high_f_band]
        else:
            self.high_f = high_f_band

        self.sim_atol = similarity_atol
        self.sim_rtol = similarity_rtol

        self.tp_diff = tr_pk_diff

        if start_pos == 'fixed' or start_pos == 'variable':
            self.start_pos = start_pos
        else:
            raise ValueError('start_pos must be either "fixed" or "variable".')

        if acc_peak_params is None:
            self.acc_pk_kw = {}
        else:
            self.acc_pk_kw = acc_peak_params
        if acc_trough_params is None:
            self.acc_tr_kw = {}
        else:
            self.acc_tr_kw = acc_trough_params

    def apply(self, raw_acc, mag_acc, mag_acc_r, time, dt, power_peaks, cwt_coefs, cwt_freqs):
        """
        Apply the similarity-based STS detection to the given data

        Parameters
        ----------
        raw_acc : numpy.ndarray
            (N, 3) array of the raw acceleration signal
        mag_acc : numpy.ndarray
            (N, ) array of filtered acceleration magnitude.
        mag_acc_r : numpy.ndarray
            (N, ) array of reconstructed acceleration magnitude.
        time : numpy.ndarray
            (N, ) array of time-stamps (in seconds) corresponding with the acceleration
        dt : float
            Sampling time difference
        power_peaks : numpy.ndarray
            Locations of the peaks in the CWT power data.
        cwt_coefs : numpy.ndarray
            (M, N) array of coefficients for the CWT, where M is the number of scales used in the computation.
        cwt_freqs : numpy.ndarray
            (M, ) array of frequencies from the CWT

        Returns
        -------
        sts : dict
            Dictionary of sit2standpy.Transition objects containing information about a individual sit-to-stand
            transition. Keys for the dictionary are string timestamps of the start of the transition.
        """
        # compute the sum of the scales in certain frequency bands
        low_mask = (cwt_freqs > self.low_f[0]) & (cwt_freqs < self.low_f[1])
        high_mask = (cwt_freqs > self.high_f[0]) & (cwt_freqs < self.high_f[1])

        # compute the powers in those bands
        low_pwr = npsum(cwt_coefs[low_mask, :], axis=0)
        high_pwr = npsum(cwt_coefs[high_mask, :], axis=0)

        # compute the indices where the two power measures are close in value
        similar = isclose(high_pwr, low_pwr, atol=self.sim_atol, rtol=self.sim_rtol)

        # find troughs and peaks in the filtered signal
        acc_pks, _ = find_peaks(mag_acc, **self.acc_pk_kw)
        acc_trs, _ = find_peaks(-mag_acc_r, **self.acc_tr_kw)

        # find the stops in similarity
        stops = where(diff(similar.astype(int)) == -1)[0]
        sim_starts = where(diff(similar.astype(int)) == 1)[0]

        # iterate over the detected power peaks and determine STS locations
        sts = {}
        for ppk in power_peaks:
            # find the first trough after the power peak
            try:
                next_tr = acc_trs[acc_trs > ppk][0]  # get the next trough
            except IndexError:
                continue
            # find the first peak after the found trough, though ensure that there is enough difference that it is
            # not just artefact
            try:
                next_pk = acc_pks[acc_pks > next_tr][0]
                if mag_acc[next_pk] - mag_acc[next_tr] < self.tp_diff:
                    next_pk = acc_pks[acc_pks > next_tr][1]
            except IndexError:
                continue
            # make sure ppk to stop isn't too long
            if (time[next_pk] - time[ppk]).total_seconds() > 2:
                continue
            # find the second previous stop of similarity in the power bands
            try:
                prev2_stop = stops[stops < ppk][-2]
                if self.start_pos == 'variable':
                    prev_start = sim_starts[sim_starts > prev2_stop][0]
                    if npabs(mag_acc[prev2_stop] - self.gravity) < npabs(mag_acc[prev_start] - self.gravity):
                        start = prev2_stop
                        alt_start = prev_start
                    else:
                        start = prev_start
                        alt_start = None
                else:
                    start = prev2_stop
                    alt_start = None
            except IndexError:
                continue
            # ensure that there is no overlap with previously detected transitions
            if len(sts) > 0:
                if (time[start] - sts[list(sts.keys())[-1]][1]).total_seconds() < 0.5:  # min time between transitions
                    if alt_start is not None:
                        if (time[alt_start] - sts[list(sts.keys())[-1]][1]) < 0.5:
                            continue
                        else:
                            start = alt_start
            # sts.append((time[start], time[next_pk]))
            a_max, a_min = mag_acc_r[start:next_pk].max(), mag_acc_r[start:next_pk].min()
            sts[f'{time[start]}'] = Transition((time[start], time[next_pk]), max_acceleration=a_max,
                                               min_acceleration=a_min)

        return sts
'''
