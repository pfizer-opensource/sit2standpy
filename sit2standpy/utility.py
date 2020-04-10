"""
Common methods for both acceleration only and imu-based postural transition detection

Lukas Adamowicz
June 2019
"""
from numpy import ndarray, zeros, mean, std, ceil, around, gradient, abs, where, diff, insert, append
from numpy.lib import stride_tricks


class Transition:
    """
    Object for storing information about a postural transition

    Parameters
    ----------
    times : array_like
        array_like of start and end timestamps (pandas.Timestamp), [start_time, end_time]. Duration will be
        calculated as the difference.
    t_type : {'SiSt', 'StSi'}, optional
        Transition type, either 'SiSt' for sit-to-stand, or 'StSi' for stand-to-sit. Default is 'SiSt'.
    v_displacement : {float, None}, optional
        Vertical displacement during the transition, or None. Default is None.
    max_v_velocity : {float, None}, optional
        Maximum vertical velocity during the transition, or None. Default is None.
    min_v_velocity : {float, None}, optional
        Minimum vertical velocity during the transition, or None. Default is None.
    max_acceleration : {float, None}, optional
        Maximum acceleration during the transition, or None. Default is None.
    min_acceleration : {float, None}, optional
        Minimum acceleration during the transition, or None. Default is None.
    sparc : {float, None}, optional
        SPectral ARC length parameter, measuring the smoothness of the transition, or None. Default is None.

    Attributes
    ----------
    times : tuple
        Tuple of start and end times.
    start_time : pandas.Timestamp
        Start timestamp of the transition.
    end_time : pandas.Timestamp
        End timestamp of the transition.
    duration : float
        Duration of the transition in seconds.
    ttype : str
        Short transition type name.
    long_type : str
        Full transition type name.
    v_displacement : {float, None}
        Vertical displacement.
    max_v_velocity : {float, None}
        Maximum vertical velocity.
    min_v_velocity : {float, None}
        Minimum vertical velocity.
    max_acceleration : {float, None}
        Maximum acceleration.
    min_acceleration : {float, None}
        Minimum acceleration.
    sparc : {float, None}
        SPectral ARC length measure of smoothness.
    """
    def __str__(self):
        return f'Postural Transition'

    def __repr__(self):
        return f'{self.long_type} (Duration: {self.duration:.2f})'

    def __init__(self, times, t_type='SiSt', v_displacement=None, max_v_velocity=None, min_v_velocity=None,
                 max_acceleration=None, min_acceleration=None, sparc=None):
        if isinstance(times, (tuple, list, ndarray)):
            if times[1] < times[0]:
                raise ValueError('End time cannot be before start time.')

            self.start_time = times[0]
            self.end_time = times[1]
            self.duration = (self.end_time - self.start_time).total_seconds()
        else:
            raise ValueError('times must be a tuple or a list-like.')

        if self.duration > 15:
            raise ValueError('Transitions should not be longer than 15s.')

        self.ttype = t_type
        if self.ttype == 'SiSt':
            self.long_type = 'Sit to Stand'
        elif self.ttype == 'StSi':
            self.long_type = 'Stand to Sit'
        else:
            raise ValueError('Unrecognized transition type (t_type). Must be either "SiSt" or "StSi".')

        self.v_displacement = v_displacement
        self.max_v_velocity = max_v_velocity
        self.min_v_velocity = min_v_velocity
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        self.sparc = sparc


def mov_stats(seq, window):
    """
    Compute the centered moving average and standard deviation.

    Parameters
    ----------
    seq : numpy.ndarray
        Data to take the moving average and standard deviation on.
    window : int
        Window size for the moving average/standard deviation.

    Returns
    -------
    m_mn : numpy.ndarray
        Moving average
    m_st : numpy.ndarray
        Moving standard deviation
    pad : int
        Padding at beginning of the moving average and standard deviation
    """

    def rolling_window(x, wind):
        if not x.flags['C_CONTIGUOUS']:
            raise ValueError("Data must be C-contiguous to be able to window for moving statistics")
        shape = x.shape[:-1] + (x.shape[-1] - wind + 1, wind)
        strides = x.strides + (x.strides[-1],)
        return stride_tricks.as_strided(x, shape=shape, strides=strides)

    m_mn = zeros(seq.shape)
    m_st = zeros(seq.shape)

    if window < 2:
        window = 2

    pad = int(ceil(window / 2))

    rw_seq = rolling_window(seq, window)

    n = rw_seq.shape[0]

    m_mn[pad:pad + n] = mean(rw_seq, axis=-1)
    m_st[pad:pad + n] = std(rw_seq, axis=-1, ddof=1)

    m_mn[:pad], m_mn[pad + n:] = m_mn[pad], m_mn[-pad - 1]
    m_st[:pad], m_st[pad + n:] = m_st[pad], m_st[-pad - 1]
    return m_mn, m_st, pad


def get_stillness(filt_accel, dt, window, gravity, thresholds):
    """
    Stillness determination based on filtered acceleration magnitude and jerk magnitude

    Parameters
    ----------
    filt_accel : numpy.ndarray
        1D array of filtered magnitude of acceleration data, units of m/s^2
    dt : float
        Sampling time, in seconds
    window : float
        Moving statistics window length, in seconds
    gravity : float
        Gravitational acceleration, as measured by the sensor during static periods.
    thresholds : dict
        Dictionary of the 4 thresholds to be used:
        - accel moving avg
        - accel moving std
        - jerk moving avg
        - jerk moving std
        Acceleration average thresholds should be for difference from gravitional acceleration.

    Returns
    -------
    still : numpy.ndarray
        (N, ) boolean array of stillness (True)
    starts : numpy.ndarray
        (Q, ) array of indices where stillness starts. Includes index 0 if still[0] is True. Q < (N/2)
    stops : numpy.ndarray
        (Q, ) array of indices where stillness ends. Includes index N-1 if still[-1] is True. Q < (N/2)
    """
    # compute the sample window length from the time value
    n_window = int(around(window / dt))
    # compute the acceleration moving stats
    acc_rm, acc_rsd, _ = mov_stats(filt_accel, n_window)
    # compute the jerk
    jerk = gradient(filt_accel, dt, edge_order=2)
    # compute the jerk moving stats
    jerk_rm, jerk_rsd, _ = mov_stats(jerk, n_window)

    # create the stillness masks
    arm_mask = abs(acc_rm - gravity) < thresholds['accel moving avg']
    arsd_mask = acc_rsd < thresholds['accel moving std']
    jrm_mask = abs(jerk_rm) < thresholds['jerk moving avg']
    jrsd_mask = jerk_rsd < thresholds['jerk moving std']

    still = arm_mask & arsd_mask & jrm_mask & jrsd_mask
    starts = where(diff(still.astype(int)) == 1)[0]
    stops = where(diff(still.astype(int)) == -1)[0]

    if still[0]:
        starts = insert(starts, 0, 0)
    if still[-1]:
        stops = append(stops, len(still) - 1)

    return still, starts, stops
